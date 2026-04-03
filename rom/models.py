import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def build_custom_attention_mask(seq_len, user_seq_len, device='cpu'):
    i = torch.arange(seq_len, device=device).view(-1, 1)
    j = torch.arange(seq_len, device=device).view(1, -1)
    causal_mask = (i >= user_seq_len) & (i < j)
    user_to_assistant_mask = (i < user_seq_len) & (j >= user_seq_len)
    mask = causal_mask | user_to_assistant_mask
    return mask


class AttentionLayer(nn.Module):
    def __init__(self, input_dim=3584, hidden_dim=128):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.register_buffer("causal_mask", torch.triu(torch.ones(1, 8192, 8192), diagonal=1).bool())

    def forward(self, x, user_seq_len=None):
        batch_size, seq_len, _ = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / ((K.size(-1) + 1e-6) ** 0.5)

        if user_seq_len is None:
            if seq_len <= self.causal_mask.size(1):
                mask = self.causal_mask[:, :seq_len, :seq_len]
            else:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool().unsqueeze(0)
        else:
            mask = build_custom_attention_mask(seq_len, user_seq_len, x.device)
        attention_scores = attention_scores.masked_fill(mask, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, V)

        return context_vector, attention_weights


class CfcCell(nn.Module):
    """Continuous-time Fluid Convolution Cell - a GRU-like cell with time-step scaling."""

    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wz = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Uz = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bz = nn.Parameter(torch.zeros(hidden_dim))

        self.Wr = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Ur = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.br = nn.Parameter(torch.zeros(hidden_dim))

        self.Wh = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Uh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]:
            torch.nn.init.orthogonal_(weight)

    def forward(self, x, h_prev, dt):
        device = x.device
        h_prev = h_prev.to(device)
        dt = dt.to(device) if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=device, dtype=x.dtype)

        z = torch.sigmoid(torch.matmul(x, self.Wz) + torch.matmul(h_prev, self.Uz) + self.bz)
        r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_prev, self.Ur) + self.br)
        h_hat = torch.tanh(torch.matmul(x, self.Wh) + torch.matmul(r * h_prev, self.Uh) + self.bh)
        h_new = (1 - z) * h_prev + z * h_hat

        return h_new + dt * (h_new - h_prev)


@torch.jit.script
def _cfc_loop_jit(
    feat: torch.Tensor,
    h: torch.Tensor,
    dt_tensor: torch.Tensor,
    offsets: torch.Tensor,
    alens: torch.Tensor,
    max_t: int,
    Wz: torch.Tensor, Uz: torch.Tensor, bz: torch.Tensor,
    Wr: torch.Tensor, Ur: torch.Tensor, br: torch.Tensor,
    Wh: torch.Tensor, Uh: torch.Tensor, bh: torch.Tensor,
    mem_w: torch.Tensor, mem_b: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled CFC loop: eliminates Python interpreter overhead."""
    B = feat.size(0)
    batch_idx = torch.arange(B, device=feat.device)
    logits_list: List[torch.Tensor] = []
    for t in range(max_t):
        active = (t < alens)
        if not active.any():
            break
        idx = torch.clamp(offsets + t, max=feat.size(1) - 1)
        x_t = feat[batch_idx, idx]
        x_t = x_t * active.unsqueeze(-1).to(x_t.dtype)
        dt = dt_tensor[:, t].unsqueeze(-1)
        z = torch.sigmoid(x_t @ Wz + h @ Uz + bz)
        r = torch.sigmoid(x_t @ Wr + h @ Ur + br)
        h_hat = torch.tanh(x_t @ Wh + (r * h) @ Uh + bh)
        h_new = (1 - z) * h + z * h_hat
        h_new = h_new + dt * (h_new - h)
        h = torch.where(active.unsqueeze(-1), h_new, h)
        logits_list.append((h @ mem_w.t() + mem_b).unsqueeze(1))
    return torch.cat(logits_list, dim=1)


class StreamingHead(nn.Module):
    """Lightweight safety classifier that attaches to a frozen base model.

    Extracts features via attention from transformer hidden states, then uses
    a CfcCell to process assistant response tokens sequentially, outputting
    per-token binary classification (0=normal, 1=overthinking).
    """

    def __init__(self, input_dim=4096, proj_dim=512, mem_dim=512,
                 num_labels=1, use_dt=False, dt=1.0, dropout=0.1, cfc=True):
        super().__init__()
        self.attention = AttentionLayer(input_dim, proj_dim)
        self.cfc_enabled = cfc

        d_in = proj_dim
        self.cfc = CfcCell(d_in, mem_dim) if cfc else None
        self.mem_head = nn.Linear(mem_dim if cfc else d_in, num_labels)

        self.prefix_to_h = nn.Sequential(
            nn.Linear(d_in, mem_dim),
            nn.Tanh()
        )
        self.prefix_scorer = nn.Linear(d_in, 1, bias=False)

        self._h = None  # (B, mem_dim)

    def reset_state(self, batch_size=None, device=None, dtype=None):
        self._h = None
        if batch_size is not None and self.cfc_enabled:
            if device is None:
                device = next(self.parameters()).device
            if dtype is None:
                dtype = next(self.parameters()).dtype
            self._h = torch.zeros(batch_size, self.cfc.hidden_dim, device=device, dtype=dtype)

    def _ensure_state(self, x):
        if self._h is None or self._h.size(0) != x.size(0) or self._h.device != x.device or self._h.dtype != x.dtype:
            self.reset_state(batch_size=x.size(0), device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def init_with_prefix(self, assist_lens, user_hidden_list):
        """Initialize hidden state from user prefix embeddings.

        Args:
            assist_lens: List[int] - assistant token count per sample
            user_hidden_list: List[Tensor] - prefix hidden states, each (U_i, D)
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        B = len(assist_lens)

        self.reset_state(batch_size=B, device=device, dtype=dtype)
        self._dt_list = []

        h_list = []
        for assist_len, user_hidden in zip(assist_lens, user_hidden_list):
            times = torch.linspace(0, 1, steps=assist_len, device=device, dtype=dtype)
            dt = torch.zeros_like(times)
            dt[1:] = times[1:] - times[:-1]
            self._dt_list.append(dt)

            scores = self.prefix_scorer(user_hidden.unsqueeze(0)).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.bmm(weights.unsqueeze(1), user_hidden.unsqueeze(0)).squeeze(1)
            h_i = self.prefix_to_h(pooled).squeeze(0)
            h_list.append(h_i)

        self._h = torch.stack(h_list, dim=0)

    def step(self, x_t, t_indices, active_mask):
        """Process a single timestep for streaming inference.

        Args:
            x_t: (B, D) - input features
            t_indices: (B,) - current time step per sample
            active_mask: (B,) - which samples are still being processed
        """
        self._ensure_state(x_t)

        device = x_t.device
        dtype = x_t.dtype
        dt_values = torch.zeros(x_t.size(0), device=device, dtype=dtype)
        for i in range(x_t.size(0)):
            if active_mask[i] and hasattr(self, '_dt_list') and t_indices[i] < len(self._dt_list[i]):
                dt_values[i] = self._dt_list[i][t_indices[i]]

        h_new = self.cfc(x_t, self._h, dt_values.unsqueeze(-1))
        self._h = torch.where(active_mask.unsqueeze(-1), h_new, self._h)
        mem_logits = self.mem_head(self._h)

        return mem_logits

    def forward(self, x, assistant_start, is_multi=False):
        """
        Args:
            x: (B, seq_total, hidden) - full sequence hidden states
            assistant_start: int, List[int], or Tensor - where assistant tokens begin
        """
        feat = self.attention(x)[0]
        B, seq_total, D = feat.shape

        if isinstance(assistant_start, int):
            assistant_starts = [assistant_start] * B
        elif isinstance(assistant_start, torch.Tensor):
            assistant_starts = assistant_start.tolist()
        else:
            assistant_starts = list(assistant_start)

        assist_lens = [seq_total - s for s in assistant_starts]
        max_assist_len = min(max(assist_lens), 4096)

        if not self.cfc_enabled:
            assistant_feat = feat[:, assistant_starts[0]:, :]
            logits = self.mem_head(assistant_feat)
            return logits

        device = feat.device
        dtype = feat.dtype

        # Vectorized prefix initialization (no_grad matches original init_with_prefix)
        max_prefix_len = max(assistant_starts)
        with torch.no_grad():
            if max_prefix_len > 0:
                prefix_padded = torch.zeros(B, max_prefix_len, D, device=device, dtype=dtype)
                prefix_mask = torch.zeros(B, max_prefix_len, device=device, dtype=torch.bool)
                for i in range(B):
                    plen = assistant_starts[i]
                    prefix_padded[i, :plen] = feat[i, :plen]
                    prefix_mask[i, :plen] = True
                scores = self.prefix_scorer(prefix_padded).squeeze(-1)
                scores = scores.masked_fill(~prefix_mask, float('-inf'))
                weights = torch.softmax(scores, dim=1)
                pooled = torch.bmm(weights.unsqueeze(1), prefix_padded).squeeze(1)
                h_init = self.prefix_to_h(pooled)
            else:
                h_init = torch.zeros(B, self.cfc.hidden_dim, device=device, dtype=dtype)

        # Pre-compute dt tensor
        dt_tensor = torch.zeros(B, max_assist_len, device=device, dtype=dtype)
        for i in range(B):
            alen = assist_lens[i]
            if alen > 1:
                dt_val = 1.0 / (alen - 1)
                dt_tensor[i, 1:alen] = dt_val

        offsets = torch.tensor(assistant_starts, device=device, dtype=torch.long)
        alens_t = torch.tensor(assist_lens, device=device, dtype=torch.long)

        # JIT-compiled CFC loop (eliminates Python loop overhead)
        logits = _cfc_loop_jit(
            feat, h_init, dt_tensor, offsets, alens_t, max_assist_len,
            self.cfc.Wz, self.cfc.Uz, self.cfc.bz,
            self.cfc.Wr, self.cfc.Ur, self.cfc.br,
            self.cfc.Wh, self.cfc.Uh, self.cfc.bh,
            self.mem_head.weight, self.mem_head.bias,
        )
        return logits


class Qwen3WithHead(torch.nn.Module):
    """Qwen3 wrapper with inline safety head. No need to modify transformers.

    Usage:
        model = Qwen3WithHead.from_pretrained("Qwen/Qwen3-8B", ckpt_path="best.pt")
        model.generate(..., with_head=True)
    """

    IDX_LAYER = 20

    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.head = head

    @classmethod
    def from_pretrained(cls, model_name, ckpt_path=None, **model_kwargs):
        from transformers import Qwen3ForCausalLM

        base_model = Qwen3ForCausalLM.from_pretrained(model_name, **model_kwargs)
        input_dim = base_model.lm_head.in_features
        head = StreamingHead(
            input_dim=input_dim, proj_dim=1024, mem_dim=1024, num_labels=2,
        )

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            if "attention.causal_mask" in state_dict:
                cur = head.attention.causal_mask.shape
                if state_dict["attention.causal_mask"].shape != cur:
                    state_dict = {k: v for k, v in state_dict.items() if k != "attention.causal_mask"}
            head.load_state_dict(state_dict, strict=False)

        head.to(device=base_model.device, dtype=base_model.dtype)
        return cls(base_model, head)

    def generate(self, *args, with_head=False, **kwargs):
        if with_head:
            kwargs["output_hidden_states"] = True
            kwargs["return_dict_in_generate"] = True

        outputs = self.base_model.generate(*args, **kwargs)

        if with_head:
            for step_hidden in outputs.hidden_states[1:]:  # skip prompt
                feat = step_hidden[self.IDX_LAYER]
                feat = self.head.attention(feat)[0]
                self.head.step(
                    feat[:, -1, :],
                    torch.tensor([0], device=feat.device),
                    torch.tensor([True], device=feat.device),
                )

        return outputs

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
