import os
import json
import torch
import glob
import tempfile
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from typing import Dict, Any, List

def collate_fn(batch):
    """Pad variable-length embeddings and labels to create a proper batch."""
    max_seq = max(item["embeddings"].size(0) for item in batch)
    B = len(batch)
    D = batch[0]["embeddings"].size(-1)

    embeddings = torch.zeros(B, max_seq, D, dtype=batch[0]["embeddings"].dtype)
    assistant_starts = []
    labels_list = []

    for i, item in enumerate(batch):
        seq_len = item["embeddings"].size(0)
        embeddings[i, :seq_len] = item["embeddings"]
        assistant_starts.append(item["assistant_start"])
        if "labels" in item:
            labels_list.append(item["labels"])

    result = {"embeddings": embeddings, "assistant_start": assistant_starts}
    if labels_list:
        max_lbl = max(l.size(0) for l in labels_list)
        labels = torch.full((B, max_lbl), -100, dtype=torch.long)
        for i, l in enumerate(labels_list):
            labels[i, :l.size(0)] = l
        result["labels"] = labels
    return result


class LengthBucketSampler(Sampler):
    """Sampler that groups samples of similar sequence length to minimize padding."""

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Sort indices by sequence length
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        # Group into chunks of batch_size
        batches = [sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(sorted_indices), self.batch_size)]

        # Shuffle chunk order each epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def find_sequence(lst, seq):
    n = len(seq)
    for i in range(len(lst) - n + 1):
        if lst[i:i+n] == seq:
            return i
    return -1

class DatasetFromJSONL(Dataset):
    """Load data from jsonl file for overthinking detection.
    
    Labels are read directly from the 'label' field in each split_solution:
        - label=0: correct/non-overthinking tokens
        - label=1: overthinking tokens
    """
    
    def __init__(self, dataset_dir, model_name, tokenizer, base_model, idx_layer=32, max_length=8192, 
                 device="cpu", build_cache_if_missing=True, overwrite=False, max_build_samples=None, 
                 start_build_idx=0, efficient_data=None, overthinking_data=None, eval_mode=False):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.idx_layer = idx_layer
        self.max_length = max_length
        self.device = device
        self.efficient_data = efficient_data
        self.overthinking_data = overthinking_data
        self.eval_mode = eval_mode
            
        # Auto-detect assistant token based on model (will use tokenizer if available)
        self.assistant_tokens = self._detect_assistant_token(model_name, tokenizer)
        self.assistant_end = -1
        # Cache name from provided files
        files = [f for f in [overthinking_data, efficient_data] if f]
        assert len(files) > 0, "At least one of efficient_data or overthinking_data must be provided"
        cache_name = "_".join([os.path.splitext(os.path.basename(f))[0] for f in files])
        self.cache_dir = os.path.join(
            dataset_dir,
            f"safety_cache/{model_name.replace('/', '-')}/{cache_name}_idx{idx_layer}_maxlength{max_length}"
        )
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Count total samples from data files
        total_samples = 0
        for path in [self.overthinking_data, self.efficient_data]:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    total_samples += sum(1 for _ in f)
        
        # Check if cache is complete
        existing_cache_files = glob.glob(os.path.join(self.cache_dir, "sample_*.pt"))
        cache_is_complete = len(existing_cache_files) == total_samples
        
        # Build cache if:
        # 1. build_cache_if_missing=True AND cache is incomplete (normal build or resume)
        # 2. max_build_samples is specified (batch building mode, always try to build)
        need_build = build_cache_if_missing and (not cache_is_complete or max_build_samples is not None)
        
        if need_build:
            assert tokenizer is not None and base_model is not None, "Building cache requires tokenizer and base_model."
            self._build_cache_per_sample(tokenizer, base_model, overwrite, max_build_samples, start_build_idx)
        
        self.files = sorted(glob.glob(os.path.join(self.cache_dir, "sample_*.pt")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No cached samples found in {self.cache_dir}")

        # Load per-sample lengths for bucketing
        self.lengths = self._load_lengths()
    
    def _detect_assistant_token(self, model_name, tokenizer=None):
        """Detect assistant token based on model type"""
        model_lower = model_name.lower()
        # Check DeepSeek models first (including DeepSeek-R1-Distill-Qwen)
        if 'deepseek' in model_lower:
            token = '<｜Assistant｜>'  # DeepSeek uses full-width vertical bar (｜not |)
        elif 'glm' in model_lower:
            token = '<|assistant|>\n'
        elif 'qwen' in model_lower or 'qwq' in model_lower:
            token = '<|im_start|>assistant\n'
        else:
            token = '<|im_start|>assistant\n'
        
        print(f"Using assistant token for {model_name}: {repr(token)}")
        return token
    
    def _load_lengths(self) -> List[int]:
        """Load seq_total_len for each sample from lengths.json, falling back to .pt files."""
        lengths_path = os.path.join(self.cache_dir, "lengths.json")
        len_map: Dict[str, int] = {}
        if os.path.exists(lengths_path):
            with open(lengths_path, 'r') as f:
                raw = json.load(f)
            for fname, info in raw.items():
                if "seq_total_len" in info:
                    len_map[fname] = info["seq_total_len"]

        lengths = []
        for fpath in self.files:
            fname = os.path.basename(fpath)
            if fname in len_map:
                lengths.append(len_map[fname])
            else:
                obj = torch.load(fpath, map_location="cpu")
                lengths.append(obj["seq_total_len"])
        return lengths

    def _build_cache_per_sample(self, tokenizer, base_model, overwrite=False, max_build_samples=None, start_build_idx=0):
        print(f"Building per-sample cache into {self.cache_dir} ...")
        
        # Load file(s)
        file_paths = [f for f in [self.overthinking_data, self.efficient_data] if f]
        data = []
        for path in file_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    file_data = [json.loads(line) for line in f]
                    data.extend(file_data)
                    print(f"Loaded {len(file_data)} samples from {path}")
        
        if not data:
            raise FileNotFoundError(f"No data files found. Expected: {file_paths}")
        
        # Calculate range to process
        start_idx = start_build_idx
        end_idx = len(data) if max_build_samples is None else min(start_idx + max_build_samples, len(data))
        
        print(f"Total {len(data)} samples loaded")
        print(f"Processing samples {start_idx} to {end_idx - 1} ({end_idx - start_idx} samples)")
    
        # Store per-sample lengths for efficient bucketing at training time
        lengths: Dict[str, Any] = {}

        base_model.eval()
        with torch.no_grad():
            for i in tqdm(range(start_idx, end_idx), desc="Build samples"):
                sample_path = os.path.join(self.cache_dir, f"sample_{i:08d}.pt")
                if (not overwrite) and os.path.exists(sample_path):
                    continue
    
                info = data[i]
                prompt = info['problem']
                response = info['response']
                
                # Build messages
                messages = [{'role':'user', 'content': prompt}, {'role':'assistant', 'content': response}]
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    max_length=self.max_length, 
                    truncation=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(self.device)

                output = base_model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = output.hidden_states[self.idx_layer]
    
                assistant_ids = tokenizer.encode(self.assistant_tokens, add_special_tokens=False)
                assistant_start = find_sequence(model_inputs.input_ids[0].tolist(), assistant_ids) + len(assistant_ids)
    
                seq_len = model_inputs.input_ids[:, assistant_start:self.assistant_end].shape[-1]
                if seq_len <= 0:
                    continue
                
                embedding_cpu = hidden_states[0, :self.assistant_end, :].detach().cpu().contiguous()
                
                # Read labels only if not in eval_mode
                if not self.eval_mode:
                    labels = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
                    if 'split_solutions' in info:
                        token_idx = 0
                        for sol in info['split_solutions']:
                            sol_label = sol.get('label', 0)
                            num_tokens = len(tokenizer.encode(sol['solution'], add_special_tokens=False))
                            end_idx = min(token_idx + num_tokens, seq_len)
                            labels[0, token_idx:end_idx] = sol_label
                            token_idx = end_idx
                            if token_idx >= seq_len:
                                break
                    labels_cpu = labels[0].detach().cpu().contiguous()
                    payload = {
                        "embeddings": embedding_cpu,
                        "assistant_start": int(assistant_start),
                        "labels": labels_cpu,
                        "assist_len": int(seq_len),
                        "seq_total_len": int(embedding_cpu.shape[0]),
                    }
                else:
                    payload = {
                        "embeddings": embedding_cpu,
                        "assistant_start": int(assistant_start),
                        "assist_len": int(seq_len),
                        "seq_total_len": int(embedding_cpu.shape[0]),
                    }
                tmp_fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir)
                os.close(tmp_fd)
                torch.save(payload, tmp_path)
                os.replace(tmp_path, sample_path)

                # Track lengths keyed by filename (robust to skipped indices)
                lengths[os.path.basename(sample_path)] = {
                    "assist_len": int(seq_len),
                    "seq_total_len": int(embedding_cpu.shape[0]),
                }

        # Write lengths index for fast bucketing
        lengths_path = os.path.join(self.cache_dir, "lengths.json")
        try:
            with open(lengths_path, "w") as f:
                json.dump(lengths, f)
        except Exception as e:
            print(f"Warning: failed to write lengths.json: {e}")

        print(f"Cache build finished at {self.cache_dir}")
  
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        obj = torch.load(self.files[idx], map_location="cpu")
        embeddings = obj["embeddings"]
        assistant_start = obj['assistant_start']
        result = {"embeddings": embeddings, "assistant_start": assistant_start}
        if "labels" in obj:
            result["labels"] = torch.as_tensor(obj["labels"], dtype=torch.long)
        return result

