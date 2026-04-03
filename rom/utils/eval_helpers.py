"""Utility functions for evaluation: probability computation, prompt
preparation, metrics calculation, and logging."""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rom.utils.math import extract_answer, check_answer_correctness
from rom.models import StreamingHead
from rom.dataset import DatasetFromJSONL
from rom.env import set_seed


def compute_probs(ckpt_path, test_dataset_dir, test_jsonl_filename, model_name, idx_layer,
                  max_length, bf16=True, device=None, seed=42, cache_dir="eval_cache",
                  cache_file=None, debug=False):
    """Compute probabilities for test data. Returns cache file path if successful, None otherwise."""
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Computing probabilities for test data...")
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if bf16 else torch.float16, device_map=device,
        trust_remote_code=True
    )
    base_model.eval()
    
    # Load test data
    jsonl_path = os.path.join(test_dataset_dir, test_jsonl_filename)
    print(f"Loading test data from {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    if debug:
        test_data = test_data[:2]
        print(f"DEBUG MODE: Processing only first 2 samples")
    
    # Load dataset for cached embeddings
    test_dataset = DatasetFromJSONL(
        dataset_dir=test_dataset_dir, model_name=model_name, tokenizer=tokenizer,
        base_model=base_model, idx_layer=idx_layer, max_length=max_length,
        device=device, build_cache_if_missing=True, overwrite=False,
        efficient_data=jsonl_path, eval_mode=True
    )
    
    # Load safety head
    print(f"Loading safety head from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfc = checkpoint.get('cfc', checkpoint.get('use_cfc', True))
    
    head = StreamingHead(
        input_dim=base_model.config.hidden_size, proj_dim=1024, mem_dim=1024,
        num_labels=2, use_dt=True, cfc=cfc
    )
    
    # Load state dict, but allow causal_mask size mismatch (it's not a trainable parameter)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Filter out causal_mask if there's a size mismatch - it will be regenerated
    if 'attention.causal_mask' in state_dict:
        current_mask_size = head.attention.causal_mask.shape
        checkpoint_mask_size = state_dict['attention.causal_mask'].shape
        if current_mask_size != checkpoint_mask_size:
            print(f"Note: Ignoring causal_mask from checkpoint (size {checkpoint_mask_size}) - using current size {current_mask_size}")
            state_dict = {k: v for k, v in state_dict.items() if k != 'attention.causal_mask'}
    
    head.load_state_dict(state_dict, strict=False)
    head.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float32)
    head.eval()
    
    num_samples = min(len(test_data), len(test_dataset.files))
    print(f"Computing probabilities for {num_samples} samples...")
    
    # Use provided cache_file or generate default filename
    if cache_file is None:
        ckpt_basename = os.path.splitext(os.path.basename(ckpt_path))[0].replace('model_', '', 1)
        test_data_basename = os.path.splitext(test_jsonl_filename)[0].split('.')[0]
        cache_file = os.path.join(cache_dir, f"test_probs_{test_data_basename}_{ckpt_basename}.json")
    
    cached_results = []
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Computing probabilities"):
            item = test_data[idx]
            
            # Load cached embeddings
            cached = torch.load(test_dataset.files[idx], map_location="cpu")
            embeddings = cached["embeddings"].to(device).unsqueeze(0)
            assistant_start = int(cached["assistant_start"])
            
            # Run safety head
            logits = head(embeddings, assistant_start)
            probs = F.softmax(logits.squeeze(0).float(), dim=-1)[:, 1].cpu().numpy()
            
            cached_results.append({
                'sample_idx': idx,
                'problem': item['problem'],
                'response': item.get('response', ''),
                'expected_answer': item.get('expected_answer'),
                'probs': probs.tolist(),
                'logits': logits.squeeze(0).cpu().tolist(),
                'assistant_start': assistant_start,
                'split_solutions': item.get('split_solutions', []),
            })
    
    # Save cached results
    with open(cache_file, 'w') as f:
        json.dump(cached_results, f)
    
    print(f"Probabilities saved to {cache_file}")
    
    # Clean up GPU memory before returning
    del base_model
    del head
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared after computing probabilities")
    
    return cache_file



def find_reasoning_length(token_ids, think_end_token):
    """Find the position of the last </think> token."""
    for i in range(len(token_ids) - len(think_end_token), -1, -1):
        if token_ids[i:i+len(think_end_token)] == think_end_token:
            return i + len(think_end_token)
    return len(token_ids)


def calculate_lengths_and_correctness(response_text, full_token_ids, assistant_start, tokenizer, problem, expected_answer, cut_diff=0, cut_idx=None):
    """Calculate response/reasoning lengths and check correctness."""
    think_end_token = tokenizer.encode("</think>", add_special_tokens=False)
    assistant_tokens = full_token_ids[assistant_start:]
    
    # Tokenize response
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    response_length = len(response_tokens)
    reasoning_length = find_reasoning_length(response_tokens, think_end_token)
    
    # Adjust lengths based on cut_diff
    if cut_diff != 0 and cut_idx is not None:
        original_reasoning_end = find_reasoning_length(assistant_tokens, think_end_token)
        # If cut position is before </think> in original, add cut_diff to both reasoning and response
        # Otherwise only add to response
        if cut_idx < original_reasoning_end:
            reasoning_length += cut_diff
        response_length += cut_diff
    
    # Check correctness
    extracted_answer = extract_answer(response_text, problem=problem)
    is_correct = check_answer_correctness(problem, response_text, expected_answer)
    
    return {
        'response_length': response_length,
        'reasoning_length': reasoning_length,
        'extracted_answer': extracted_answer,
        'is_correct': is_correct,
        'original_response_length': len(assistant_tokens),
        'original_reasoning_length': find_reasoning_length(assistant_tokens, think_end_token)
    }


def safe_ratio(a, b, default=1.0):
    """Calculate ratio safely, returning default if denominator is 0."""
    return a / b if b > 0 else default


def create_log_print(log_file):
    """Create a logging function that prints to console and optionally to file."""
    def log_print(text):
        # print(text)
        if log_file:
            log_file.write(text + '\n')
            log_file.flush()
    return log_print


def prepare_prompts_for_checkpoint(test_data, tokenizer, device,
                                    max_prompt_len, summarization_prompt, log_print,
                                    use_backtrack=True, threshold=0.5):
    """Prepare generation prompts for a single checkpoint.

    Args:
        use_backtrack: If True, backtrack to sentence boundary before cutting.
                       If False, cut directly at first overthinking token.
    """
    generation_prompts = []
    prompt_indices = []
    all_metadata = []

    for idx in tqdm(range(len(test_data)), desc="Preparing prompts"):
        item = test_data[idx]
        problem = item['problem']
        response = item.get('response', '')
        assistant_start = item['assistant_start']

        # Use cached logits
        logits = torch.tensor(item['logits'], device=device)

        # Tokenize full text
        messages = [{"role": "user", "content": problem}, {"role": "assistant", "content": response}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer([text], return_tensors="pt")
        full_token_ids = model_inputs.input_ids[0].tolist()

        # Get probabilities
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().float().numpy()  # P(class=1)

        # Find first overthinking token
        first_ot_idx = None
        for i, p in enumerate(probs):
            if p > threshold:
                first_ot_idx = i
                break

        # Cut at first overthinking token
        cut_idx = first_ot_idx

        # Apply backtracking to sentence boundary if enabled
        if use_backtrack and cut_idx is not None:
            response_token_ids = full_token_ids[assistant_start:]
            backtracked_idx = None
            for i in range(cut_idx - 1, -1, -1):
                if i < len(response_token_ids):
                    token_text = tokenizer.decode([response_token_ids[i]], skip_special_tokens=False)
                    if '\n' in token_text:
                        backtracked_idx = i + 1  # Cut after the newline
                        break

            # If found a newline, check if we need to go back one more sentence
            if backtracked_idx is not None and backtracked_idx > 0:
                prev_dot_idx = None
                for i in range(backtracked_idx - 1, -1, -1):
                    if i < len(response_token_ids):
                        token_text = tokenizer.decode([response_token_ids[i]], skip_special_tokens=False)
                        if '.' in token_text:
                            prev_dot_idx = i + 1
                            break

                if prev_dot_idx is not None:
                    sentence_text = tokenizer.decode(response_token_ids[prev_dot_idx:backtracked_idx], skip_special_tokens=False)
                else:
                    sentence_text = tokenizer.decode(response_token_ids[:backtracked_idx], skip_special_tokens=False)

                sentence_stripped = sentence_text.strip()
                if sentence_stripped.lower().startswith('wait') or not sentence_stripped.endswith('.'):
                    if prev_dot_idx is not None:
                        backtracked_idx = prev_dot_idx

            # Update cut_idx with backtracked position
            if backtracked_idx is not None:
                cut_idx = backtracked_idx
        
        # Build prompt
        prompt_ids = None
        if cut_idx is not None:
            absolute_cut_position = assistant_start + cut_idx
            if absolute_cut_position < len(full_token_ids):
                cut_token_ids = full_token_ids[assistant_start:absolute_cut_position]
            else:
                cut_token_ids = full_token_ids[assistant_start:]
            
            prompt_ids = full_token_ids[:assistant_start] + cut_token_ids
            
            if len(prompt_ids) > max_prompt_len:
                log_print(f"Warning: Sample {idx} prompt too long ({len(prompt_ids)} tokens), truncating to {max_prompt_len}")
                prompt_ids = prompt_ids[:max_prompt_len]
            
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False) + summarization_prompt
            generation_prompts.append(prompt_text)
            prompt_indices.append(idx)
        
        all_metadata.append({
            'idx': idx,
            'item': item,
            'first_ot_idx': first_ot_idx,
            'cut_idx': cut_idx,
            'assistant_start': assistant_start,
            'full_token_ids': full_token_ids,
            'prompt_ids': prompt_ids,
        })
    
    return generation_prompts, prompt_indices, all_metadata


def build_response_from_generation(metadata, outputs_map, tokenizer, summarization_prompt):
    """Build final response from generation output."""
    if metadata['cut_idx'] is not None and metadata['idx'] in outputs_map:
        cut_response = tokenizer.decode(
            metadata['prompt_ids'][metadata['assistant_start']:], 
            skip_special_tokens=False
        )
        return cut_response + summarization_prompt + outputs_map[metadata['idx']].outputs[0].text
    else:
        return metadata['item'].get('response', '')



def extract_basenames(test_data_path, ckpt_path):
    """Extract simplified basenames from paths for output filenames."""
    # Handle test data path
    test_jsonl_path = test_data_path if test_data_path.endswith('.jsonl') else f"{test_data_path}.jsonl"
    test_dataset_dir = os.path.dirname(test_jsonl_path)
    test_jsonl = os.path.basename(test_jsonl_path)
    test_data_basename = os.path.splitext(test_jsonl)[0].split('.')[0]
    
    # Handle checkpoint path
    ckpt_full = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_basename = ckpt_full.replace('model_', '', 1)
    
    return test_dataset_dir, test_jsonl, test_data_basename, ckpt_basename



def calculate_group_metrics(group):
    """Calculate averaged metrics for a group of results (e.g., 3 responses per problem)."""
    our_acc = sum(1 for r in group if r['correct'] is True) / len(group)
    original_acc = sum(1 for r in group if r['original_correct'] is True) / len(group)
    avg_our_reasoning = sum(r['our_reasoning_length'] for r in group) / len(group)
    avg_original_reasoning = sum(r['original_reasoning_length'] for r in group) / len(group)
    avg_our_response = sum(r['our_response_length'] for r in group) / len(group)
    avg_original_response = sum(r['original_response_length'] for r in group) / len(group)
    reasoning_ratio = safe_ratio(avg_our_reasoning, avg_original_reasoning)
    response_ratio = safe_ratio(avg_our_response, avg_original_response)
    acc_ratio = safe_ratio(our_acc, original_acc, default=0)
    
    return {
        'our_acc': our_acc,
        'original_acc': original_acc,
        'our_reasoning_length': avg_our_reasoning,
        'original_reasoning_length': avg_original_reasoning,
        'our_response_length': avg_our_response,
        'original_response_length': avg_original_response,
        'reasoning_length_ratio': reasoning_ratio,
        'response_length_ratio': response_ratio,
        'acc_ratio': acc_ratio
    }


def calculate_summary_metrics(grouped_results):
    """Calculate overall summary metrics from grouped results."""
    if not grouped_results:
        return {
            'avg_our_acc': 0, 'avg_original_acc': 0,
            'avg_our_reasoning_length': 0, 'avg_original_reasoning_length': 0,
            'avg_our_response_length': 0, 'avg_original_response_length': 0,
            'final_reasoning_ratio': 1.0, 'final_response_ratio': 1.0, 'final_acc_ratio': 0
        }
    
    avg_our_acc = sum(r['our_acc'] for r in grouped_results) / len(grouped_results)
    avg_original_acc = sum(r['original_acc'] for r in grouped_results) / len(grouped_results)
    avg_our_reasoning = sum(r['our_reasoning_length'] for r in grouped_results) / len(grouped_results)
    avg_original_reasoning = sum(r['original_reasoning_length'] for r in grouped_results) / len(grouped_results)
    avg_our_response = sum(r['our_response_length'] for r in grouped_results) / len(grouped_results)
    avg_original_response = sum(r['original_response_length'] for r in grouped_results) / len(grouped_results)
    final_reasoning_ratio = safe_ratio(avg_our_reasoning, avg_original_reasoning)
    final_response_ratio = safe_ratio(avg_our_response, avg_original_response)
    final_acc_ratio = safe_ratio(avg_our_acc, avg_original_acc, default=0)
    
    return {
        'avg_our_acc': avg_our_acc,
        'avg_original_acc': avg_original_acc,
        'avg_our_reasoning_length': avg_our_reasoning,
        'avg_original_reasoning_length': avg_original_reasoning,
        'avg_our_response_length': avg_our_response,
        'avg_original_response_length': avg_original_response,
        'final_reasoning_ratio': final_reasoning_ratio,
        'final_response_ratio': final_response_ratio,
        'final_acc_ratio': final_acc_ratio,
    }


