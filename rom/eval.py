import os
import torch
import json
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from rom.utils.eval_helpers import (
    compute_probs, calculate_lengths_and_correctness,
    safe_ratio, create_log_print, build_response_from_generation,
    extract_basenames, calculate_group_metrics,
    calculate_summary_metrics, prepare_prompts_for_checkpoint
)
from rom.env import set_seed
# Disable v1 engine which has initialization issues
os.environ["VLLM_USE_V1"] = "0"
# Note: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are NOT set here
# to allow vLLM to access HuggingFace config even with local model cache
from vllm import LLM, SamplingParams


def _load_and_initialize(model_name, cached_probs_file, seed, bf16, debug, log_print, gpu_memory_utilization=0.9):
    """Load data and initialize vLLM model."""
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load test data from cache
    log_print(f"Loading cached probabilities from {cached_probs_file}...")
    with open(cached_probs_file, 'r') as f:
        all_test_data = json.load(f)

    if debug:
        all_test_data = all_test_data[:2]
        log_print(f"DEBUG MODE: Processing only first 2 samples")

    num_samples = len(all_test_data)
    log_print(f"Loaded {num_samples} samples from cache")

    # Calculate max problem length
    log_print("Calculating maximum problem length...")
    max_problem_tokens = 0
    for item in all_test_data:
        problem = item.get('problem', '')
        problem_tokens = len(tokenizer.encode(problem, add_special_tokens=False))
        max_problem_tokens = max(max_problem_tokens, problem_tokens)
    
    max_model_len = max_problem_tokens + 8192 + 512
    log_print(f"Max problem tokens: {max_problem_tokens}, setting max_model_len: {max_model_len}")
    
    # Initialize vLLM
    log_print(f"Loading model {model_name} with vLLM...")
    llm = LLM(
        model=model_name,
        dtype="bfloat16" if bf16 else "float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len, # prompt+output
        trust_remote_code=True,
        enforce_eager=True,
        seed=seed
    )
    log_print("vLLM model loaded successfully!")
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=8192, # output
        seed=seed
    )
    
    return tokenizer, all_test_data, llm, sampling_params, max_model_len


def evaluate_head(
    ckpt_path,
    test_dataset_dir,
    test_jsonl_filename=None,
    model_name="Qwen/Qwen3-8B",
    idx_layer=32,
    max_length=8192,
    bf16=True,
    device=None,
    eval_mode=True,
    seed=42,
    log_file=None,
    cached_probs_file=None,
    summarization_prompt="\nBased on my analysis above, I will now provide the final analysis first, and then the final answer in the box. </think>",
    debug=False,
    no_backtrack=False,
    threshold=0.5,
    gpu_memory_utilization=0.9,
    samples_per_problem=3,
    results_jsonl_path=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print = create_log_print(log_file)
    
    # Load and initialize
    tokenizer, test_data, llm, sampling_params, max_model_len = _load_and_initialize(
        model_name, cached_probs_file, seed, bf16, debug, log_print, gpu_memory_utilization
    )
    
    num_samples = len(test_data)
    
    # Prepare prompts
    log_print("Preparing prompts for generation...")
    generation_prompts, prompt_indices, prompt_metadata = prepare_prompts_for_checkpoint(
        test_data, tokenizer, device,
        max_model_len, summarization_prompt, log_print, use_backtrack=not no_backtrack,
        threshold=threshold
    )
    
    # Batch generation with vLLM
    log_print(f"Generating responses for {num_samples} samples with vLLM...")
    outputs = llm.generate(generation_prompts, sampling_params)
    
    # Create outputs map
    outputs_map = {idx: outputs[i] for i, idx in enumerate(prompt_indices)}
    
    # Process results
    results = []
    log_print("Processing generated responses...")
    results_jf = open(results_jsonl_path, 'w', encoding='utf-8') if results_jsonl_path else None

    with torch.no_grad():
        for metadata in tqdm(prompt_metadata, desc="Processing results"):
            idx = metadata['idx']
            item = metadata['item']
            assistant_start = metadata['assistant_start']
            full_token_ids = metadata['full_token_ids']
            
            problem = item['problem']
            response = item.get('response', '')
            expected_answer = item.get('expected_answer')
            probs = np.array(item['probs'])
            
            # Get our method response
            our_method_response = build_response_from_generation(
                metadata, outputs_map, tokenizer, summarization_prompt
            )
            
            # Calculate lengths and check correctness
            cut_diff = metadata.get('cut_diff', 0)
            cut_idx = metadata.get('cut_idx')
            our_results = calculate_lengths_and_correctness(
                our_method_response, full_token_ids, assistant_start, tokenizer, problem, expected_answer, cut_diff, cut_idx
            )
            orig_results = calculate_lengths_and_correctness(
                response, full_token_ids, assistant_start, tokenizer, problem, expected_answer, 0, None
            )
            
            # Calculate ratios
            reasoning_length_ratio = safe_ratio(our_results['reasoning_length'], our_results['original_reasoning_length'])
            response_length_ratio = safe_ratio(our_results['response_length'], our_results['original_response_length'])
            
            # Log detailed results to file only
            if log_file:
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"Sample {idx + 1}/{num_samples}\n")
                log_file.write(f"Problem: {problem}\n")
                log_file.write(f"First OT token: {metadata['first_ot_idx']}\n")
                log_file.write(f"Cut at token: {metadata['cut_idx']}\n")
                log_file.write(f"Our reasoning length: {our_results['reasoning_length']} tokens\n")
                log_file.write(f"Original reasoning length: {our_results['original_reasoning_length']} tokens\n")
                log_file.write(f"Reasoning length ratio: {reasoning_length_ratio:.3f}\n")
                log_file.write(f"Our response length: {our_results['response_length']} tokens\n")
                log_file.write(f"Original response length: {our_results['original_response_length']} tokens\n")
                log_file.write(f"Response length ratio: {response_length_ratio:.3f}\n")
                log_file.write(f"Expected answer: {expected_answer}\n")
                log_file.write(f"Extracted answer: {our_results['extracted_answer']}\n")
                log_file.write(f"Original answer: {orig_results['extracted_answer']}\n")
                log_file.write(f"Our method correct: {our_results['is_correct']}\n")
                log_file.write(f"Original correct: {orig_results['is_correct']}\n")
                log_file.write(f"\nOur method response:\n{our_method_response}\n")
                log_file.write(f"\nOriginal response:\n{response}\n")
                log_file.write(f"{'='*80}\n")
                log_file.flush()
            
            results.append({
                'sample_idx': idx + 1,
                'problem': problem,
                'original_response': response,
                'our_method_response': our_method_response,
                'first_ot_idx': metadata['first_ot_idx'],
                'cut_idx': metadata['cut_idx'],
                'our_reasoning_length': our_results['reasoning_length'],
                'original_reasoning_length': our_results['original_reasoning_length'],
                'reasoning_length_ratio': reasoning_length_ratio,
                'our_response_length': our_results['response_length'],
                'original_response_length': our_results['original_response_length'],
                'response_length_ratio': response_length_ratio,
                'expected_answer': expected_answer,
                'extracted_answer': our_results['extracted_answer'],
                'original_extracted_answer': orig_results['extracted_answer'],
                'correct': our_results['is_correct'],
                'original_correct': orig_results['is_correct'],
                'probs': probs,
                'split_solutions': item.get('split_solutions', []),
                'assistant_start': assistant_start,
            })

            # Stream write per-sample results to JSONL
            if results_jf:
                record = {
                    'sample_idx': idx + 1,
                    'problem': problem,
                    'expected_answer': expected_answer,
                    'correct': our_results['is_correct'],
                    'original_correct': orig_results['is_correct'],
                    'our_reasoning_length': our_results['reasoning_length'],
                    'original_reasoning_length': our_results['original_reasoning_length'],
                    'reasoning_length_ratio': reasoning_length_ratio,
                    'our_response_length': our_results['response_length'],
                    'original_response_length': our_results['original_response_length'],
                    'response_length_ratio': response_length_ratio,
                    'cut_idx': metadata['cut_idx'],
                    'first_ot_idx': metadata['first_ot_idx'],
                    'extracted_answer': our_results['extracted_answer'],
                }
                json.dump(record, results_jf, ensure_ascii=False)
                results_jf.write('\n')
                results_jf.flush()

    if results_jf:
        results_jf.close()

    # Group results by problem (N responses per problem)
    grouped_results = []
    total_problems = len(results) // samples_per_problem
    for i in range(0, len(results), samples_per_problem):
        group = results[i:i+samples_per_problem]
        # Calculate averages per problem
        problem_result = calculate_group_metrics(group)
        problem_result['problem'] = group[0]['problem']
        grouped_results.append(problem_result)
        
        # Log per-problem summary to file only
        if log_file:
            problem_idx = len(grouped_results)
            log_file.write(f'\n{"="*80}\n')
            log_file.write(f"Problem {problem_idx}/{total_problems} Summary:\n")
            log_file.write(f"  Our method: acc={problem_result['our_acc']*100:.2f}%, reasoning={problem_result['our_reasoning_length']:.1f}, response={problem_result['our_response_length']:.1f} tokens\n")
            log_file.write(f"  Original: acc={problem_result['original_acc']*100:.2f}%, reasoning={problem_result['original_reasoning_length']:.1f}, response={problem_result['original_response_length']:.1f} tokens\n")
            log_file.write(f"  Ratios: reasoning={problem_result['reasoning_length_ratio']:.3f}, response={problem_result['response_length_ratio']:.3f}, acc={problem_result['acc_ratio']:.3f}\n")
            log_file.write("="*80 + "\n")
            log_file.flush()
    
    # Summary statistics - average across all problems
    summary = calculate_summary_metrics(grouped_results)
    avg_our_acc = summary['avg_our_acc']
    avg_original_acc = summary['avg_original_acc']
    avg_our_reasoning = summary['avg_our_reasoning_length']
    avg_original_reasoning = summary['avg_original_reasoning_length']
    avg_our_response = summary['avg_our_response_length']
    avg_original_response = summary['avg_original_response_length']
    final_reasoning_ratio = summary['final_reasoning_ratio']
    final_response_ratio = summary['final_response_ratio']
    final_acc_ratio = summary['final_acc_ratio']
    
    log_print(f'\n{"="*80}')
    log_print("EVALUATION SUMMARY")
    log_print(f'{"="*80}')
    log_print(f'Total problems: {len(grouped_results)} (sampling time: {samples_per_problem})')
    log_print(f'\nOur Method:')
    log_print(f'  Average accuracy: {avg_our_acc*100:.2f}%')
    log_print(f'  Average reasoning length: {avg_our_reasoning:.1f} tokens')
    log_print(f'  Average response length: {avg_our_response:.1f} tokens')
    log_print(f'\nOriginal:')
    log_print(f'  Average accuracy: {avg_original_acc*100:.2f}%')
    log_print(f'  Average reasoning length: {avg_original_reasoning:.1f} tokens')
    log_print(f'  Average response length: {avg_original_response:.1f} tokens')
    log_print(f'\nRatios:')
    log_print(f'  Reasoning length ratio: {final_reasoning_ratio:.3f}')
    log_print(f'  Response length ratio: {final_response_ratio:.3f}')
    log_print(f'  Accuracy ratio: {final_acc_ratio:.3f}')
    log_print(f'{"="*80}')
    
    # Return both sample-level and grouped results
    return {
        'sample_results': results,
        'grouped_results': grouped_results,
        'summary': summary
    }


def _get_or_compute_probs(ckpt_path, ckpt_basename, test_dataset_dir, test_jsonl, 
                           test_data_basename, args, cache_dir, step_name):
    """Get cached probabilities or compute them if needed."""
    print("\n" + "="*80)
    print(f"STEP {step_name}")
    print("="*80)
    
    # Build cache filename with debug and suffix info (ws/wc not included since probs are pre-computed)
    suffix_str = args.suffix if args.suffix else ""
    debug_str = "_debug" if args.debug else ""
    cache_file = os.path.join(
        cache_dir, 
        f"test_probs_{test_data_basename}_{ckpt_basename}{debug_str}{suffix_str}.json"
    )
    
    if os.path.exists(cache_file) and not args.force_recompute:
        print(f"Found existing cache: {cache_file}")
        return cache_file
    
    # Pass the expected cache_file path to compute_probs
    return compute_probs(
        ckpt_path=ckpt_path,
        test_dataset_dir=test_dataset_dir,
        test_jsonl_filename=test_jsonl,
        model_name=args.model_name,
        idx_layer=args.idx_layer,
        max_length=args.max_length,
        seed=args.seed,
        cache_dir=cache_dir,
        cache_file=cache_file,  # Pass the expected cache file path
        debug=args.debug
    )



def load_config(config_path):
    """Load YAML config file. Returns empty dict if file not found."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    # Pre-parse to get --config path
    pre_parser = argparse.ArgumentParser(add_help=False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(os.path.dirname(script_dir), "configs", "eval.yaml")
    pre_parser.add_argument("--config", type=str, default=default_config)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Evaluate StreamingHead. Edit configs/eval.yaml or pass --key value to override.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to YAML config file")

    # --- Model & Checkpoint ---
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint file")
    parser.add_argument("--model_name", type=str, help="HuggingFace model ID or local path")
    parser.add_argument("--idx_layer", type=int, help="Transformer layer index")
    parser.add_argument("--max_length", type=int, help="Max sequence length")

    # --- Test Data ---
    parser.add_argument("--test_data", type=str, help="Path to test dataset JSONL file")

    # --- Evaluation ---
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--force_recompute", action="store_true", default=None, help="Force recomputation even if cache exists")
    parser.add_argument("--debug", action="store_true", default=None, help="Debug mode: process only first 2 samples")
    parser.add_argument("--no_backtrack", action="store_true", default=None, help="Cut directly at first OT token without backtracking")
    parser.add_argument("--threshold", type=float, help="Probability threshold for overthinking detection (default: 0.5)")
    parser.add_argument("--gpu_memory_utilization", type=float, help="Fraction of GPU memory for vLLM KV cache")

    # --- Output ---
    parser.add_argument("--suffix", type=str, help="Suffix for output directory and filenames")
    parser.add_argument("--summarization_prompt", type=str, help="Prompt appended after cutting for summarization")
    parser.add_argument("--samples_per_problem", type=int, help="Number of samples per problem for grouping (default: 3)")
    parser.add_argument("--save_results_jsonl", action="store_true", default=False, help="Save per-sample results to JSONL file")

    # Apply YAML defaults, then parse CLI (CLI overrides YAML)
    parser.set_defaults(**config)
    args = parser.parse_args()

    # Default samples_per_problem to 3 if not set
    if args.samples_per_problem is None:
        args.samples_per_problem = 3

    # Extract paths and basenames
    test_dataset_dir, test_jsonl, test_data_basename, ckpt_basename = extract_basenames(
        args.test_data, args.ckpt_path
    )

    suffix_str = args.suffix if args.suffix else ""
    debug_str = "_debug" if args.debug else ""
    output_dir = f"eval_results_seed{args.seed}{debug_str}{suffix_str}"
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(
        output_dir,
        f"evaluation_results_{test_data_basename}_{ckpt_basename}_seed{args.seed}{debug_str}{suffix_str}.txt"
    )

    # Cache directory
    cache_dir = f"eval_cache{debug_str}{suffix_str}"
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = _get_or_compute_probs(
        args.ckpt_path, ckpt_basename, test_dataset_dir, test_jsonl,
        test_data_basename, args, cache_dir, "Computing probabilities"
    )

    # Prepare streaming JSONL path
    results_jsonl_path = os.path.join(
        output_dir,
        f"results_{test_data_basename}_{ckpt_basename}_seed{args.seed}{debug_str}{suffix_str}.jsonl"
    )

    # Run evaluation with logging
    with open(log_path, 'w', encoding='utf-8') as log_file:
        eval_results = evaluate_head(
            ckpt_path=args.ckpt_path,
            test_dataset_dir=test_dataset_dir,
            test_jsonl_filename=test_jsonl,
            model_name=args.model_name,
            idx_layer=args.idx_layer,
            max_length=args.max_length,
            bf16=True,
            eval_mode=True,
            seed=args.seed,
            log_file=log_file,
            cached_probs_file=cache_file,
            summarization_prompt=args.summarization_prompt,
            debug=args.debug,
            no_backtrack=args.no_backtrack,
            threshold=args.threshold,
            gpu_memory_utilization=args.gpu_memory_utilization,
            samples_per_problem=args.samples_per_problem,
            results_jsonl_path=results_jsonl_path,
        )

    print(f"\nEvaluation log saved to {log_path}")
    print(f"Per-sample results streamed to {results_jsonl_path}")



if __name__ == '__main__':
    main()