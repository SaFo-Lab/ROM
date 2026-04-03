import os
import torch
import json
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from rom.utils.eval_helpers import (
    compute_probs, find_reasoning_length, calculate_lengths_and_correctness,
    safe_ratio, create_log_print, build_response_from_generation,
    create_response_metadata, format_checkpoint_log, extract_basenames,
    build_output_paths, calculate_checkpoint_stats, calculate_group_metrics,
    calculate_summary_metrics, write_comparison_summary,
    prepare_prompts_for_checkpoint
)
from rom.env import set_seed
# Disable v1 engine which has initialization issues
os.environ["VLLM_USE_V1"] = "0"
# Note: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are NOT set here
# to allow vLLM to access HuggingFace config even with local model cache
from vllm import LLM, SamplingParams


def _load_and_initialize(model_name, cached_probs_files, seed, bf16, debug, log_print, gpu_memory_utilization=0.9):
    """Load data and initialize vLLM model."""
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load test data from cache(s)
    if isinstance(cached_probs_files, list):
        # Multiple files for comparison mode
        all_test_data = []
        for cache_file in cached_probs_files:
            log_print(f"Loading cached probabilities from {cache_file}...")
            with open(cache_file, 'r') as f:
                test_data = json.load(f)
            all_test_data.append(test_data)
        
        if debug:
            all_test_data = [data[:2] for data in all_test_data]
            log_print(f"DEBUG MODE: Processing only first 2 samples")
        
        num_samples = len(all_test_data[0])
        log_print(f"Loaded {num_samples} samples from {len(cached_probs_files)} cache(s)")
    else:
        # Single file for single evaluation mode
        log_print(f"Loading cached probabilities from {cached_probs_files}...")
        with open(cached_probs_files, 'r') as f:
            all_test_data = json.load(f)
        
        if debug:
            all_test_data = all_test_data[:2]
            log_print(f"DEBUG MODE: Processing only first 2 samples")
        
        num_samples = len(all_test_data)
        log_print(f"Loaded {num_samples} samples from cache")
    
    # Calculate max problem length
    log_print("Calculating maximum problem length...")
    max_problem_tokens = 0
    # Handle both single mode (list of dicts) and comparison mode (list of lists)
    data_to_scan = all_test_data[0] if isinstance(cached_probs_files, list) else all_test_data
    for item in data_to_scan:
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
    window_size=1,
    window_count=1,
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
        window_size=window_size, window_count=window_count, threshold=threshold
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


def evaluate_head_comparison(
    base_ckpt_path,
    compare_ckpt_path,
    base_ckpt_basename,
    compare_ckpt_basename,
    test_dataset_dir,
    test_jsonl_filename,
    model_name,
    idx_layer,
    max_length,
    bf16=True,
    device=None,
    eval_mode=True,
    seed=42,
    log_file=None,
    base_cached_probs_file=None,
    compare_cached_probs_file=None,
    summarization_prompt="\nBased on my analysis above, I will now provide the final analysis first, and then the final answer in the box. </think>",
    debug=False,
    no_backtrack=False,
    window_size=1,
    window_count=1,
    threshold=0.5,
    gpu_memory_utilization=0.9,
    results_jsonl_path=None,
):
    """Run evaluation for both checkpoints and generate comparison output."""
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print = create_log_print(log_file)
    
    # Load and initialize
    tokenizer, (base_test_data, compare_test_data), llm, sampling_params, max_model_len = _load_and_initialize(
        model_name, [base_cached_probs_file, compare_cached_probs_file], seed, bf16, debug, log_print, gpu_memory_utilization
    )
    
    num_samples = len(base_test_data)
    
    # Prepare prompts for BOTH checkpoints
    log_print("Preparing prompts for both checkpoints...")
    base_generation_prompts, base_prompt_indices, base_metadata = prepare_prompts_for_checkpoint(
        base_test_data, tokenizer, device,
        max_model_len, summarization_prompt, log_print, use_backtrack=False,
        window_size=window_size, window_count=window_count, threshold=threshold
    )

    compare_generation_prompts, compare_prompt_indices, compare_metadata = prepare_prompts_for_checkpoint(
        compare_test_data, tokenizer, device,
        max_model_len, summarization_prompt, log_print, use_backtrack=(not no_backtrack),
        window_size=window_size, window_count=window_count, threshold=threshold
    )
    
    # Merge metadata from both checkpoints
    prompt_metadata = []
    for i in range(num_samples):
        prompt_metadata.append({
            'idx': i,
            'problem': base_test_data[i]['problem'],
            'response': base_test_data[i].get('response', ''),
            'expected_answer': base_test_data[i].get('expected_answer'),
            'base_item': base_test_data[i],
            'compare_item': compare_test_data[i],
            'base_first_ot_idx': base_metadata[i]['first_ot_idx'],
            'base_cut_idx': base_metadata[i]['cut_idx'],
            'base_prompt_ids': base_metadata[i]['prompt_ids'],
            'compare_first_ot_idx': compare_metadata[i]['first_ot_idx'],
            'compare_cut_idx': compare_metadata[i]['cut_idx'],
            'compare_prompt_ids': compare_metadata[i]['prompt_ids'],
            'assistant_start': base_metadata[i]['assistant_start'],
            'full_token_ids': base_metadata[i]['full_token_ids'],
        })
    
    # Batch generation for BASE checkpoint
    base_outputs_map = {}
    if base_generation_prompts:
        log_print(f"\nGenerating responses for BASE checkpoint ({base_ckpt_basename})...")
        log_print(f"  Generating for {len(base_generation_prompts)}/{num_samples} samples with cuts")
        base_outputs = llm.generate(base_generation_prompts, sampling_params)
        base_outputs_map = {idx: base_outputs[i] for i, idx in enumerate(base_prompt_indices)}
    else:
        log_print(f"\nNo cuts detected for BASE checkpoint ({base_ckpt_basename}), using original responses")
    
    # Batch generation for COMPARE checkpoint
    compare_outputs_map = {}
    if compare_generation_prompts:
        log_print(f"Generating responses for COMPARE checkpoint ({compare_ckpt_basename})...")
        log_print(f"  Generating for {len(compare_generation_prompts)}/{num_samples} samples with cuts")
        compare_outputs = llm.generate(compare_generation_prompts, sampling_params)
        compare_outputs_map = {idx: compare_outputs[i] for i, idx in enumerate(compare_prompt_indices)}
    else:
        log_print(f"No cuts detected for COMPARE checkpoint ({compare_ckpt_basename}), using original responses")
    
    # Process results
    results = []
    log_print("\nProcessing and comparing results...")
    results_jf = open(results_jsonl_path, 'w', encoding='utf-8') if results_jsonl_path else None

    with torch.no_grad():
        for metadata in tqdm(prompt_metadata, desc="Processing results"):
            idx = metadata['idx']
            problem = metadata['problem']
            response = metadata['response']
            expected_answer = metadata['expected_answer']
            assistant_start = metadata['assistant_start']
            full_token_ids = metadata['full_token_ids']
            
            # Build responses for both checkpoints
            base_meta = create_response_metadata(idx, metadata['base_cut_idx'], metadata['base_prompt_ids'], assistant_start, response)
            compare_meta = create_response_metadata(idx, metadata['compare_cut_idx'], metadata['compare_prompt_ids'], assistant_start, response)
            
            base_response = build_response_from_generation(base_meta, base_outputs_map, tokenizer, summarization_prompt)
            compare_response = build_response_from_generation(compare_meta, compare_outputs_map, tokenizer, summarization_prompt)
            
            # Calculate all metrics
            base_cut_diff = base_meta.get('cut_diff', 0)
            base_cut_idx = base_meta.get('cut_idx')
            compare_cut_diff = compare_meta.get('cut_diff', 0)
            compare_cut_idx = compare_meta.get('cut_idx')
            base_results = calculate_lengths_and_correctness(
                base_response, full_token_ids, assistant_start, tokenizer, problem, expected_answer, base_cut_diff, base_cut_idx
            )
            compare_results = calculate_lengths_and_correctness(
                compare_response, full_token_ids, assistant_start, tokenizer, problem, expected_answer, compare_cut_diff, compare_cut_idx
            )
            orig_results = calculate_lengths_and_correctness(
                response, full_token_ids, assistant_start, tokenizer, problem, expected_answer, 0, None
            )
            
            # Calculate ratios
            base_reasoning_ratio = safe_ratio(base_results['reasoning_length'], base_results['original_reasoning_length'])
            base_response_ratio = safe_ratio(base_results['response_length'], base_results['original_response_length'])
            compare_reasoning_ratio = safe_ratio(compare_results['reasoning_length'], compare_results['original_reasoning_length'])
            compare_response_ratio = safe_ratio(compare_results['response_length'], compare_results['original_response_length'])
            
            # Calculate diffs
            reasoning_length_diff = compare_results['reasoning_length'] - base_results['reasoning_length']
            response_length_diff = compare_results['response_length'] - base_results['response_length']
            cut_idx_diff = (metadata['compare_cut_idx'] or 0) - (metadata['base_cut_idx'] or 0)
            
            # Build detailed log string
            detailed_log = []
            detailed_log.append(f"\n{'='*80}\n")
            detailed_log.append(f"Sample {idx + 1}/{num_samples}\n")
            detailed_log.append(f"Problem: {problem}\n")
            
            detailed_log.append(format_checkpoint_log(
                f"BASE ({base_ckpt_basename})", metadata['base_first_ot_idx'], metadata['base_cut_idx'],
                base_results, orig_results, expected_answer, base_reasoning_ratio, base_response_ratio
            ))
            
            detailed_log.append(format_checkpoint_log(
                f"COMPARE ({compare_ckpt_basename})", metadata['compare_first_ot_idx'], metadata['compare_cut_idx'],
                compare_results, orig_results, expected_answer, compare_reasoning_ratio, compare_response_ratio
            ))
            
            detailed_log.append(f"\n--- DIFF (COMPARE - BASE) ---\n")
            detailed_log.append(f"Reasoning length diff: {reasoning_length_diff:+d} tokens\n")
            detailed_log.append(f"Response length diff: {response_length_diff:+d} tokens\n")
            detailed_log.append(f"Cut position diff: {cut_idx_diff:+d} tokens\n")
            detailed_log.append(f"Reasoning ratio diff: {compare_reasoning_ratio - base_reasoning_ratio:+.3f}\n")
            detailed_log.append(f"Response ratio diff: {compare_response_ratio - base_response_ratio:+.3f}\n")
            
            # Add regression/improvement indicator
            if base_results['is_correct'] and not compare_results['is_correct']:
                detailed_log.append(f"\n⚠️ REGRESSION: Base correct but Compare wrong!\n")
            elif not base_results['is_correct'] and compare_results['is_correct']:
                detailed_log.append(f"\n✓ IMPROVEMENT: Base wrong but Compare correct!\n")
            
            detailed_log.append(f"\n--- BASE Response ---\n{base_response}\n")
            detailed_log.append(f"\n--- COMPARE Response ---\n{compare_response}\n")
            detailed_log.append(f"\n--- ORIGINAL Response ---\n{response}\n")
            detailed_log.append(f"{'='*80}\n")
            
            results.append({
                'sample_idx': idx + 1,
                'problem': problem,
                'original_response': response,
                'base_response': base_response,
                'compare_response': compare_response,
                'base_first_ot_idx': metadata['base_first_ot_idx'],
                'base_cut_idx': metadata['base_cut_idx'],
                'base_reasoning_length': base_results['reasoning_length'],
                'base_response_length': base_results['response_length'],
                'base_extracted_answer': base_results['extracted_answer'],
                'base_correct': base_results['is_correct'],
                'compare_first_ot_idx': metadata['compare_first_ot_idx'],
                'compare_cut_idx': metadata['compare_cut_idx'],
                'compare_reasoning_length': compare_results['reasoning_length'],
                'compare_response_length': compare_results['response_length'],
                'compare_extracted_answer': compare_results['extracted_answer'],
                'compare_correct': compare_results['is_correct'],
                'original_reasoning_length': orig_results['original_reasoning_length'],
                'original_response_length': orig_results['original_response_length'],
                'expected_answer': expected_answer,
                'original_extracted_answer': orig_results['extracted_answer'],
                'original_correct': orig_results['is_correct'],
                'base_probs': np.array(metadata['base_item']['probs']),
                'compare_probs': np.array(metadata['compare_item']['probs']),
                'assistant_start': assistant_start,
                'detailed_log': ''.join(detailed_log),
            })

            # Stream write per-sample results to JSONL
            if results_jf:
                record = {
                    'sample_idx': idx + 1,
                    'problem': problem,
                    'expected_answer': expected_answer,
                    'base_correct': base_results['is_correct'],
                    'compare_correct': compare_results['is_correct'],
                    'original_correct': orig_results['is_correct'],
                    'base_reasoning_length': base_results['reasoning_length'],
                    'compare_reasoning_length': compare_results['reasoning_length'],
                    'original_reasoning_length': orig_results['original_reasoning_length'],
                    'base_response_length': base_results['response_length'],
                    'compare_response_length': compare_results['response_length'],
                    'original_response_length': orig_results['original_response_length'],
                    'base_cut_idx': metadata['base_cut_idx'],
                    'compare_cut_idx': metadata['compare_cut_idx'],
                    'base_extracted_answer': base_results['extracted_answer'],
                    'compare_extracted_answer': compare_results['extracted_answer'],
                }
                json.dump(record, results_jf, ensure_ascii=False)
                results_jf.write('\n')
                results_jf.flush()

    if results_jf:
        results_jf.close()

    # Summary statistics
    num_results = len(results)
    base_correct_count = sum(1 for r in results if r['base_correct'] is True)
    compare_correct_count = sum(1 for r in results if r['compare_correct'] is True)
    original_correct_count = sum(1 for r in results if r['original_correct'] is True)
    
    regression_count = sum(1 for r in results if r['base_correct'] and not r['compare_correct'])
    improvement_count = sum(1 for r in results if not r['base_correct'] and r['compare_correct'])
    
    log_print(f'\n{"="*80}')
    log_print("COMPARISON SUMMARY")
    log_print(f'{"="*80}')
    log_print(f'Total samples: {num_results}')
    log_print(f'\nBASE ({base_ckpt_basename}):')
    log_print(f'  Accuracy: {base_correct_count}/{num_results} = {base_correct_count/num_results*100:.2f}%')
    log_print(f'  Avg reasoning length: {np.mean([r["base_reasoning_length"] for r in results]):.1f} tokens')
    log_print(f'  Avg response length: {np.mean([r["base_response_length"] for r in results]):.1f} tokens')
    log_print(f'\nCOMPARE ({compare_ckpt_basename}):')
    log_print(f'  Accuracy: {compare_correct_count}/{num_results} = {compare_correct_count/num_results*100:.2f}%')
    log_print(f'  Avg reasoning length: {np.mean([r["compare_reasoning_length"] for r in results]):.1f} tokens')
    log_print(f'  Avg response length: {np.mean([r["compare_response_length"] for r in results]):.1f} tokens')
    log_print(f'\nORIGINAL:')
    log_print(f'  Accuracy: {original_correct_count}/{num_results} = {original_correct_count/num_results*100:.2f}%')
    log_print(f'  Avg reasoning length: {np.mean([r["original_reasoning_length"] for r in results]):.1f} tokens')
    log_print(f'  Avg response length: {np.mean([r["original_response_length"] for r in results]):.1f} tokens')
    log_print(f'\nDIFFERENCES:')
    log_print(f'  Regressions (Base✓ → Compare✗): {regression_count}')
    log_print(f'  Improvements (Base✗ → Compare✓): {improvement_count}')
    log_print(f'{"="*80}')
    
    return {
        'sample_results': results,
        'base_name': base_ckpt_basename,
        'compare_name': compare_ckpt_basename,
        'summary': {
            'base_acc': base_correct_count / num_results,
            'compare_acc': compare_correct_count / num_results,
            'original_acc': original_correct_count / num_results,
            'regression_count': regression_count,
            'improvement_count': improvement_count,
        }
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


def run_comparison_mode(args):
    """Run evaluation in comparison mode: compare base checkpoint with another checkpoint."""
    
    # Extract paths and basenames
    test_dataset_dir, test_jsonl, test_data_basename, base_ckpt_basename = extract_basenames(
        args.test_data, args.ckpt_path
    )
    _, _, _, compare_ckpt_basename = extract_basenames(args.test_data, args.compare_ckpt)
    
    # Output dir: suffix only (e.g. _sliding). Cache / html / txt: suffix + ws/wc.
    suffix_str = args.suffix if args.suffix else ""
    suffix_str_ws_wc = suffix_str + f"_ws{args.window_size}_wc{args.window_count}_t{args.threshold}"
    debug_str = "_debug" if args.debug else ""
    output_dir = f"Kelp-my/eval_results_seed{args.seed}{debug_str}{suffix_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_path, output_html = build_output_paths(
        output_dir, test_data_basename, base_ckpt_basename, args.seed, suffix_str_ws_wc, compare_ckpt_basename
    )
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Cache directory (ws/wc not included since probs are pre-computed)
    cache_dir = f"Kelp-my/eval_cache{debug_str}{suffix_str}"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Compute/load probabilities for both checkpoints
    base_cache_file = _get_or_compute_probs(
        args.ckpt_path, base_ckpt_basename, test_dataset_dir, test_jsonl, 
        test_data_basename, args, cache_dir, "1: Processing BASE checkpoint"
    )
    
    compare_cache_file = _get_or_compute_probs(
        args.compare_ckpt, compare_ckpt_basename, test_dataset_dir, test_jsonl,
        test_data_basename, args, cache_dir, "2: Processing COMPARE checkpoint"
    )
    
    # Run evaluation
    print("\n" + "="*80)
    print("STEP 3: Running evaluations and generating comparison")
    print("="*80)

    # Prepare streaming JSONL path
    results_jsonl_path = os.path.join(
        output_dir,
        f"results_{test_data_basename}_{compare_ckpt_basename}_seed{args.seed}{debug_str}{suffix_str_ws_wc}.jsonl"
    )

    eval_results = evaluate_head_comparison(
        base_ckpt_path=args.ckpt_path,
        compare_ckpt_path=args.compare_ckpt,
        base_ckpt_basename=base_ckpt_basename,
        compare_ckpt_basename=compare_ckpt_basename,
        test_dataset_dir=test_dataset_dir,
        test_jsonl_filename=test_jsonl,
        model_name=args.model_name,
        idx_layer=args.idx_layer,
        max_length=args.max_length,
        bf16=True,
        eval_mode=True,
        seed=args.seed,
        log_file=None,
        base_cached_probs_file=base_cache_file,
        compare_cached_probs_file=compare_cache_file,
        summarization_prompt=args.summarization_prompt,
        debug=args.debug,
        no_backtrack=args.no_backtrack,
        window_size=args.window_size,
        window_count=args.window_count,
        threshold=args.threshold,
        gpu_memory_utilization=args.gpu_memory_utilization,
        results_jsonl_path=results_jsonl_path,
    )
    
    # Write results to log file
    print(f"\nWriting comparison results to {log_path}...")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        results = eval_results['sample_results']
        
        # Write summary
        write_comparison_summary(log_file, results, base_ckpt_basename, compare_ckpt_basename)
        
        # Write detailed per-sample results
        log_file.write("="*80 + "\n")
        log_file.write("DETAILED PER-SAMPLE RESULTS\n")
        log_file.write("="*80 + "\n")
        
        for res in results:
            if 'detailed_log' in res:
                log_file.write(res['detailed_log'])
    
    print(f"Comparison evaluation log saved to {log_path}")
    print(f"Per-sample results streamed to {results_jsonl_path}")




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
    parser.add_argument("--ckpt_path", type=str, help="Path to base checkpoint file")
    parser.add_argument("--compare_ckpt", type=str, help="Comparison checkpoint (enables comparison mode)")
    parser.add_argument("--model_name", type=str, help="HuggingFace model ID or local path")
    parser.add_argument("--idx_layer", type=int, help="Transformer layer index")
    parser.add_argument("--max_length", type=int, help="Max sequence length")

    # --- Test Data ---
    parser.add_argument("--test_data", type=str, help="Path to test dataset JSONL file")

    # --- Evaluation ---
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--force_recompute", action="store_true", default=None, help="Force recomputation even if cache exists")
    parser.add_argument("--debug", action="store_true", default=None, help="Debug mode: process only first 2 samples")
    parser.add_argument("--no_backtrack", action="store_true", default=None, help="Cut directly at first_ot_idx without backtracking")
    parser.add_argument("--window_size", type=int, help="Sliding window size for overthinking detection")
    parser.add_argument("--window_count", type=int, help="Overthinking tokens required in window to trigger cut")
    parser.add_argument("--threshold", type=float, help="Probability threshold for overthinking detection (default: 0.5)")
    parser.add_argument("--gpu_memory_utilization", type=float, help="Fraction of GPU memory for vLLM KV cache")

    # --- Output ---
    parser.add_argument("--output_html", type=str, help="Output HTML filename")
    parser.add_argument("--skip_html", action="store_true", default=False, help="Skip HTML visualization generation")
    parser.add_argument("--suffix", type=str, help="Suffix for output directory and filenames")
    parser.add_argument("--summarization_prompt", type=str, help="Prompt appended after cutting for summarization")
    parser.add_argument("--samples_per_problem", type=int, help="Number of samples per problem for grouping (default: 3)")
    parser.add_argument("--save_results_jsonl", action="store_true", default=False, help="Save per-sample results to JSONL file")
    parser.add_argument("--no_auto_compare", action="store_true", default=False, help="Disable auto-detection of comparison mode for _aug checkpoints")

    # Apply YAML defaults, then parse CLI (CLI overrides YAML)
    parser.set_defaults(**config)
    args = parser.parse_args()

    # Default samples_per_problem to 3 if not set
    if args.samples_per_problem is None:
        args.samples_per_problem = 3

    # Auto-detect comparison mode: if current checkpoint has '_aug' and no compare_ckpt specified,
    # try to find the base checkpoint automatically
    if not args.no_auto_compare and not args.compare_ckpt and ('_aug' in args.ckpt_path or 'aug_' in args.ckpt_path):
        # Try to find base checkpoint by removing _aug
        base_ckpt_path = args.ckpt_path.replace('_aug', '').replace('aug_', '')
        if os.path.exists(base_ckpt_path) and base_ckpt_path != args.ckpt_path:
            print(f"\n{'='*80}")
            print("AUTO-DETECTED COMPARISON MODE")
            print(f"{'='*80}")
            print(f"Found 'aug' checkpoint: {args.ckpt_path}")
            print(f"Auto-found base checkpoint: {base_ckpt_path}")
            print(f"Running in comparison mode...")
            print(f"{'='*80}\n")
            
            # Swap: base should be in ckpt_path, aug should be in compare_ckpt
            args.compare_ckpt = args.ckpt_path
            args.ckpt_path = base_ckpt_path
    
    # Check if comparison mode
    if args.compare_ckpt:
        run_comparison_mode(args)
        return

    # Single evaluation mode - extract paths and basenames
    test_dataset_dir, test_jsonl, test_data_basename, ckpt_basename = extract_basenames(
        args.test_data, args.ckpt_path
    )
    
    # Output dir: suffix only (e.g. _sliding). Cache / html / txt: suffix + ws/wc.
    suffix_str = args.suffix if args.suffix else ""
    suffix_str_ws_wc = suffix_str + f"_ws{args.window_size}_wc{args.window_count}_t{args.threshold}"
    debug_str = "_debug" if args.debug else ""
    output_dir = f"Kelp-my/eval_results_seed{args.seed}{debug_str}{suffix_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output HTML filename (with ws/wc)
    if args.output_html == "debug.html":
        output_html_name = f"visualization_{test_data_basename}_{ckpt_basename}_seed{args.seed}{debug_str}{suffix_str_ws_wc}.html"
    else:
        base, ext = os.path.splitext(args.output_html)
        ext = ext or ".html"
        output_html_name = f"{base}_{test_data_basename}_{ckpt_basename}_seed{args.seed}{debug_str}{suffix_str_ws_wc}{ext}"
    
    output_html = os.path.join(output_dir, output_html_name)
    log_path = os.path.join(
        output_dir, 
        f"evaluation_results_{test_data_basename}_{ckpt_basename}_seed{args.seed}{debug_str}{suffix_str_ws_wc}.txt"
    )
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Cache directory (ws/wc not included since probs are pre-computed)
    cache_dir = f"Kelp-my/eval_cache{debug_str}{suffix_str}"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = _get_or_compute_probs(
        args.ckpt_path, ckpt_basename, test_dataset_dir, test_jsonl,
        test_data_basename, args, cache_dir, "Computing probabilities"
    )
    
    # Prepare streaming JSONL path
    results_jsonl_path = os.path.join(
        output_dir,
        f"results_{test_data_basename}_{ckpt_basename}_seed{args.seed}{debug_str}{suffix_str_ws_wc}.jsonl"
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
            window_size=args.window_size,
            window_count=args.window_count,
            threshold=args.threshold,
            gpu_memory_utilization=args.gpu_memory_utilization,
            samples_per_problem=args.samples_per_problem,
            results_jsonl_path=results_jsonl_path,
        )

    print(f"\nEvaluation log saved to {log_path}")
    print(f"Per-sample results streamed to {results_jsonl_path}")



if __name__ == '__main__':
    main()