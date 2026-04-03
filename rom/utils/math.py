#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified utilities for both MATH500 and MMLU-Pro datasets.
Auto-detects dataset type and applies appropriate answer extraction/validation logic.
"""

import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class SplitSolution:
    solution: str
    answer: Optional[str]
    correct: bool


# ============================================================================
# Dataset Detection
# ============================================================================

def detect_dataset_type(data_sample: Dict[str, Any]) -> str:
    """
    Auto-detect dataset type from data sample.
    Returns: 'mmlu-pro' or 'math500'
    """
    problem = data_sample.get('problem', '')
    
    # MMLU-Pro typically has options like "A. ", "B. ", etc. in the problem
    if re.search(r'\b[A-J]\.\s+', problem):
        return 'mmlu-pro'
    
    return 'math500'


# ============================================================================
# MATH500 Functions
# ============================================================================

def extract_answer_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    if "\\boxed{" not in text:
        return None
    start_idx = text.rfind("\\boxed{")
    answer_text = text[start_idx + len("\\boxed{"):]
    bracket = 1
    idx = 0
    while idx < len(answer_text) and bracket > 0:
        if answer_text[idx] == "{":
            bracket += 1
        if answer_text[idx] == "}":
            bracket -= 1
        idx += 1
    return answer_text[:idx-1] if bracket == 0 else None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    answer = str(answer).strip()

    # Strip \boxed{...} wrapper if present
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[len("\\boxed{"):-1]

    # Remove variable assignment prefixes like "d = ", "x = ", "y = ", etc.
    answer = re.sub(r'^[a-zA-Z]\s*=\s*', '', answer)
    
    # Normalize \text{...} to plain text
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    
    answer = answer.replace("\\dfrac{", "\\frac{")
    # Normalize \frac{a}{b} to a/b
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    # Normalize LaTeX \left and \right parentheses/brackets/braces
    answer = re.sub(r'\\left\(', '(', answer)
    answer = re.sub(r'\\right\)', ')', answer)
    answer = re.sub(r'\\left\[', '[', answer)
    answer = re.sub(r'\\right\]', ']', answer)
    answer = re.sub(r'\\left\{', '{', answer)
    answer = re.sub(r'\\right\}', '}', answer)
    # Normalize sqrt(...) to \sqrt... (convert plain sqrt to LaTeX format)
    answer = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt\1', answer)
    # Normalize \sqrt{...} to \sqrt... (remove braces around sqrt argument)
    answer = re.sub(r'\\sqrt\{([^}]+)\}', r'\\sqrt\1', answer)
    # Normalize LaTeX math symbols to Unicode (common ones)
    latex_to_unicode = {
        r'\\pi': 'π',
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\theta': 'θ',
        r'\\lambda': 'λ',
        r'\\mu': 'μ',
        r'\\sigma': 'σ',
        r'\\Delta': 'Δ',
        r'\\Omega': 'Ω',
        r'\\phi': 'φ',
        r'\\omega': 'ω',
    }
    for latex, unicode_char in latex_to_unicode.items():
        answer = re.sub(latex, unicode_char, answer)
    answer = re.sub(r'\s+', '', answer)
    return answer.lower()


def check_answer_correctness_math500(response: str, expected_answer: str) -> bool:
    """Check if the response contains the correct answer for MATH500."""
    if not expected_answer:
        return False
    
    # Extract boxed answer from response
    extracted = extract_answer_boxed(response)
    if not extracted:
        # Fallback: look for "Final Answer" or similar patterns
        patterns = [
            r"Final Answer[:\s]+(.+?)(?:\n|$)",
            r"answer is[:\s]+(.+?)(?:\n|$)",
            r"answer[:\s]+(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                break
    
    if not extracted:
        return False
    
    # Normalize and compare
    extracted_norm = normalize_answer(extracted)
    expected_norm = normalize_answer(expected_answer)
    
    # Exact match or containment check
    return extracted_norm == expected_norm or expected_norm in extracted_norm


# ============================================================================
# MMLU-Pro Functions
# ============================================================================

def extract_boxed_answer_mmlu(response: str) -> Optional[str]:
    """Extract the last boxed answer from response, handling nested braces and \text{}."""
    boxed_pattern = r'\\boxed\{'
    matches = []
    
    for match in re.finditer(boxed_pattern, response):
        start = match.end()
        brace_count = 1
        end = start
        
        # Find matching closing brace
        while end < len(response) and brace_count > 0:
            if response[end] == '{':
                brace_count += 1
            elif response[end] == '}':
                brace_count -= 1
            end += 1
        
        if brace_count == 0:
            content = response[start:end-1].strip()
            # Remove \text{} wrapper if present
            text_start = content.find(r'\text{')
            if text_start != -1:
                text_start += len(r'\text{')
                text_brace_count = 1
                text_end = text_start
                while text_end < len(content) and text_brace_count > 0:
                    if content[text_end] == '{':
                        text_brace_count += 1
                    elif content[text_end] == '}':
                        text_brace_count -= 1
                    text_end += 1
                if text_brace_count == 0:
                    content = content[text_start:text_end-1].strip()
            matches.append(content)
    
    return matches[-1] if matches else None


def extract_text_answer(response: str) -> Optional[str]:
    """Extract answer from text format like 'Answer: E. interference' or '**Answer**: **E. interference**' or '### ✅ **Correct Answer: H.' or '**A.'"""
    patterns = [
        # Match "Final Answer:" or "Answer:" with content on the same line or next line
        r'(?:^|\n)\s*(?:###\s*)?[✅✓]?\s*\*{0,2}(?:Correct|Final)\s+Answer\*{0,2}\s*:?\s*\*{0,2}\s*\n?\s*([A-J](?:\.\s*[^\n]*)?)',
        r'(?:^|\n)\s*\*{0,2}Answer\*{0,2}\s*:?\s*\*{0,2}\s*\n?\s*([A-J](?:\.\s*[^\n]*)?)',
        r'(?:^|\n)\s*The\s+answer\s+is\s*:?\s*([^\n]+)',
        r'(?:^|\n)\s*Final\s+answer\s*:?\s*([^\n]+)',
        # Match standalone letter answers like "**A." or "A." 
        r'\*{0,2}([A-J])\.(?:\s*$|\s*\n)',
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            answer_text = match.group(1).strip()
            # Remove markdown formatting (**, ###, emojis, etc.)
            answer_text = re.sub(r'\*+', '', answer_text).strip()
            answer_text = re.sub(r'^###\s*', '', answer_text).strip()
            answer_text = re.sub(r'^[✅✓]\s*', '', answer_text).strip()
            if answer_text:
                all_matches.append((match.start(), answer_text))
    
    if all_matches:
        # Return the last match (by position in text)
        all_matches.sort(key=lambda x: x[0])
        return all_matches[-1][1]
    
    return None


def extract_option_content(text: str, option_letter: str) -> Optional[str]:
    """Extract the content of an option from text (problem or response). E.g., "H. 100" -> "100" """
    pattern = re.compile(
        r'^\s*' + re.escape(option_letter) + r'\s*[\.:]\s*(.+)$',
        flags=re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def check_answer_correctness_mmlu(problem: str, response: str, expected_answer: str) -> bool:
    """Check overall correctness for MMLU-Pro based on the last boxed answer or text answer."""
    # First try to extract from \boxed{}
    answer = extract_boxed_answer_mmlu(response)
    
    # If no boxed answer, try to extract from text format
    if not answer:
        answer = extract_text_answer(response)
    
    if not answer:
        return False
    
    # Check if answer contains multiple options (e.g., "A and F" or "A. X and F. Y")
    # This should be considered incorrect for single-choice questions
    answer_upper = answer.upper()
    # Look for patterns like "A and B", "A AND B", "A, B", etc.
    multiple_options_patterns = [
        r'\b([A-J])\s+(?:and|AND|&|,)\s+([A-J])\b',  # "A and B", "A & B", "A, B"
        r'\b([A-J])[\.:]\s+[^,]+\s+(?:and|AND|&)\s+([A-J])[\.:]\s+',  # "A. X and B. Y"
    ]
    for pattern in multiple_options_patterns:
        if re.search(pattern, answer):
            return False
    
    # Clean up answer: remove trailing punctuation for single letter answers
    answer_clean = answer.strip()
    # If answer is just a letter with optional punctuation, extract just the letter
    single_letter_match = re.match(r'^\s*([A-J])\s*[\.:!?]?\s*$', answer_clean, re.IGNORECASE)
    if single_letter_match:
        answer_clean = single_letter_match.group(1)
    
    # Case 1: Direct letter match (no normalization)
    if answer_clean.upper() == expected_answer.upper():
        return True
    
    # Case 2: Answer contains "Letter. Content" format (e.g., "E. interference" or "J. Cranial")
    # Must start with the letter and colon/period, but ensure no other options follow
    letter_content_match = re.match(
        r'^\s*' + re.escape(expected_answer) + r'\s*[\.:]\s*(.+)$',
        answer,
        re.IGNORECASE
    )
    if letter_content_match:
        # Additional check: ensure the content doesn't contain another option letter
        content_after = letter_content_match.group(1)
        # If content contains another letter option pattern, it's likely multiple choices
        if not re.search(r'\b[A-J]\s*[\.:]\s*\w+', content_after, re.IGNORECASE):
            return True
    
    # Case 3: Content match (with normalization) - extract from problem
    option_content = extract_option_content(problem, expected_answer)
    if option_content:
        return normalize_answer(answer) == normalize_answer(option_content)
    
    return False


def sentence_has_correct_answer(sentence: str, expected_answer: str, option_content: Optional[str]) -> bool:
    """Check if a sentence contains the correct answer with required conditions."""
    sent_lower = sentence.lower()
    
    # Check uncertainty words
    if any(word in sent_lower for word in ['wait', 'seems', 'suggest', 'suggests']):
        return False
    
    # Check if contains 'answer' or 'option'
    if 'answer' not in sent_lower and 'option' not in sent_lower:
        return False
    
    # Check for option letter (case-insensitive, whole word, no normalization)
    has_letter = bool(re.search(r'\b' + re.escape(expected_answer) + r'\b', sentence, re.IGNORECASE))
    
    # Check for option content (with normalization)
    has_content = False
    if option_content:
        has_content = normalize_answer(option_content) in normalize_answer(sentence)
    
    return has_letter or has_content


def split_into_solutions_mmlu(problem: str, response: str, expected_answer: str) -> List[SplitSolution]:
    """Split response at the first sentence containing correct answer for MMLU-Pro."""
    sentence_pattern = re.compile(r'[^.]+\.', re.DOTALL)
    option_content = extract_option_content(problem, expected_answer)
    
    # Find first correct answer sentence
    first_correct_end = None
    for match in sentence_pattern.finditer(response):
        sentence = match.group(0)
        if sentence_has_correct_answer(sentence, expected_answer, option_content):
            first_correct_end = match.end()
            break
    
    # Overall correctness based on boxed answer
    overall_correct = check_answer_correctness_mmlu(problem, response, expected_answer)
    
    # No split found, return whole response
    if first_correct_end is None:
        return [SplitSolution(solution=response.strip(), answer=None, correct=overall_correct)]
    
    # Split into two parts
    part1 = response[:first_correct_end].strip()
    part2 = response[first_correct_end:].strip()
    
    # Check if part2 is empty or only contains </think>
    part2_stripped = part2.strip()
    is_part2_empty = not part2_stripped or part2_stripped == '</think>'
    
    # If part2 is empty or only </think>, use part1's correctness for overall
    if is_part2_empty:
        first_split_correct = overall_correct
        splits = [SplitSolution(solution=part1, answer=None, correct=first_split_correct)]
    else:
        # If overall response is incorrect, all splits should be incorrect
        # If overall response is correct, first split with correct answer is correct
        first_split_correct = overall_correct
        splits = [SplitSolution(solution=part1, answer=None, correct=first_split_correct)]
        splits.append(SplitSolution(solution=part2, answer=None, correct=overall_correct))
    
    return splits


# ============================================================================
# Unified Interface
# ============================================================================

def check_answer_correctness(problem: str, response: str, expected_answer: str, dataset_type: str = None) -> bool:
    """
    Unified answer correctness check. Auto-detects dataset type if not provided.
    """
    if dataset_type is None:
        dataset_type = detect_dataset_type({'problem': problem})
    
    if dataset_type == 'mmlu-pro':
        return check_answer_correctness_mmlu(problem, response, expected_answer)
    else:  # math500
        return check_answer_correctness_math500(response, expected_answer)


def split_into_solutions(problem: str, response: str, expected_answer: str, dataset_type: str = None) -> List[Dict[str, Any]]:
    """
    Unified split solutions function. Auto-detects dataset type if not provided.
    Returns list of dicts with 'solution', 'answer', 'correct' keys.
    """
    if dataset_type is None:
        dataset_type = detect_dataset_type({'problem': problem})
    
    if dataset_type == 'mmlu-pro':
        splits = split_into_solutions_mmlu(problem, response, expected_answer)
    else:  # math500
        # For MATH500, we don't split - return full response
        overall_correct = check_answer_correctness_math500(response, expected_answer)
        splits = [SplitSolution(solution=response.strip(), answer=None, correct=overall_correct)]
    
    # Convert to dict format
    return [
        {
            'solution': s.solution,
            'answer': s.answer,
            'correct': s.correct,
        }
        for s in splits
    ]


def extract_answer(response: str, dataset_type: str = None, problem: str = None) -> Optional[str]:
    """
    Unified answer extraction. Auto-detects dataset type if not provided.
    """
    if dataset_type is None and problem is not None:
        # Auto-detect dataset type from problem
        dataset_type = detect_dataset_type({'problem': problem})
    
    if dataset_type == 'mmlu-pro':
        answer = extract_boxed_answer_mmlu(response)
        if not answer:
            answer = extract_text_answer(response)
        return answer
    else:  # math500 or unknown
        # For MATH500, only use boxed answer extraction
        return extract_answer_boxed(response)
