import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList
)
from gsm8k_utils.utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from datasets import load_dataset
from collections import Counter
import json

from statistics import mean
  

def save_cache(jobs, cache_path):
    with open(cache_path, "w") as g:
        for job in jobs:
            g.write(json.dumps(job, ensure_ascii=False) + "\n")
            g.flush()

def compute_scores(jobs, cache_path):

    
    for job in jobs:
        assert len(job["gen"]) == 1
        gen = job['gen'][0]
        model_answers = []
        ground_truth_answer = extract_ground_truth(job['answer'])
        model_answer = extract_predicted_answer(gen)
        model_answers.append({'text': gen, 'numeric': model_answer})

        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        
        job.update({'correct': correct})
    save_cache(jobs, cache_path)
    
    cnt = 0
    for job in jobs:
        if job['correct']:
            cnt += 1
    total = len(jobs)

    # print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")

    return cnt / total
