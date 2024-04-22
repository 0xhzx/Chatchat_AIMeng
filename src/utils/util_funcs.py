import gc
import os, sys

import torch
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import re
import string
from transformers import AutoTokenizer, GenerationConfig
from .constants import MODEL_ID, MODEL_DIR
from dotenv import load_dotenv
load_dotenv()


## Clean the dataset text
def data_cleaning(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# tokenization for LLM with batch processing
def tokenized_batch_function(input_data, hf_token=None):
    if hf_token is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token = os.getenv("HF_TOKEN"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token = hf_token)
    pre_prompt = "You are a Question-Answering assistant. According to the following question:\n\n"
    # concatenate the question with the prompt
    prompt = [pre_prompt + question + "\n\nHere are some good answers:\n" for question in input_data["question"]]
    input_data['input_ids'] = tokenizer(prompt, truncation=True, padding="max_length", return_tensors="pt").input_ids
    input_data['labels'] = tokenizer(input_data['answer'], truncation=True, padding="max_length", return_tensors="pt").input_ids
    
    return input_data

# def tokenizer(question, answer):
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     pre_prompt = "You are a Question-Answering assistant. According to the following question:\n\n"
#     # concatenate the question with the prompt
#     prompt = pre_prompt + question + "\n\nHere are some good answers:\n"
#     # return value 'token_type_ids' and 'input_ids'
#     input_ids = tokenizer(prompt, truncation=True, padding="max_length", return_tensors="pt").input_ids
#     labels = tokenizer(answer, truncation=True, padding="max_length", return_tensors="pt").input_ids
    
#     return input_ids, labels


def clear_gpu_memory(in_detail=False):
    """
    Functionality:
    Clears GPU memory cache and garbage collection to free up memory references.
    
    Args:
    in_detail (bool): If True, print memory stats before and after cleanup.

    Ref:
    https://stackoverflow.com/questions/15197286/how-can-i-flush-gpu-memory-using-cuda-physical-reset-is-unavailable
    """
    if in_detail:
        print("Before cleanup:")
        print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Reserved:  {torch.cuda.memory_reserved()} bytes")

    # Collect garbage to potentially free up memory references
    gc.collect()
    # Clear PyTorch's CUDA memory cache
    torch.cuda.empty_cache()
    
    if in_detail:
        print("After cleanup:")
        print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Reserved:  {torch.cuda.memory_reserved()} bytes")