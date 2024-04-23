import gc
import os, sys

import torch
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse
import chainlit as cl
from langchain import PromptTemplate
from typing import List
from langchain.docstore.document import Document
from transformers import AutoTokenizer, GenerationConfig

try:
    from modules.constants import *
except:
    from constants import *


def get_prompt(config):
    if config["llm_params"]["use_history"]:
        if config["llm_params"]["llm_loader"] != None:
            custom_prompt_template = general_template_with_history
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "chat_history", "question"],
        )
    else:
        custom_prompt_template = general_template
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"],
        )
    return prompt

def get_sources(res, answer):
    source_documents = res["source_documents"]  # type: List[Document]
    text_elements = []  # type: List[cl.Text]
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    return answer, text_elements

def clear_gpu_memory():
    print("Before cleanup:")
    print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"Reserved:  {torch.cuda.memory_reserved()} bytes")
    # Collect garbage to potentially free up memory references
    gc.collect()
    # Clear PyTorch's CUDA memory cache
    with torch.no_grad():
        torch.cuda.empty_cache()


# generate the suitable prompt for the model
def generate_prompt(question):
    pre_prompt = "You are a Question-Answering assistant. According to the following question:\n\n"
    prompt = pre_prompt + question + "\n\nHere are some good answers:\n"
    return prompt

# generate the response
def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    '''
    The contrastive search decoding strategy was proposed in the 2022 paper A Contrastive Framework for Neural Text Generation.
    Ref: https://huggingface.co/docs/transformers/v4.40.0/en/generation_strategies
    '''
    model.eval()
    tokenized_prompt = tokenizer(generate_prompt(prompt), return_tensors="pt")["input_ids"].to(DEVICE)
    model_output = model.generate(tokenized_prompt, penalty_alpha=0.6, top_k=4, generation_config=GenerationConfig(max_new_tokens=max_new_tokens, eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id))[0]
    response = tokenizer.decode(model_output, skip_special_tokens=True)
    clear_gpu_memory()
    return response