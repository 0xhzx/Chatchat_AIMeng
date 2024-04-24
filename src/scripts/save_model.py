import os
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
# from huggingface_hub import notebook_login
from utils.constants import *
from utils.util_funcs import *

def change_directory():
    os.chdir("..")

def load_model(model_path, tokenizer_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def push_to_hub(model, tokenizer, model_name, token):
    model.push_to_hub(model_name, token=token)
    tokenizer.push_to_hub(model_name, token=token)

def main():
    change_directory()
    # notebook_login()
    SAVE_MODEL_PATH = os.path.join(MODEL_DIR, 'save')
    finetuned_model, finetune_tokenizer = load_model(SAVE_MODEL_PATH, "google/flan-t5-base")
    push_to_hub(finetuned_model, finetune_tokenizer, "0xhzx/nv-qa", os.getenv('HF_TOKEN'))

if __name__ == "__main__":
    main()