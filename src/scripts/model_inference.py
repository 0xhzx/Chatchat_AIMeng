import os
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, T5Tokenizer
from utils.constants import *
from utils.util_funcs import *
from sentence_transformers import SentenceTransformer
import numpy as np

def load_model(model_path, tokenizer_path, device):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    clear_gpu_memory()
    return model, tokenizer

def generate_responses(model, tokenizer, questions):
    responses = [generate_response(model=model, tokenizer=tokenizer, prompt=generate_prompt(question)) for question in questions]
    return responses

def compute_similarity(correct_answers, generated_answers):
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedded_correct_ans = sentence_model.encode(correct_answers)
    embedded_generated_ans = sentence_model.encode(generated_answers)
    cosine_similarity_scores = [cosine_similarity(embedded_correct_ans[i], embedded_generated_ans[i]) for i in range(len(correct_answers))]
    return cosine_similarity_scores

def main():
    SAVE_MODEL_PATH = os.path.join(MODEL_DIR, 'save')
    finetuned_model, tokenizer = load_model(SAVE_MODEL_PATH, "google/flan-t5-base", DEVICE)
    # Load the original (pre-fine-tuning) model
    original_model, _ = load_model(BASE_MODEL_PATH, MODEL_ID, DEVICE)
    # Load the test data
    test_df = pd.read_csv('data/processed/test.csv')
    res_df = test_df[0:100]
    
    # Generate responses to the questions in the test data using the original model and the fine-tuned model
    res_df['generated_answer_original'] = generate_responses(original_model, tokenizer, res_df['question'])
    res_df['generated_answer_finetuned'] = generate_responses(finetuned_model, tokenizer, res_df['question'])

    #  Save the results to a CSV file
    res_df.to_csv('data/output/compared_results.csv', index=False)

    res_df['cosine_similarity_score'] = compute_similarity(res_df['answer'].tolist(), res_df['generated_answer_finetuned'].tolist())

    res_df.to_csv('data/output/compared_results_with_score.csv', index=False)

if __name__ == "__main__":
    main()