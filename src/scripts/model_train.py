import os
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from utils.util_funcs import clear_gpu_memory
from utils.constants import *

def load_data():
    train_tokenized = load_from_disk('data/processed/train_tokenized')
    val_tokenized = load_from_disk('data/processed/val_tokenized')
    test_tokenized = load_from_disk('data/processed/test_tokenized')
    return train_tokenized, val_tokenized, test_tokenized

def load_model():
    original_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    original_model.save_pretrained(BASE_MODEL_PATH)
    del original_model
    clear_gpu_memory()
    loaded_original_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    clear_gpu_memory()
    return loaded_original_model, tokenizer

def fine_tune(loaded_original_model, tokenizer, train_tokenized, val_tokenized):
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=FINE_TUNED_MODEL_PATH,
        save_total_limit=SAVE_TOTAL_LIMIT,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        evaluation_strategy=EVALUATION_STRATEGY,
    )
    trainer = Trainer(
        model=loaded_original_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized
    )
    trainer.train()
    SAVE_MODEL_PATH = os.path.join(MODEL_DIR, 'save')
    trainer.model.save_pretrained(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_MODEL_PATH)

def clear_gpu():
    # Delete the original model and clear GPU memory
    del loaded_original_model
    clear_gpu_memory()

def main():
    train_tokenized, val_tokenized, test_tokenized = load_data()
    loaded_original_model, tokenizer = load_model()
    fine_tune(loaded_original_model, tokenizer, train_tokenized, val_tokenized)
    clear_gpu()

if __name__ == "__main__":
    main()