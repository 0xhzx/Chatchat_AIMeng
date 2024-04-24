import os
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from utils.constants import *
from utils.util_funcs import clear_gpu_memory

def load_data():
    # Load the tokenized training, validation, and test datasets from disk
    train_tokenized = load_from_disk('data/processed/train_tokenized')
    val_tokenized = load_from_disk('data/processed/val_tokenized')
    test_tokenized = load_from_disk('data/processed/test_tokenized')
    return train_tokenized, val_tokenized, test_tokenized

# Load a pre-trained model from the specified path
def load_model(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model


def create_trainer(model, train_tokenized, val_tokenized):
    training_args = TrainingArguments(
        output_dir=FINE_TUNED_MODEL_PATH,
        save_total_limit=SAVE_TOTAL_LIMIT,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        evaluation_strategy=EVALUATION_STRATEGY,
    )
    # Create a Trainer instance with the model and training arguments, and the training and validation datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized
    )
    return trainer

def evaluate_model(trainer, test_tokenized):
    eval_results = trainer.evaluate(eval_dataset=test_tokenized)
    return eval_results

def clear_gpu(model):
    del model
    clear_gpu_memory()

def main():
    train_tokenized, val_tokenized, test_tokenized = load_data()
    
    original_model = load_model(BASE_MODEL_PATH)
    trainer = create_trainer(original_model, train_tokenized, val_tokenized)
    original_eval_results = evaluate_model(trainer, test_tokenized)
    clear_gpu(original_model)
    
    fintuned_model = load_model(os.path.join(MODEL_DIR, 'save'))
    trainer = create_trainer(fintuned_model, train_tokenized, val_tokenized)
    finetuned_eval_results = evaluate_model(trainer, test_tokenized)
    clear_gpu(fintuned_model)
    
    print("before fine tune:", original_eval_results)
    print("==========================Compared==========================")
    print("after fine tune:", finetuned_eval_results)

if __name__ == "__main__":
    main()