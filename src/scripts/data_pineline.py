import os
import pandas as pd
from utils.constants import RANDOM_SEED
from utils.util_funcs import data_cleaning, tokenized_batch_function
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, GenerationConfig

def load_data(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset['train']

def split_dataset(dataset, test_size=0.2):
    dataset_split = dataset.train_test_split(test_size=test_size, seed=RANDOM_SEED)
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']
    train_val_dataset = train_dataset.train_test_split(test_size=test_size, seed=RANDOM_SEED)
    train_dataset = train_val_dataset['train']
    val_dataset = train_val_dataset['test']
    return train_dataset, val_dataset, test_dataset

# Convert the datasets into pandas dataframes
def convert_to_dataframe(train_dataset, val_dataset, test_dataset):
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    val_df = pd.DataFrame(val_dataset)
    return train_df, val_df, test_df

# Apply the data cleaning function to the 'question' and 'answer' columns of each dataframe
def clean_data(train_df, val_df, test_df):
    train_df['question'] = train_df['question'].apply(data_cleaning)
    val_df['question'] = val_df['question'].apply(data_cleaning)
    test_df['question'] = test_df['question'].apply(data_cleaning)

    train_df['answer'] = train_df['answer'].apply(data_cleaning)
    val_df['answer'] = val_df['answer'].apply(data_cleaning)
    test_df['answer'] = test_df['answer'].apply(data_cleaning)
    return train_df, val_df, test_df
    
# Convert the dataframes back into Hugging Face datasets
def tokenize_data(train_df, val_df, test_df, hf_token):
    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)
    test_data = Dataset.from_pandas(test_df)

    # Tokenize the datasets using the provided Hugging Face tokenizer
    train_tokenized = train_data.map(lambda x: tokenized_batch_function(x, hf_token = hf_token), batched=True)
    val_tokenized = val_data.map(lambda x: tokenized_batch_function(x,hf_token = hf_token), batched=True)
    test_tokenized = test_data.map(lambda x: tokenized_batch_function(x,hf_token = hf_token), batched=True)
    
    # Select only the 'input_ids' and 'labels' columns from the tokenized datasets
    train_tokenized = train_tokenized.select_columns(["input_ids", "labels"])
    val_tokenized = val_tokenized.select_columns(["input_ids", "labels"])
    test_tokenized = test_tokenized.select_columns(["input_ids", "labels"])

    return train_tokenized, val_tokenized, test_tokenized

def save_data(train_df, val_df, test_df, train_tokenized, val_tokenized, test_tokenized):
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)

    train_tokenized.save_to_disk('data/processed/train_tokenized')
    val_tokenized.save_to_disk('data/processed/val_tokenized')
    test_tokenized.save_to_disk('data/processed/test_tokenized')

def process_data(dataset_name, hf_token):
    dataset = load_data(dataset_name)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    train_df, val_df, test_df = convert_to_dataframe(train_dataset, val_dataset, test_dataset)
    train_df, val_df, test_df = clean_data(train_df, val_df, test_df)
    train_tokenized, val_tokenized, test_tokenized = tokenize_data(train_df, val_df, test_df, hf_token)
    save_data(train_df, val_df, test_df, train_tokenized, val_tokenized, test_tokenized)