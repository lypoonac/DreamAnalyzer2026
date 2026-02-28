# Updated Python Code for Dream Analysis App Prototype (with Index Type Fix)
# Handles potential string/list type issues in local Parquet loads.
# Uses TWO deep learning models: SentenceTransformer for recommendation, BERT for interpretation.

import os
import glob
import json
import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd

# Step 1: Load the dataset and embeddings (with fallback, auto-detect, and type fixes)
def load_dream_dataset(local_dir='./dream_dataset'):
    # Create local dir if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"Created local directory: {local_dir}. Please download files into it and rerun.")

    try:
        # Try online load
        ds = load_dataset("samvlad/dream-decoder-dataset")
        dreams = ds['train']

        embeddings_path = hf_hub_download(
            "samvlad/dream-decoder-dataset",
            "data/embeddings.npy",
            repo_type="dataset"
        )
        embeddings = np.load(embeddings_path)

        print("Loaded dataset online successfully.")
    except Exception as e:
        print(f"Online load failed: {e}. (Try clearing cache with 'huggingface-cli delete-cache'.) Falling back to local files...")

        # Local fallback: Find any .parquet file in the directory
        parquet_files = glob.glob(os.path.join(local_dir, '*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found in {local_dir}. "
                "Steps to fix:\n"
                "1. Go to https://huggingface.co/datasets/samvlad/dream-decoder-dataset\n"
                "2. Download the train Parquet file (e.g., train-00000-of-00001.parquet) from 'Files and versions'.\n"
                "3. Download data/embeddings.npy.\n"
                "4. Place both in {local_dir} and rerun the script."
            )

        parquet_path = parquet_files[0]  # Use the first one found
        print(f"Using local Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        dreams = Dataset.from_pandas(df)

        local_embeddings_path = os.path.join(local_dir, 'embeddings.npy')
        if not os.path.exists(local_embeddings_path):
            raise FileNotFoundError(
                f"Local embeddings not found at {local_embeddings_path}. "
                "Download it from the dataset page (under 'Files and versions' > data/embeddings.npy) and place in {local_dir}."
            )

        embeddings = np.load(local_embeddings_path)

        print("Loaded dataset from local files.")

    # Validate and fix types (e.g., ensure lists are actual lists, not strings)
    list_fields = ['symbols', 'emotions', 'actions', 'tags']
    for field in list_fields:
        if isinstance(dreams[0][field], str):  # If loaded as string, parse
            print(f"Warning: Field '{field}' loaded as string; parsing to list.")
            parsed_column = []
            for item in dreams[field]:
                try:
                    parsed_column.append(json.loads(item) if isinstance(item, str) else item)
                except json.JSONDecodeError:
                    parsed_column.append([])  # Fallback to empty list
            dreams = dreams.remove_columns([field]).add_column(field, parsed_column)

    # Validate loaded data
    if len(dreams) != 1200:
        print(f"Warning: Loaded {len(dreams)} examples, but expected 1200. Check dataset integrity.")
    print(f"Sample data types: emotions={type(dreams[0]['emotions'])}, symbols={type(dreams[0]['symbols'])}")  # Debug print

    return dreams, embeddings

# Step 2: Prepare BERT for fine-tuning (with safer column access)
class DreamDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def fine_tune_bert(dreams):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10, problem_type="multi_label_classification")

    emotion_list = ['anxious', 'excited', 'angry', 'curious', 'afraid', 'confused', 'hopeful', 'embarrassed', 'lonely', 'relieved']

    # Safer access: Use column slicing instead of iteration
    num_samples = min(200, len(dreams))
    texts = dreams['dream_text'][:num_samples]
    labels = []
    for i in range(num_samples):
        emotions = dreams['emotions'][i]
        # Ensure emotions is a list (fallback if not)
        if not isinstance(emotions, list):
            print(f"Warning: emotions at index {i} is {type(emotions)}; treating as empty list.")
            emotions = []
        label_vec = [1 if emo in emotions else 0 for emo in emotion_list]
        labels.append(label_vec)

    train_dataset = DreamDataset(texts, labels, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        save_steps=100,
        eval_strategy="no",  # Use 'eval_strategy' for compatibility
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    print("Fine-tuning BERT...")
    trainer.train()
    return model, tokenizer, emotion_list

# Step 3: Generate Interpretation using BERT (with type checks)
def generate_interpretation(user_dream, bert_model, tokenizer, emotion_list, matched_dream):
    inputs = tokenizer(user_dream, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    preds = torch.sigmoid(outputs.logits).detach().numpy()[0]
    predicted_emotions = [emotion_list[i] for i in range(len(preds)) if preds[i] > 0.38]

    # Safely access with type checks
    symbols = matched_dream['symbols'] if isinstance(matched_dream.get('symbols'), list) else []
    symbol = symbols[0] if symbols else "unknown"
    setting = matched_dream.get('setting', "unknown")
    actions_list = matched_dream['actions'] if isinstance(matched_dream.get('actions'), list) else []
    actions = ", ".join(actions_list)

    interpretation = (
        f"Predicted emotions: {', '.join(predicted_emotions) or 'none detected'}. "
        f"The symbol '{symbol}' may represent a challenge. Setting: {setting}. "
        f"Consider your actions like {actions} in real life."
    )
    return interpretation

# Step 4: Find similar dreams
def find_similar_dreams(user_embedding, embeddings, top_k=3):
    similarities = cosine_similarity([user_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# Step 5: Generate Recommendations (with type checks)
def generate_recommendations(matched_dream):
    emotions = matched_dream['emotions'] if isinstance(matched_dream.get('emotions'), list) else []
    symbols = matched_dream['symbols'] if isinstance(matched_dream.get('symbols'), list) else []

    recs = []
    if any(e in emotions for e in ["anxious", "afraid"]):
        recs.append("Try meditation for calm.")
    if any(s in symbols for s in ["flying", "storm"]):
        recs.append("Journal about freedom or change.")
    return "\n".join(recs) or "Reflect on your dream."

# Main Function
def analyze_dream(user_dream, dreams, embeddings, embedder, bert_model, bert_tokenizer, emotion_list):
    user_embedding = embedder.encode(user_dream)
    top_indices, similarities = find_similar_dreams(user_embedding, embeddings, top_k=1)

    # Debug print for index type
    print(f"Top index type: {type(top_indices[0])}")

    # FIXED: Cast numpy.int64 to Python int for Dataset indexing
    matched_index = int(top_indices[0])
    matched_dream = dreams[matched_index]  # Now safe

    # Debug print to verify type
    print(f"Matched dream type: {type(matched_dream)}")

    print(f"\nTop Match Similarity: {similarities[0]:.2f}")
    print(f"Matched Dream: {matched_dream.get('dream_text', 'unknown')}")

    interpretation = generate_interpretation(user_dream, bert_model, bert_tokenizer, emotion_list, matched_dream)
    print("\nInterpretation (via BERT):")
    print(interpretation)

    recommendations = generate_recommendations(matched_dream)
    print("\nRecommendations (via SentenceTransformer similarity):")
    print(recommendations)

# Entry Point
if __name__ == "__main__":
    print("Loading dataset and models...")
    dreams, embeddings = load_dream_dataset()

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    bert_model, bert_tokenizer, emotion_list = fine_tune_bert(dreams)

    user_dream = input("\nEnter your dream description: ") or "I was flying over a stormy beach, feeling anxious and excited."

    analyze_dream(user_dream, dreams, embeddings, embedder, bert_model, bert_tokenizer, emotion_list)

    print("\nAnalysis complete!")
