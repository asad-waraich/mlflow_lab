import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import TrainingArguments, Trainer
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = "/Users/asadullahwaraich/Library/CloudStorage/OneDrive-HigherEducationCommission/Desktop/NEU Fall 2025/MLOps/Project/Github/lab-lens"

# Add data-pipeline to Python path
sys.path.append(os.path.join(BASE_DIR, 'data-pipeline'))

class MedicalTextDataset(Dataset):
    """Custom dataset for medical text from your pipeline"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_pipeline_data():
    """Load processed data from your data pipeline"""
    data_dir = os.path.join(BASE_DIR, 'data-pipeline', 'data', 'processed')
    
    # Load 47-feature dataset
    features_path = os.path.join(data_dir, 'mimic_features_advanced.csv')
    
    # Load the original processed text
    summaries_path = os.path.join(data_dir, 'processed_discharge_summaries.csv')
    
    # Check if files exist
    if not os.path.exists(features_path):
        print(f"Error: Cannot find {features_path}")
        return None
    
    if not os.path.exists(summaries_path):
        print(f"Error: Cannot find {summaries_path}")
        return None
    
    print(f"Loading features from: {features_path}")
    df_features = pd.read_csv(features_path)
    
    print(f"Loading text from: {summaries_path}")
    df_text = pd.read_csv(summaries_path)
    
    # Merge on hadm_id to combine features with text
    if 'hadm_id' in df_features.columns and 'hadm_id' in df_text.columns:
        df_complete = pd.merge(
            df_features, 
            df_text[['hadm_id', 'cleaned_text']], 
            on='hadm_id', 
            how='left'
        )
        print(f"Merged data: {df_complete.shape[0]} rows, {df_complete.shape[1]} columns")
    else:
        print("Warning: No hadm_id column found for merging")
        df_complete = df_features
    
    return df_complete

def prepare_summarization_data(df):
    """Prepare data for summarization task"""
    # Use features to create labels
    # For summarization, we'll create importance scores based on features
    
    # Create pseudo-labels for training
    # High importance if: urgent, high abnormal labs, complex
    df['importance_score'] = (
        df['urgency_indicator'] * 0.4 +
        df['abnormal_lab_ratio'] * 0.3 +
        df['complexity_score'] * 0.3
    )
    
    # Convert to binary classification for sentence importance
    df['is_important'] = (df['importance_score'] > df['importance_score'].median()).astype(int)
    
    # Get text and labels
    texts = df['cleaned_text'].fillna('')
    labels = df['is_important'].values
    
    return texts, labels, df

def main():
    print("Starting Model training script...")
    import mlflow
    if mlflow.active_run():
        mlflow.end_run()
    print("Loading data from pipeline...")
    df = load_pipeline_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare data
    texts, labels, df_full = prepare_summarization_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    X_train = X_train[:100]  #  100 training samples
    X_test = X_test[:50]     #  50 test samples
    y_train = y_train[:100]
    y_test = y_test[:50]

    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize BioBERT tokenizer and model
    print("Loading Model...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # For extractive summarization, can use DistilBERT oro BioBERT for sequence classification
    # This will classify whether a sentence is important or not
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary: important/not important
    )
    
    # Create datasets
    train_dataset = MedicalTextDataset(X_train, y_train, tokenizer)
    test_dataset = MedicalTextDataset(X_test, y_test, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Start MLflow tracking
    mlflow.set_experiment("biobert-medical-summarization")
    
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("max_length", 512)
        mlflow.log_param("train_size", len(X_train))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            eval_strategy="no",
            save_strategy="no",
            logging_steps=10,
            # load_best_model_at_end=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        # Log metrics
        mlflow.log_metrics(eval_results)
        
        # Save model
        model_save_path = "../models/biobert_summarizer"
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="biobert_model",
            registered_model_name="BioBERT-Medical-Summarizer"
        )
        
        print(f"Model saved to {model_save_path}")
    
    # Test the model with sample predictions
    print("\n=== Testing Model ===")
    model.eval()
    sample_text = X_test.iloc[0] if hasattr(X_test, 'iloc') else X_test[0]
    
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)
        
    print(f"Sample text (first 200 chars): {sample_text[:200]}...")
    print(f"Predicted importance: {'Important' if predicted_class == 1 else 'Not Important'}")
    print(f"Confidence: {predictions[0][predicted_class].item():.2%}")

if __name__ == "__main__":
    main()