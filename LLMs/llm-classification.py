## Predict human preference between two response from two large language models (LLMs) given the same prompt
# Data:
# Prompt, response A, response B, label indicating which response the human user preferred (or none/tie)
# Reinforcement Learning from Human Feedback (RLHF)

## load the data and check is any parsing / cleaning is required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

train = pd.read_csv("C:/Users/Shreya Tanguturi/Documents/GitHub/ML-DL-Portfolio/LLMs/train.csv")
test = pd.read_csv("C:/Users/Shreya Tanguturi/Documents/GitHub/ML-DL-Portfolio/LLMs/test.csv")
sample = pd.read_csv("C:/Users/Shreya Tanguturi/Documents/GitHub/ML-DL-Portfolio/LLMs/sample_submission.csv")

## check data types and non-null values
#print(train.info())
## shows no missing values, as non-null values for all fields are the same as number of entries (57477)

## check data structure
#print("Train shape:", train.shape)
#print("Test shape:", test.shape)


## Preprocessing & Feature Engineering
# clean text: lowercasing, removing extraneous characters 
# decide how to represent each data point: Prompt + response A, Prompt + response B, might be treated as a pair
## EDA
sns.countplot(x=train["winner_model_b"], palette="viridis")
plt.title("Class Distribution")
#plt.show()

## inspect length of responses
train["response_a_len"] = train["response_a"].apply(lambda x: len(x))
train["response_b_len"] = train["response_b"].apply(lambda x: len(x))

#print(train[["response_a_len", "response_b_len"]].describe())

## create simple "text-pair" input
def combine_text(row):
    return (
        "Prompt: " + str(row["prompt"]) +
        " Response A: " + str(row["response_a"]) +
        " Response B: " + str(row["response_b"])
    )

train["text"] = train.apply(combine_text, axis=1)
test["text"] = test.apply(combine_text, axis=1)

X = train["text"]
y = train["winner_model_a"]   # target classes: 0,1,2 maybe?

## Convert Text --> TF-IDF (Term Frequency-Inverse Document Frequency) (Baseline NLP model)
## numerical statistics that reflect how important a word is to a document in a collection. fundamental concept in information retrieval and text mining
vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
X_vec = vectorizer.fit_transform(X)
X_test_vec = vectorizer.transform(test["text"])


## Train a simple classifier --> Logistic Regression
X_train, x_val, y_train, y_val = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=20000, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict_proba(x_val)
print("Validation Log Loss: ", log_loss(y_val, preds))

val_preds = model.predict(x_val)
print("Validation Accuracy: ", accuracy_score(y_val, val_preds))

## Improve this baseline model: 
# Validation Log Loss: 0.6582402768642496
# Validation Accuracy: 0.6352644398051496

## Predict on test data + create submission
print(sample.head())

test_preds = model.predict_proba(X_test_vec)

submission = sample.copy()
#submission.iloc[:,1:] = test_preds
#submission.to_csv("submission_baseline.csv", index=False)
print(submission.head())

## use different models to fine-tune
## HuggingFace Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from trl import RewardTrainer, PPOTrainer
## trl - Transformer Reinforcement Learning

## format the dataset for transformers
## tokenize the text for the model, break down text into smaller units (tokens) so that LLM can process it
## use a pretrained tokenizer (e.g., bert-base-uncased)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(ex):
    return tokenizer(
        ex = ["text"],
        padding = "max_length",
        truncation = True,
        max_length = 256
    )

## prepare dataset for hugging face (datasets.Dataset format)

from datasets import Dataset

# Convert pandas DataFrame to Hugging Face Dataset
train_ds = Dataset.from_pandas(train)
test_ds = Dataset.from_pandas(test)
## split similar to sklearn

# Map the tokenizer over input texts
train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

print(train_ds)

## choose a model for classification -- AutoModelForSequenceClassification

num_labels = len(train["winner_model_b"].unique()) ## typically 2 or 3
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = num_labels)

## set up TrainingArguments and Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

## train the model
trainer.Train()

## evaluate and compare, use trainer.evaluate() to get accuracy, log loss, etc. and compare results to baseline logistic regression model
eval_results = trainer.evaluate()
print("Evaluation results: ", eval_results)

## Why do transformer models generally outperform TF-IDF + logistic regression on language preference tasks?
    # transformers are built to understand linguistic context, semantics, and word order
## What are some common pitfalls to watch for when fine-tuning large models on text data?
    # overfitting: large models may memorize training data, especially if it is limited, leading to poor generalization on new prompts / responses
    # tokenization issues: inconsistent tokenization parameters
    # imbalanced classes: if your dataset favours one label, model may become biased.
    # compute and resource constraints: transformer models are resource-intensive, insufficient compute limit may limit performance or make hyperparameter runs infeasible

## Mitigation:
    # apply regularization, early stopping, and augmentation to reduce overfitting