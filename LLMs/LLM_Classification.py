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

train = pd.read_csv("C:/Users/Shreya Tanguturi/Desktop/PythonPersonalProjects/llm-classification-finetuning/train.csv")
test = pd.read_csv("C:/Users/Shreya Tanguturi/Desktop/PythonPersonalProjects/llm-classification-finetuning/test.csv")
sample = pd.read_csv("C:/Users/Shreya Tanguturi/Desktop/PythonPersonalProjects/llm-classification-finetuning/sample_submission.csv")

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
