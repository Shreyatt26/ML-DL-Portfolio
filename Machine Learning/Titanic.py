## numerical & data handling
import numpy as np
import pandas as pd

## visualization
import matplotlib.pyplot as plt
import seaborn as sns

## machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import streamlit as st

## Train.csv will contain the details of a subset of the passengers on board (891 to be exact)
# and importantly, will reveal whether they survived or not, also known as the “ground truth”.

## The test.csv dataset contains similar information but does not disclose the “ground truth”
# for each passenger. It’s your job to predict these outcomes.

## Using the patterns you find in the train.csv data, predict whether the other 418 passengers
# on board (found in test.csv) survived.

## Load and explore the data using pandas (read train.csv and test.csv)

@st.cache_data
def load_data():
    train1 = pd.read_csv("C:/Users/Shreya Tanguturi/Documents/GitHub/ML-DL-Portfolio/Machine Learning/train.csv")
    test1 = pd.read_csv("C:/Users/Shreya Tanguturi/Documents/GitHub/ML-DL-Portfolio/Machine Learning/test.csv")
    return train1, test1

# check for missing values, distribution of features, how many survived vs died
# Cabin has missing values

@st.cache_data
def preprocess_and_train():
    train1, test1 = load_data()
    # Extract the deck letter from the cabin, only first letter
    train1['Deck'] = train1['Cabin'].str[0]
    train1['Deck'] = train1['Deck'].fillna('U')

    test1['Deck'] = test1['Cabin'].str[0]
    test1['Deck'] = test1['Deck'].fillna('U')


    ## encode Deck column as categorical for the model, as number (e.g., 1, 2, 8)
    train1['Deck'] = train1['Deck'].astype('category').cat.codes
    test1['Deck'] = test1['Deck'].astype('category').cat.codes

    ## Age has many missing values, use fillna()

    ## fill missing numerical values (e.g., Age, Fare) with a median
    train1['Age'] = train1['Age'].fillna(train1['Age'].median())
    train1['Fare'] = train1['Fare'].fillna(train1['Fare'].median())

    test1['Age'] = test1['Age'].fillna(test1['Age'].median())
    test1['Fare'] = test1['Fare'].fillna(test1['Fare'].median())

    ## fill missing categorical values (e.g., Deck, Sex) with a placeholder
    train1['Deck'] = train1['Deck'].fillna('U')
    train1['Sex'] = train1['Sex'].fillna('unknown')

    test1['Deck'] = test1['Deck'].fillna('U')
    test1['Sex'] = test1['Sex'].fillna('unknown')
    
    # Select features (x) and target (y)
    x = train1[['Pclass', 'Sex', 'Age', 'Fare', 'Deck']]
    y = train1['Survived']

    x_test_final = test1[['Pclass', 'Sex', 'Age', 'Fare', 'Deck']]

    ## convert categorical features like 'Sex' to numeric
    x = pd.get_dummies(x, drop_first=True)
    x_test_final = pd.get_dummies(x_test_final, drop_first=True)

    # split data into training and testing sets to evaluate model's performance on unseen data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    ## align test features with training features
    x_test_final = x_test_final.reindex(columns=x_train.columns, fill_value=0)

    # initialize and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    #Predict
    y_pred_test = model.predict(x_test_final)
    print(y_pred_test)

    submission = pd.DataFrame({
        'PassengerId': test1['PassengerId'],
        'Survived':y_pred_test
    })

    ## Metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)

    return {
        "train": train1,
        "test": test1,
        "model": model,
        "submission": submission,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": cls_report,
        }
    }

def main():
    st.title("Titanic Survival Prediction (Logistic Regression)")
    st.write("Using the Titanic dataset and a logistic regression model to predict which passengers survived")

    st.header("1. Data Overview")
    train1, test1 = load_data()

    st.subheader("Train data (first 5 rows)")
    st.dataframe(train1.head())

    st.subheader("Test data (first 5 rows)")
    st.dataframe(test1.head())

    if st.checkbox("Show basic info about missing values"):
        st.subheader("Missing values in train.csv")
        st.write(train1.isna().sum())
        st.subheader("Missing values in test.csv")
        st.write(test1.isna().sum())

    st.header("2. Train Model & Evaluate")
    if st.button("Train / Retrain model"):
        with st.spinner("Training model..."):
            results = preprocess_and_train()

        metrics = results["metrics"]

        st.success("Model trained!")

        st.subheader("Validation Metrics (on 20% hold-out)")
        st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
        st.write(f"**Precision:** {metrics['precision']:.3f}")
        st.write(f"**Recall:** {metrics['recall']:.3f}")
        st.write(f"**F1-score:** {metrics['f1']:.3f}")

        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]

        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt = "d", 
            xticklabels=["Pred 0 (died)", "Pred 1 (survived)"],
            yticklabels=["Actual 0 (died)", "Actual 1 (survived)"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(metrics["classification_report"])

        st.header("3. Submission File")
        submission = results["submission"]
        st.write("Preview of submission: ")
        st.dataframe(submission.head())

        csv_data = submission.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Train / Retrain model** to run the pipeline and see results.")


if __name__ == "__main__":
    main()






#submission.to_csv("submission.csv", index=False)
#print("submission.csv created successfully!")


## Notes ##
## output means it is predicting one value per passenger in the test set
# 0 = model predicts passenger did not survive
# 1 = model predicts passenger did survive

## check model accuracy
#accuracy = accuracy_score(y_test, y_pred_test)
#print("Model accuracy:", accuracy) ## 0.81, 81% of total predictions were correct

## view confusion matrix
#print(confusion_matrix(y_test, y_pred_test))

##                          Predicted 0 (died)          Predicted 1 (survived)
##    Actual 0 (died)               TN = 93                      FP = 12
##    Actual 1 (survived)           FN = 42                      TP = 32

## model accuracy is around 70%, which is reasonable for a simple model and only one feature
## True positives are low because Deck feature doesn't have strong predictive power for survival, 
# test other features to boost performance

##                          Predicted 0 (died)          Predicted 1 (survived)
##    Actual 0 (died)               TN = 89                      FP = 16
##    Actual 1 (survived)           FN = 18                      TP = 56
# model accuracy is 81%, classes are not imbalanced, more passengers died than survival

## adding more features (Sex, Age, Fare, Deck) made the model more sensitive to identifying survivors, it is better
## at detecting positives (Survived = 1)

# how many predicted survivors were actually survivors
#precision = precision_score(y_test, y_pred)
# how many actual survivors were correctly predicted
#recall = recall_score(y_test, y_pred)
# the harmonic mean (balance) between them
#f1 = f1_score(y_test, y_pred)

#print(f"Precision: {precision:.2f}")  ## 0.78, of all passengers predicted as survived, 78% actually survived
#print(f"Recall: {recall:.2f}")  ## 0.76, of all actual survivors, 76% were correctly predicted
#print(f"F1-score: {f1:.2f}")  ## 0.77

## if accuracy > precision & recall, the model might be biased toward the majority class (predicting "died" more often)
## if F1 ~ 0.75+ with accuracy around 0.8, the model is still quite strong for logistic regression

## to balance precision and recall better
# Feature scaling (since Fare and Age have different ranges)
#scaler = StandardScaler()

#x_train_scaled = scaler.fit_transform(x_train)
#x_test_scaled = scaler.fit_transform(x_test)
