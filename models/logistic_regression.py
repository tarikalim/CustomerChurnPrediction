"""
This module implements a Logistic Regression classifier utilizing the BaseModel framework.
It handles data loading, model training, evaluation, and reporting.

The model uses StandardScaler preprocessing and balanced class weights to handle
potential class imbalance in the churn dataset.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from config import (
    OUTPUT_CSV_PATH, TEST_SIZE, RANDOM_STATE,
    LR_C, LR_MAX_ITER, LR_MODEL_PATH, LR_REPORT_PATH
)
from models.base_model import BaseModel
from evaluation.plots import EvaluationReport


def load_processed_data():
    """
    Load cleaned CSV and split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test: Split dataset components
    """
    df = pd.read_csv(OUTPUT_CSV_PATH)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model implementation for churn prediction.

    This class extends BaseModel to provide a logistic regression classifier
    with standardization preprocessing.
    """

    def __init__(self):
        lr = LogisticRegression(
            C=LR_C,
            class_weight='balanced',  # Important for handling imbalanced churn data
            random_state=RANDOM_STATE,
            max_iter=LR_MAX_ITER
        )
        # Pipeline: scaling + logistic regression
        pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('model', lr)
        ])
        super().__init__(pipeline, model_name='logistic_regression', random_state=RANDOM_STATE)


def main_lr():
    """
    Main function to run the logistic regression modeling pipeline.

    This function:
    1. Loads and splits the processed data
    2. Trains the logistic regression model
    3. Evaluates model performance with multiple metrics
    4. Saves the model and generates an evaluation report
    """
    # Load and split data
    X_train, X_test, y_train, y_test = load_processed_data()
    # Train model
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)

    # Evaluate metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef
    }
    results = model.evaluate(X_test, y_test, metrics)
    print('Logistic Regression Results:', results)

    model.save(LR_MODEL_PATH)
    print(f" Logistic Regression model saved at {LR_MODEL_PATH}")

    # Generate evaluation report PDF
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = EvaluationReport(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_importances=None,
        feature_names=None
    )
    report.save_report(LR_REPORT_PATH)