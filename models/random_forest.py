"""
This module implements a Random Forest classifier using the BaseModel framework.
It handles data loading, model training, evaluation, and reporting.

Key features:
- No scaling required (unlike logistic regression)
- Feature importance extraction
- Balanced class weights for handling imbalanced churn data
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from config import OUTPUT_CSV_PATH, TEST_SIZE, RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MODEL_PATH, \
    RF_REPORT_PATH
from evaluation.plots import EvaluationReport
from models.base_model import BaseModel


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


class RandomForestModel(BaseModel):
    """
    Random Forest model implementation for churn prediction.

    This class extends BaseModel to provide a random forest classifier.
    Note: Unlike LogisticRegression, no scaling is required in the pipeline.
    """

    def __init__(self):
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        pipeline = Pipeline([
            ('model', rf)
        ])
        super().__init__(pipeline, model_name='random_forest', random_state=RANDOM_STATE)


def main_rf():
    """
    Main function to run the random forest modeling pipeline.

    This function:
    1. Loads and splits the processed data
    2. Trains the random forest model
    3. Evaluates model performance with multiple metrics
    4. Saves the model and generates an evaluation report
    5. Extracts feature importances (a key advantage over logistic regression)
    """
    X_train, X_test, y_train, y_test = load_processed_data()
    model = RandomForestModel()
    model.fit(X_train, y_train)

    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef
    }
    results = model.evaluate(X_test, y_test, metrics)
    print('Random Forest Results:', results)

    model.save(RF_MODEL_PATH)
    print(f"Random Forest model saved at {RF_MODEL_PATH}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fi = model.pipeline.named_steps['model'].feature_importances_
    fn = X_test.columns

    report = EvaluationReport(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_importances=fi,
        feature_names=fn
    )
    report.save_report(RF_REPORT_PATH)