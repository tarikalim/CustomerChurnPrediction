import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from config import OUTPUT_CSV_PATH, TEST_SIZE, RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MODEL_PATH
from models.base_model import BaseModel


def load_processed_data():
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
    print(f"Model saved at {RF_MODEL_PATH}")
