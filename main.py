from config import TEST_SIZE, RANDOM_STATE, RF_MODEL_PATH
from preprocessing.preprocess_data import preprocess_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from models.random_forest import RandomForestModel


def main():
    df = preprocess_file()
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    rf_results = rf.evaluate(X_test, y_test, {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef
    })
    print('Random Forest Results:', rf_results)

    rf.save(RF_MODEL_PATH)
    print(f"Models saved: RF->{RF_MODEL_PATH}")


if __name__ == '__main__':
    main()
