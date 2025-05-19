import os

"""
Global configuration file to manage all constants and paths easily.
"""
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_EXCEL_PATH = os.path.join(DATA_DIR, 'Telco_customer_churn.xlsx')
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, 'Telco_customer_churn_cleaned.csv')
RF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'artifacts', 'random_forest.pkl')
RF_REPORT_PATH = os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'random_forest_results.pdf')
LR_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'artifacts', 'logistic_regression.pkl')
LR_REPORT_PATH = os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'logistic_regression_results.pdf')
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None

# Logistic Regression parameters
LR_C = 1.0
LR_MAX_ITER = 1000