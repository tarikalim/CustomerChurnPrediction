# Customer Churn Prediction

Predict customer churn on the IBM Telco dataset by training and comparing two supervised models: Random Forest and
Logistic Regression.

---

## Pipeline

1. **Data Preprocessing**
    - Drop identifiers, geo-location, and leakage columns
    - Convert “Total Charges” to numeric & impute missing values
    - Encode binary features (Yes/No → 1/0)
    - One-hot encode remaining categorical variables


2. **Model Training & Evaluation**
    - **Random Forest**
        - Train with balanced class weights
        - Compute accuracy, precision, recall, F1, MCC
        - Save model artifact & generate PDF report (confusion matrix, ROC/PR curves, feature importances)
    - **Logistic Regression**
        - Standard scale numeric features
        - Train with balanced class weights and regularization (`C`)
        - Compute same metrics
        - Save model artifact & generate PDF report (confusion matrix, ROC/PR curves)


3. **Reports**
    - `models/artifacts/random_forest_report.pdf`
    - `models/artifacts/logistic_regression_report.pdf`

---

## Installation & Usage

1. **Clone & install dependencies**
   ```bash
   git clone https://github.com/tarikalim/CustomerChurnPrediction.git
   cd CustomerChurnPrediction
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**
   ```bash
   python main.py
   ```

3. **Check artifacts**
    - Models: `models/artifacts/*.pkl`
    - Reports: `models/artifacts/*_report.pdf`

---

##  Configuration

All paths, metrics, and hyperparameters (`RF_N_ESTIMATORS`, `LR_C`, `TEST_SIZE`, etc.) are managed in
`config.py`.


