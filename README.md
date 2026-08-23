# Loan Status Prediction

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details, using a `scikit-learn` pipeline with preprocessing and Logistic Regression.

## 📌 Overview

This project uses the classic Loan Prediction dataset to build a classification model that determines loan approval status (`Loan_Status`) based on applicant and loan attributes.

## 📂 Dataset

The dataset contains the following columns:

| Column | Description |
|---|---|
| `Loan_ID` | Unique loan application ID (dropped before modeling) |
| `Gender` | Applicant's gender |
| `Married` | Marital status (dropped before modeling) |
| `Dependents` | Number of dependents (dropped before modeling) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employment status |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Loan amount requested |
| `Loan_Amount_Term` | Term of the loan (in months) |
| `Credit_History` | Credit history meets guidelines (1) or not (0) |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | Target variable — loan approved (Y) or not (N) |

## 🧹 Data Preprocessing

- Dropped irrelevant/low-value columns: `Loan_ID`, `Married`, `Dependents`
- Handled missing values:
  - Numeric columns → imputed with **median**
  - Categorical columns → imputed with **most frequent value**
- Encoded categorical variables using **One-Hot Encoding** (`drop="first"`, `handle_unknown="ignore"`)
- Scaled numeric features using **StandardScaler**

All preprocessing steps are combined into a single `ColumnTransformer`:

```python
preprocessor = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
])
```

## 🏗️ Model Pipeline

The final pipeline combines preprocessing and the model into one object, so raw input data can be passed directly for prediction:

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

## 🔮 Making Predictions

Use the `predictor()` function to test the model on new applicant data:

```python
def predictor(Gender, Education, Self_Employed, ApplicantIncome, CoapplicantIncome,
              LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    input_data = pd.DataFrame([{
        'Gender': Gender,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }])

    pred = pipeline.predict(input_data)[0]

    if pred == 'Y':
        print("Loan Approved ✅")
    else:
        print("Loan Rejected ❌")
```

### Example

```python
predictor("Male", "Graduate", "No", 5000, 1500, 128, 360, 1.0, "Urban")
```

## 🛠️ Tech Stack

- Python
- pandas
- numpy
- scikit-learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`)

## ⚙️ Setup

```bash
pip install pandas numpy scikit-learn
```

## 🚀 Usage

1. Load and clean the dataset
2. Split into features (`X`) and target (`y`)
3. Fit the pipeline on training data
4. Use `predictor()` to test new samples

## 📈 Possible Improvements

- Try other models (Random Forest, XGBoost) and compare accuracy
- Handle class imbalance if present in `Loan_Status`
- Add cross-validation and hyperparameter tuning (`GridSearchCV`)
- Add model evaluation metrics (precision, recall, F1, confusion matrix)
