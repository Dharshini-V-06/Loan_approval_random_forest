##  Loan Approval Prediction using Machine Learning

###  Overview

This project predicts whether a loan application will be **approved or rejected** based on applicant details such as age, income, employment experience, credit score, and loan intent.

A **Random Forest Classifier** was trained on a dataset of 45,000 records, achieving over **92% accuracy**. The project also includes an **interactive Streamlit web app** for real-time loan approval predictions.

---

Dataset link: [Loan approval dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data )

App link: [Loan approval prediction app](https://loanapprovalrandomforest-d9fx7hftxtvdjdjjcvuyyv.streamlit.app/)

###  Project Structure

```
 Loan_Prediction_Project
│
├── loan_data.csv             # Original dataset
├── loan.ipynb                # Jupyter notebook (data cleaning, model training)
├── loan_approval.pkl         # Saved trained model + label encoder
├── app.py                    # Streamlit web app
└── README.md                 # Project documentation
```

---

###  Steps Performed

#### 1. Data Preprocessing

* Loaded dataset `loan_data.csv` with 45,000 entries.
* Checked for null values, duplicates, and outliers.
* Handled outliers using the IQR method.
* Filtered unrealistic ages (`18 ≤ age ≤ 80`).
* Label encoded categorical variables such as gender, education, and loan intent.

#### 2. Model Building

* Split dataset into training (80%) and testing (20%) sets.
* Trained a **RandomForestClassifier** with:

  ```python
  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
  ```
* Evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

#### 3. Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 92.86% |
| Precision | 89.15% |
| Recall    | 77.30% |
| F1 Score  | 82.81% |
| ROC-AUC   | 0.87   |

#### 4. Model Saving

Both the trained model and label encoder were saved as:

```python
import pickle
with open('loan_approval.pkl', 'wb') as f:
    pickle.dump((le, rf), f)
```

#### 5. Streamlit Deployment

A web app (`app.py`) was built using Streamlit to allow users to:

* Enter applicant details.
* Predict loan approval status.
* View model confidence scores.

---

###  Features of the Streamlit App

* User-friendly UI with labeled inputs.
* Dropdown menus for categorical features (education, intent, ownership).
* Real-time prediction of loan status (Approved / Rejected).
* Displays model confidence percentage.

---

###  Example Input

| Feature               | Example Value |
| --------------------- | ------------- |
| Person Age            | 28            |
| Gender                | Male          |
| Education             | Master        |
| Annual Income         | 75,000        |
| Employment Experience | 4             |
| Home Ownership        | RENT          |
| Loan Amount           | 10,000        |
| Loan Intent           | PERSONAL      |
| Loan Interest Rate    | 12.5          |
| Loan Percent Income   | 0.15          |
| Credit History Length | 6             |
| Credit Score          | 680           |
| Previous Defaults     | No            |

**Prediction Output:**  Loan Approved with 93.7% confidence

---

###  Technologies Used

* **Python 3.10+**
* **Pandas, NumPy, Scikit-learn**
* **Matplotlib, Seaborn**
* **Streamlit**
