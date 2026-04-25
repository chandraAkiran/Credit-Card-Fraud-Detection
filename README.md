# Credit Card Fraud Detection

## Overview
This project uses Machine Learning to detect fraudulent credit card transactions using Python and the Credit Card Fraud dataset.

## Tools Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

## Project Steps
- Data Cleaning and Exploration  
- Fraud vs Non-Fraud Analysis  
- Handling Imbalanced Data  
- Random Forest Model Building  
- Model Evaluation  

## Dataset
- 284,807 transactions  
- 492 fraud cases  
- Features: V1–V28, Time, Amount, Class  

## Model
Used **Random Forest Classifier** for fraud detection.

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- ROC Curve  
- AUC Score  

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 99.96% |
| Precision | 0.93 |
| Recall | 0.82 |
| F1 Score | 0.87 |
| ROC-AUC | 0.97 |

## ROC-AUC Insight
- **ROC-AUC Score: 0.97** indicates excellent model ability to distinguish fraudulent and legitimate transactions.  
- High AUC suggests strong classification performance even with imbalanced data.

## Project Structure
```bash
├── Credit_Card_Fraud_Detection.ipynb
├── creditcard.csv
└── README.md
```

## Run the Project
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook
```
## Author
Your Name  
GitHub: https://github.com/yourusername
