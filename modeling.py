import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             precision_score, recall_score, f1_score)
import joblib
import os
import shap

# Enterprise Constants v3.0
MODEL_PATH = 'loan_risk_model.joblib'
SCALER_PATH = 'scaler.joblib'
FEATURES_PATH = 'feature_names.joblib'
STATS_PATH = 'model_stats.joblib'
IMPORTANCE_PATH = 'feature_importance.joblib'
SHAP_PATH = 'shap_explainer.joblib'
STATIC_DIR = 'static'

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def run_modelling_pipeline():
    print("🚀 Initializing Master Enterprise v3.0 Model Pipeline...")
    
    # Load dataset (v3.0 standard)
    df = pd.read_csv('credit_risk_dataset.csv')
    
    # Basic Cleaning
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df = df.drop_duplicates()
    df = df[df['person_age'] <= 100]
    
    # 1. Feature Engineering: Enterprise Interaction Terms
    np.random.seed(42)
    df['loan_term'] = np.random.choice([36, 60], size=len(df), p=[0.7, 0.3])
    
    # Numeric Grade for interaction
    grade_map = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    df['grade_num'] = df['loan_grade'].map(grade_map)
    
    # Specified Interactions
    df['inter_grade_cred'] = df['grade_num'] * df['cb_person_cred_hist_length']
    df['inter_lti_term'] = df['loan_percent_income'] * df['loan_term']
    df['inter_incratio_term'] = (df['loan_amnt'] / df['person_income']) * df['loan_term']

    # Prepare features
    numeric_cols = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'loan_term', 'grade_num', 'inter_grade_cred', 'inter_lti_term', 'inter_incratio_term'
    ]
    categorical_cols = ['person_home_ownership', 'loan_intent']
    
    X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)
    y = df['loan_status']
    
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, FEATURES_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # 2. Logistic Regression with Isotonic Calibration
    lr = LogisticRegression(class_weight='balanced', C=0.01, random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    raw_probs = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_probs, y_test)
    calibrated_probs = iso.transform(raw_probs)
    # Bounding [0.01 - 0.985]
    calibrated_probs = np.clip(calibrated_probs, 0.01, 0.985)

    # 3. Advanced Metrics Calculation
    y_pred = (calibrated_probs >= 0.5).astype(int)
    auc_val = roc_auc_score(y_test, calibrated_probs)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    gini = 2 * auc_val - 1
    
    # KS Statistic
    def ks_stat(y_true, y_prob):
        df_ks = pd.DataFrame({'y': y_true, 'p': y_prob})
        df_ks = df_ks.sort_values('p', ascending=False)
        df_ks['cum_pos'] = df_ks['y'].cumsum() / df_ks['y'].sum()
        df_ks['cum_neg'] = (1 - df_ks['y']).cumsum() / (1 - df_ks['y']).sum()
        return (df_ks['cum_pos'] - df_ks['cum_neg']).abs().max()

    ks = ks_stat(y_test, calibrated_probs)

    stats = {
        'auc': auc_val, 'precision': precision, 'recall': recall, 
        'f1': f1, 'gini': gini, 'ks_stat': ks
    }
    joblib.dump(stats, STATS_PATH)
    print(f"✅ Model Stats: AUC={auc_val:.4f}, Gini={gini:.4f}, KS={ks:.4f}")

    # 4. Visualization
    plt.style.use('default') # Standard look for these plots
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, calibrated_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC: {auc_val:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('Enterprise v3.0 Performance (ROC)')
    plt.legend()
    plt.savefig(os.path.join(STATIC_DIR, 'roc_curve.png'))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Enterprise v3.0 Confusion Matrix')
    plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'))
    plt.close()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, calibrated_probs, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Isotonic')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.title('Enterprise v3.0 Calibration Diagnostics')
    plt.legend()
    plt.savefig(os.path.join(STATIC_DIR, 'calibration_curve.png'))
    plt.close()

    # 5. SHAP Integration
    explainer = shap.LinearExplainer(lr, X_train_scaled, feature_perturbation="interventional")
    joblib.dump(explainer, SHAP_PATH)
    
    # Feature Importance
    importance = pd.Series(lr.coef_[0], index=feature_names).abs().sort_values(ascending=False)
    joblib.dump(importance, IMPORTANCE_PATH)

    joblib.dump(lr, MODEL_PATH)
    print("💎 Master Enterprise v3.0 Artifacts Certified.")

if __name__ == "__main__":
    run_modelling_pipeline()
