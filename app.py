import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

# Enterprise Constants v3.0
DATA_PATH = 'credit_risk_dataset.csv'
CLEANED_DATA_PATH = 'final_loan_risk_results.csv'
MODEL_PATH = 'loan_risk_model.joblib'
STATIC_CHARTS_DIR = os.path.join('static', 'charts')

os.makedirs(STATIC_CHARTS_DIR, exist_ok=True)

# Enterprise Global Resources v3.0
_model = None
_scaler = None
_feature_names = None
_df_results = None
_importance = None
_stats = None
_explainer = None

def get_resources():
    global _model, _scaler, _feature_names, _df_results, _importance, _stats, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load('scaler.joblib')
        _feature_names = joblib.load('feature_names.joblib')
        _df_results = pd.read_csv(CLEANED_DATA_PATH)
        _importance = joblib.load('feature_importance.joblib')
        _stats = joblib.load('model_stats.joblib')
        _explainer = joblib.load('shap_explainer.joblib')
    return _model, _scaler, _feature_names, _df_results, _importance, _stats, _explainer

def generate_dashboard_charts(df):
    plt.style.use('dark_background')
    
    # 1. Risk Category Distribution (Bar Chart)
    plt.figure(figsize=(10, 6))
    counts = df['loan_status'].value_counts()
    sns.barplot(x=['Non-Default', 'Default'], y=counts.values, palette=['#3b82f6', '#ef4444'])
    plt.title('Risk Category Distribution', pad=20, fontsize=14)
    plt.savefig(os.path.join(STATIC_CHARTS_DIR, 'risk_distribution.png'), transparent=True, bbox_inches='tight')
    plt.close()

    # 2. Income vs Default Probability (Scatter - Sampled for performance)
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(min(1000, len(df)))
    sns.scatterplot(data=sample_df, x='person_income', y='loan_amnt', hue='loan_status', palette=['#3b82f6', '#ef4444'], alpha=0.6)
    plt.title('Income vs Loan Amount Risk Mapping', pad=20, fontsize=14)
    plt.savefig(os.path.join(STATIC_CHARTS_DIR, 'income_scatter.png'), transparent=True, bbox_inches='tight')
    plt.close()

    # 3. Credit History vs Risk (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='loan_status', y='cb_person_cred_hist_length', palette=['#3b82f6', '#ef4444'])
    plt.title('Credit History Length vs Default Status', pad=20, fontsize=14)
    plt.savefig(os.path.join(STATIC_CHARTS_DIR, 'credit_boxplot.png'), transparent=True, bbox_inches='tight')
    plt.close()
    
    # Note: ROC Curve is generated during modeling.py and saved to static/roc_curve.png

@app.route('/')
@app.route('/dashboard')
def dashboard():
    model, scaler, feature_names, df, importance, stats, explainer = get_resources()
    
    total_apps = len(df)
    approval_rate = (df['loan_status'] == 0).mean() * 100 # In raw data, 0 is no-default
    default_rate = (df['loan_status'] == 1).mean() * 100
    
    if not os.path.exists(os.path.join(STATIC_CHARTS_DIR, 'risk_distribution.png')):
        generate_dashboard_charts(df)
    
    return render_template('dashboard.html', 
                          total_apps=f"{total_apps:,}", 
                           approval_rate=f"{approval_rate:.1f}%", 
                           default_rate=f"{default_rate:.1f}%",
                           auc=f"{stats['auc']:.3f}",
                           gini=f"{stats['gini']:.3f}",
                           ks=f"{stats['ks_stat']:.3f}")

def classify_risk(pd):
    """Enterprise-grade risk band classification"""
    if pd < 0.05:
        return "Very Low Risk"
    elif pd < 0.15:
        return "Low Risk"
    elif pd < 0.25:
        return "Medium Risk"
    elif pd < 0.40:
        return "High Risk"
    else:
        return "Critical Risk"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            model, scaler, feature_names, df_results, importance, stats, explainer = get_resources()
            
            # Form Inputs
            loan_amnt = float(request.form.get('loan_amount', 10000))
            income = float(request.form.get('income', 50000))
            emp_length = float(request.form.get('emp_length', 0))
            age = float(request.form.get('age', 0))
            grade = request.form.get('loan_grade', 'A') # Match predict.html name
            home_ownership = request.form.get('home_ownership', 'RENT')
            loan_intent = request.form.get('loan_intent', 'PERSONAL')
            cred_hist = float(request.form.get('cred_hist', 0))
            loan_term = int(request.form.get('loan_term', 36))
            
            # --- Master Enterprise v3.0: Financial Intelligence ---
            INTEREST_RATE = 0.08 # 8% per annum
            LGD = 0.60 # Standard Loss Given Default
            
            # Engineering numeric grade for interaction
            grade_map = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
            grade_num = grade_map.get(grade, 1)
            
            # Interaction Terms Sync
            lti = loan_amnt / income if income > 0 else 0
            raw_input_data = {
                'person_age': age,
                'person_income': income,
                'person_emp_length': emp_length,
                'loan_amnt': loan_amnt,
                'cb_person_cred_hist_length': cred_hist,
                'loan_int_rate': df_results['loan_int_rate'].median(),
                'loan_percent_income': lti,
                'loan_term': loan_term,
                'grade_num': grade_num,
                'inter_grade_cred': grade_num * cred_hist,
                'inter_lti_term': lti * loan_term,
                'inter_incratio_term': lti * loan_term
            }
            
            # Explicit One-Hot for Category Matches
            input_df = pd.DataFrame([raw_input_data])
            if f'person_home_ownership_{home_ownership}' in feature_names:
                input_df[f'person_home_ownership_{home_ownership}'] = 1
            if f'loan_intent_{loan_intent}' in feature_names:
                input_df[f'loan_intent_{loan_intent}'] = 1
            
            # Align and Scale
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            input_df_scaled = scaler.transform(input_df)
            
            # Prediction [0.01 - 0.985]
            prob = model.predict_proba(input_df_scaled)[0][1]
            prob = min(max(prob, 0.01), 0.985)
            
            # --- Enterprise v3.0: Risk Bands & Logic ---
            risk_band = classify_risk(prob)
            threshold = 0.50 # Default baseline
            
            # Tiered Decision Logic
            if prob < threshold and prob < 0.25:
                decision = "APPROVE"
            elif prob < threshold and prob < 0.40:
                decision = "REVIEW REQUIRED"
            else:
                decision = "REJECT"

            # --- Master v3.0 Financials ---
            expected_loss = prob * loan_amnt * LGD
            interest_income = loan_amnt * INTEREST_RATE * (loan_term/12)
            roi = (interest_income - expected_loss) / loan_amnt if loan_amnt > 0 else 0
            
            # Financial Safety Rule: Expected Loss Safety Cap
            safety_cap = expected_loss > loan_amnt * 0.20
            if safety_cap:
                decision = "REJECT"
                risk_band = "Critical Risk (Safety Cap Triggered)"

            decision_status = f"{decision} – {risk_band}"

            # --- Color Mapping based on Risk Band ---
            if prob < 0.15: # Very Low or Low
                risk_color = "#16a34a" # Green
            elif prob < 0.25: # Medium
                risk_color = "#f59e0b" # Yellow
            else: # High or Critical
                risk_color = "#dc2626" # Red

            # --- SHAP Explainability ---
            shap_output = explainer(input_df_scaled)
            
            # Generate Individual SHAP Waterfall Plot
            plt.figure(figsize=(10, 4))
            exp = shap.Explanation(values=shap_output.values[0], 
                                   base_values=shap_output.base_values[0], 
                                   data=input_df.iloc[0], 
                                   feature_names=feature_names)
            
            shap.plots.waterfall(exp, show=False)
            plt.gcf().set_facecolor('none')
            plt.title('Risk Contribution Analysis (SHAP)', color='white')
            plt.tight_layout()
            shap_path = os.path.join(STATIC_CHARTS_DIR, 'latest_shap.png')
            plt.savefig(shap_path, transparent=True)
            plt.close()
            
            factors = []
            val_items = []
            for i, name in enumerate(feature_names):
                val = shap_output.values[0][i]
                if abs(val) > 0.01:
                    val_items.append((name, val))
            val_items.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for name, val in val_items[:6]: # Show more factors for enterprise depth
                color = "text-danger" if val > 0 else "text-success"
                clean_name = name.replace('person_', '').replace('loan_', '').replace('_', ' ').title()
                factors.append({
                    'name': clean_name, 
                    'raw_name': name,
                    'val': f"{val:+.2f}", 
                    'color': color
                })

            return render_template('predict.html', 
                                 prediction=True, prob=prob, 
                                 probability_percent=round(prob * 100, 1),
                                 risk_color=risk_color,
                                 risk_cat=risk_band, decision=decision, 
                                 decision_status=decision_status,
                                 safety_cap=safety_cap, # Pass flag
                                 expected_loss=f"${expected_loss:,.0f}",
                                 roi=f"{roi*100:.1f}%",
                                 factors=factors,
                                 form_data=request.form)
                                 
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('predict.html', error=f"Intelligence Exception: {str(e)}", form_data=request.form)
            
    return render_template('predict.html', prediction=False)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Enterprise REST API v3.0"""
    try:
        model, scaler, feature_names, df_results, importance, stats, explainer = get_resources()
        data = request.get_json()
        
        income = float(data.get('income', 50000))
        loan_amnt = float(data.get('loan_amount', 10000))
        grade = data.get('grade', 'A')
        home_ownership = data.get('home_ownership', 'RENT')
        loan_intent = data.get('loan_intent', 'PERSONAL')
        loan_term = int(data.get('loan_term', 36))
        
        # Internal Interaction Logic (Replicated from main predict for parity)
        grade_map = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
        grade_num = grade_map.get(grade, 1)
        lti = loan_amnt / income if income > 0 else 0
        
        input_data = {
            'person_age': float(data.get('age', 30)),
            'person_income': income,
            'person_emp_length': float(data.get('emp_length', 5)),
            'loan_amnt': loan_amnt,
            'cb_person_cred_hist_length': float(data.get('cred_hist', 5)),
            'loan_int_rate': df_results['loan_int_rate'].median(),
            'loan_percent_income': lti,
            'loan_term': loan_term,
            'grade_num': grade_num,
            'inter_grade_cred': grade_num * float(data.get('cred_hist', 5)),
            'inter_lti_term': lti * loan_term,
            'inter_incratio_term': lti * loan_term
        }
        
        # Explicit One-Hot for Category Matches in API
        input_df = pd.DataFrame([input_data])
        if f'person_home_ownership_{home_ownership}' in feature_names:
            input_df[f'person_home_ownership_{home_ownership}'] = 1
        if f'loan_intent_{loan_intent}' in feature_names:
            input_df[f'loan_intent_{loan_intent}'] = 1
            
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_df_scaled = scaler.transform(input_df)
        
        prob = model.predict_proba(input_df_scaled)[0][1]
        bounded_prob = min(max(prob, 0.01), 0.985)
        
        # --- Financial Metrics for API ---
        expected_loss = bounded_prob * loan_amnt * 0.60
        interest_income = loan_amnt * 0.08 * (loan_term/12)
        roi = (interest_income - expected_loss) / loan_amnt if loan_amnt > 0 else 0
        
        # SHAP for API
        shap_output = explainer(input_df_scaled)
        
        return jsonify({
            "probability": round(bounded_prob, 4),
            "decision": "APPROVE" if bounded_prob < 0.35 else "CONDITIONAL APPROVAL" if bounded_prob < 0.55 else "REJECT",
            "expected_loss": round(expected_loss, 2),
            "roi_annual": round(roi * 100, 2),
            "risk_tier": "LOW" if bounded_prob < 0.35 else "MEDIUM" if bounded_prob < 0.55 else "HIGH",
            "shap_explainability": {feature_names[i]: float(shap_output.values[0][i]) for i in range(len(feature_names)) if abs(shap_output.values[0][i]) > 0.05}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/simulate')
def simulate():
    """Master Portfolio Simulation (500-Loan Standard)"""
    try:
        model, scaler, feature_names, df, importance, stats, explainer = get_resources()
        sim_df = df.sample(500, replace=True).copy()
        
        # Proxy metrics for dashboard alignment
        avg_risk = df['loan_status'].mean()
        total_exposure = sim_df['loan_amnt'].sum()
        expected_defaults = int(500 * avg_risk)
        total_expected_loss = total_exposure * avg_risk * 0.60
        
        return jsonify({
            "simulation_size": 500,
            "total_exposure": f"${total_exposure:,.0f}",
            "expected_defaults": expected_defaults,
            "total_expected_loss": f"${total_expected_loss:,.0f}",
            "portfolio_health": "STABLE" if avg_risk < 0.22 else "WATCH"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
