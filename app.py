from flask import Flask, render_template, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# ── Model loading (lazy) ──────────────────────────────────────────────────────
# Loaded on first request so gunicorn workers don't all load at startup.
_model = None
_feature_list = None
_label_encoders = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_artifacts():
    global _model, _feature_list, _label_encoders
    if _model is None:
        _model = joblib.load(os.path.join(BASE_DIR, 'pl_model.pkl'))
        _feature_list = joblib.load(os.path.join(BASE_DIR, 'pl_features.pkl'))
        _label_encoders = joblib.load(os.path.join(BASE_DIR, 'pl_encoders.pkl'))
    return _model, _feature_list, _label_encoders


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_rate(cibil, emp_type):
    if cibil >= 800:
        rate = 10.5
    elif cibil >= 750:
        rate = 11.5
    elif cibil >= 700:
        rate = 13.0
    elif cibil >= 650:
        rate = 15.0
    else:
        rate = 17.5
    if emp_type == 'Self-Employed':
        rate += 1.0
    return round(rate, 1)


def calc_emi(principal, annual_rate, months):
    r = annual_rate / 100 / 12
    if r == 0:
        return round(principal / months)
    emi = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    return round(emi)


def get_max_loan(income, cibil, existing_emi, expenses, interest_rate, tenure):
    max_emi_capacity = (income * 0.40) - existing_emi
    disposable = income - expenses - existing_emi
    disposable_cap = disposable * 0.30
    affordable_emi = max(0, min(max_emi_capacity, disposable_cap))
    if affordable_emi <= 0:
        return 0
    r = interest_rate / 100 / 12
    if r == 0:
        loan_from_emi = affordable_emi * tenure
    else:
        loan_from_emi = affordable_emi * ((1 + r) ** tenure - 1) / (r * (1 + r) ** tenure)
    if cibil >= 750:
        cibil_cap = income * 20
    elif cibil >= 700:
        cibil_cap = income * 15
    elif cibil >= 650:
        cibil_cap = income * 10
    else:
        cibil_cap = income * 6
    max_loan = min(loan_from_emi, cibil_cap)
    return max(0, int(round(max_loan / 5000) * 5000))


def build_feedback(income, dti, cibil, loan_amt, work_exp, emp_type, emp_cat,
                   education, existing_emi, expenses):
    strengths, problems, tips = [], [], []
    loan_to_income = loan_amt / income if income > 0 else 99

    if cibil >= 750:
        strengths.append(f"Excellent CIBIL score ({cibil}) \u2014 strongly favours approval")
    elif cibil >= 700:
        strengths.append(f"Good CIBIL score ({cibil})")
    elif cibil >= 650:
        problems.append(f"Average CIBIL score ({cibil}) \u2014 lenders prefer 700+")
        tips.append("Pay all EMIs and credit card bills on time for 6-12 months to push CIBIL above 700")
    else:
        problems.append(f"Low CIBIL score ({cibil}) \u2014 banks see this as high default risk")
        tips.append("Check your CIBIL report for errors. Clear any overdue loans first")

    if dti < 0.35:
        strengths.append(f"Healthy debt-to-income ratio ({dti:.0%}) \u2014 good repayment capacity")
    elif dti < 0.50:
        strengths.append(f"Acceptable DTI ({dti:.0%})")
    elif dti < 0.65:
        problems.append(f"High DTI ({dti:.0%}) \u2014 over 50% of income is going to obligations")
        tips.append("Close one existing EMI before applying \u2014 it will bring DTI below 50%")
    else:
        problems.append(f"Very high DTI ({dti:.0%}) \u2014 almost no buffer left for new repayment")
        tips.append("Clear at least one existing loan before reapplying")

    if income >= 100000:
        strengths.append(f"Strong monthly income (\u20b9{income:,.0f}) \u2014 good repayment ability")
    elif income >= 50000:
        strengths.append(f"Decent income (\u20b9{income:,.0f})")
    elif income >= 25000:
        problems.append(f"Moderate income (\u20b9{income:,.0f}) \u2014 limits how much you can borrow")
        tips.append("Adding a co-applicant with income can increase your eligible amount")
    else:
        problems.append(f"Low income (\u20b9{income:,.0f}) \u2014 significantly restricts eligibility")
        tips.append("Consider a smaller loan or a secured loan against FD or property")

    if loan_to_income < 5:
        strengths.append(f"Requested amount is {loan_to_income:.1f}x monthly income \u2014 very manageable")
    elif loan_to_income >= 15:
        problems.append(f"Requested amount is {loan_to_income:.1f}x monthly income \u2014 too high for your income")
        tips.append("Reduce the loan amount or increase tenure to lower the EMI burden")
    elif loan_to_income >= 10:
        problems.append(f"Requested amount is {loan_to_income:.1f}x monthly income \u2014 on the higher side")
        tips.append("Request a lower loan amount or longer tenure")

    if emp_cat in ['Government', 'MNC']:
        strengths.append(f"{emp_cat} employee \u2014 lenders trust this income stability")
    elif emp_cat == 'Startup':
        problems.append("Startup employment is seen as higher risk by lenders")
        tips.append("Switching to a more stable company will improve your chances")

    if emp_type == 'Self-Employed':
        tips.append("File ITR for at least 2 years \u2014 lenders need proof of stable self-employment income")

    if work_exp >= 5:
        strengths.append(f"{work_exp} years of experience shows career stability")
    elif work_exp < 2:
        problems.append("Less than 2 years experience \u2014 lenders prefer a stable job history")
        tips.append("Complete 2 years in your current role before reapplying")

    if education in ['Master', 'PhD']:
        strengths.append(f"{education}'s degree suggests higher earning potential")

    return strengths, problems, tips


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, feature_list, label_encoders = load_artifacts()
        data = request.get_json() if request.is_json else request.form.to_dict()

        income       = float(data['Net_Monthly_Income'])
        expenses     = float(data['Monthly_Expenses'])
        existing_emi = float(data.get('Existing_EMI', 0) or 0)
        cibil        = int(data['CIBIL_Score'])
        loan_amt     = float(data['Loan_Amount_Requested'])
        tenure       = int(data['Loan_Tenure_Months'])
        emp_type     = data['Employment_Type']
        emp_cat      = data['Employer_Category']
        education    = data['Education']
        work_exp     = int(data['Work_Experience_Yrs'])

        dti = round((expenses + existing_emi) / income, 3) if income > 0 else 0.99

        row = {
            'Age':                   int(data['Age']),
            'Gender':                label_encoders['Gender'][data['Gender']],
            'Marital_Status':        label_encoders['Marital_Status'][data['Marital_Status']],
            'Dependents':            int(data['Dependents']),
            'Education':             label_encoders['Education'][education],
            'Employment_Type':       label_encoders['Employment_Type'][emp_type],
            'Employer_Category':     label_encoders['Employer_Category'][emp_cat],
            'Work_Experience_Yrs':   work_exp,
            'City_Tier':             label_encoders['City_Tier'][data['City_Tier']],
            'Net_Monthly_Income':    income,
            'Monthly_Expenses':      expenses,
            'Existing_EMI':          existing_emi,
            'Debt_To_Income_Ratio':  dti,
            'CIBIL_Score':           cibil,
            'Loan_Amount_Requested': loan_amt,
            'Loan_Tenure_Months':    tenure,
        }

        input_df   = pd.DataFrame([row], columns=feature_list)
        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        confidence = round(float(max(proba)) * 100, 1)
        approved   = bool(prediction == 1)

        interest_rate    = get_rate(cibil, emp_type)
        max_loan         = get_max_loan(income, cibil, existing_emi, expenses, interest_rate, tenure)
        emi_on_requested = calc_emi(loan_amt, interest_rate, tenure)
        emi_on_max       = calc_emi(max_loan, interest_rate, tenure) if max_loan > 0 else 0

        strengths, problems, tips = build_feedback(
            income, dti, cibil, loan_amt, work_exp,
            emp_type, emp_cat, education, existing_emi, expenses
        )

        return jsonify({
            'approved':         approved,
            'confidence':       confidence,
            'dti':              round(dti * 100, 1),
            'interest_rate':    interest_rate,
            'max_eligible':     max_loan,
            'emi_on_requested': emi_on_requested,
            'emi_on_max':       emi_on_max,
            'strengths':        strengths,
            'reasons':          problems,
            'tips':             tips,
        })

    except KeyError as e:
        return jsonify({'error': f'Missing or invalid field: {e}'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'GradientBoostingClassifier'})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
