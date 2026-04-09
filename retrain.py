"""
retrain.py – Regenerate synthetic data and retrain the Personal Loan model.
Usage:
    python retrain.py               # generates fresh synthetic data + retrains
    python retrain.py --data my.csv # trains on your own CSV (must have correct columns)
"""
import argparse, numpy as np, pandas as pd, joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

REQUIRED_COLS = [
    'Age','Gender','Marital_Status','Dependents','Education',
    'Employment_Type','Employer_Category','Work_Experience_Yrs',
    'City_Tier','Net_Monthly_Income','Monthly_Expenses','Existing_EMI',
    'Debt_To_Income_Ratio','CIBIL_Score','Loan_Amount_Requested',
    'Loan_Tenure_Months','Loan_Approved'
]

def generate_synthetic(n=6000, seed=42):
    np.random.seed(seed)
    gender    = np.random.choice(['Male','Female'], n, p=[0.58,0.42])
    age       = np.random.randint(21,62,n)
    marital   = np.random.choice(['Single','Married','Divorced'], n, p=[0.38,0.52,0.10])
    deps      = np.random.choice([0,1,2,3], n, p=[0.30,0.30,0.28,0.12])
    education = np.random.choice(['High School','Bachelor','Master','PhD'], n, p=[0.20,0.45,0.28,0.07])
    city_tier = np.random.choice(['Tier 1','Tier 2','Tier 3'], n, p=[0.38,0.38,0.24])
    emp_type  = np.random.choice(['Salaried','Self-Employed'], n, p=[0.65,0.35])

    emp_cat_list=[]
    for e in emp_type:
        if e=='Salaried': emp_cat_list.append(np.random.choice(['MNC','Government','Private SME','Startup'],p=[0.30,0.22,0.30,0.18]))
        else: emp_cat_list.append('Self-Employed')
    emp_cat = np.array(emp_cat_list)

    work_exp = np.clip(np.random.normal(6,4,n).astype(int),0,35)
    edu_boost= {'High School':0.7,'Bachelor':1.0,'Master':1.35,'PhD':1.60}
    base_inc = np.array([np.random.normal(55000,20000)*edu_boost[e] for e in education])
    base_inc = np.clip(base_inc,12000,400000).astype(int)

    exp_ratio= np.random.uniform(0.30,0.75,n)
    monthly_exp = (base_inc*exp_ratio).astype(int)
    existing_emi= np.array([int(np.random.uniform(2000,15000)) if np.random.rand()<0.4 else 0 for _ in range(n)])
    existing_emi= np.clip(existing_emi,0,base_inc*0.40).astype(int)
    dti = np.round((monthly_exp+existing_emi)/base_inc,3)

    loan_amt = np.random.choice([50000,100000,150000,200000,300000,500000,750000,1000000],n)
    tenure   = np.random.choice([12,24,36,48,60],n)

    cibil = np.clip(np.random.normal(680,80,n)+(base_inc>60000)*30-(dti>0.55)*40+(work_exp>5)*20+(emp_type=='Salaried')*15+np.random.normal(0,25,n),300,900).astype(int)

    score = np.zeros(n)
    score += np.where(cibil>=750,35,np.where(cibil>=700,25,np.where(cibil>=650,10,np.where(cibil>=600,0,-25))))
    score += np.where(dti<0.35,20,np.where(dti<0.50,10,np.where(dti<0.65,0,-20)))
    score += np.where(base_inc>=100000,20,np.where(base_inc>=60000,12,np.where(base_inc>=35000,5,-10)))
    lti    = loan_amt/base_inc
    score += np.where(lti<3,10,np.where(lti<6,5,np.where(lti<10,0,-15)))
    score += np.where(emp_cat=='Government',10,np.where(emp_cat=='MNC',7,np.where(emp_cat=='Private SME',3,np.where(emp_cat=='Startup',0,-3))))
    score += np.where(work_exp>=5,8,np.where(work_exp>=2,3,-5))
    score += np.random.normal(0,5,n)

    hard_rej = (cibil<550)|(dti>0.80)|(base_inc<15000)|(age<21)
    approved = ((score>=20)&~hard_rej).astype(int)

    return pd.DataFrame({'Age':age,'Gender':gender,'Marital_Status':marital,'Dependents':deps,
        'Education':education,'Employment_Type':emp_type,'Employer_Category':emp_cat,
        'Work_Experience_Yrs':work_exp,'City_Tier':city_tier,'Net_Monthly_Income':base_inc,
        'Monthly_Expenses':monthly_exp,'Existing_EMI':existing_emi,
        'Debt_To_Income_Ratio':dti,'CIBIL_Score':cibil,
        'Loan_Amount_Requested':loan_amt,'Loan_Tenure_Months':tenure,'Loan_Approved':approved})


def train(df):
    enc_map = {}
    for col in ['Gender','Marital_Status','Education','Employment_Type','Employer_Category','City_Tier']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        enc_map[col] = {cls:int(i) for i,cls in enumerate(le.classes_)}

    feat_cols = [c for c in df.columns if c!='Loan_Approved']
    X, y = df[feat_cols], df['Loan_Approved']
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    model = GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.08,random_state=42)
    model.fit(X_tr,y_tr)
    acc = accuracy_score(y_te,model.predict(X_te))
    cv  = cross_val_score(model,X,y,cv=5).mean()
    print(f"\nTest Accuracy : {acc:.3f}")
    print(f"CV Accuracy   : {cv:.3f}")
    print(classification_report(y_te,model.predict(X_te)))

    joblib.dump(model,       'pl_model.pkl')
    joblib.dump(feat_cols,   'pl_features.pkl')
    joblib.dump(enc_map,     'pl_encoders.pkl')
    print("✅ Saved pl_model.pkl, pl_features.pkl, pl_encoders.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, help='Path to custom CSV')
    args = parser.parse_args()

    if args.data:
        print(f"Loading {args.data}…")
        df = pd.read_csv(args.data)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")
    else:
        print("Generating synthetic dataset (6000 rows)…")
        df = generate_synthetic()
        df.to_csv('personal_loan_dataset.csv', index=False)
        print(f"Saved personal_loan_dataset.csv  |  Approval rate: {df['Loan_Approved'].mean()*100:.1f}%")

    train(df)
