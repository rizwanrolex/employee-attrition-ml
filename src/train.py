"""Train and evaluate 3 classifiers with a protected held-out test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             roc_auc_score,roc_curve,confusion_matrix,classification_report)

NUMERIC = ['satisfaction_level','last_evaluation','number_project',
           'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years']
CATEGORICAL=['Department','salary']
TARGET='Attrition'
METRICS=['Accuracy','Precision','Recall','F1-Score','ROC-AUC']

def read_dataset(path: Path, sheet='Attrition_Data'):
    df=pd.read_excel(path,sheet_name=sheet)
    df.columns=df.columns.str.strip()
    required=NUMERIC+CATEGORICAL+[TARGET]
    miss=set(required)-set(df.columns)
    if miss:
        raise ValueError(f'Missing required columns: {sorted(miss)}')
    df=df[required].copy()
    y=df[TARGET].astype('string').str.strip().str.lower().map({'yes':1,'no':0})
    if y.isna().any():
        raise ValueError('Unexpected Attrition labels; expected Yes or No.')
    for col in NUMERIC:df[col]=pd.to_numeric(df[col],errors='coerce')
    return df[NUMERIC+CATEGORICAL],y.astype(int)

def preprocessing():
    return ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),
                             ('scaler',StandardScaler())]),NUMERIC),
        ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                                 ('onehot',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL),
    ])

def classifiers(random_state=42):
    return {
        'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=random_state),
        'Random Forest':RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',random_state=random_state,n_jobs=-1),
        'SVM':SVC(kernel='rbf',C=1.0,class_weight='balanced',probability=True,random_state=random_state),
    }

def evaluate(y,pred,prob):
    return {
        'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,zero_division=0),
        'Recall':recall_score(y,pred,zero_division=0),'F1-Score':f1_score(y,pred,zero_division=0),
        'ROC-AUC':roc_auc_score(y,prob),
    }

def run(data_path: Path, out: Path, random_state: int = 42):
    X,y=read_dataset(data_path)
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=random_state)
    X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=.2,stratify=y_dev,random_state=random_state)
    base=preprocessing();models=classifiers(random_state)
    comparison=[]
    for name,algorithm in models.items():
        print('Training:',name,flush=True)
        pipe=Pipeline([('preprocessor',clone(base)),('classifier',clone(algorithm))])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_val);prob=pipe.predict_proba(X_val)[:,1]
        comparison.append({'Model':name,**evaluate(y_val,pred,prob)})
    cv=pd.DataFrame(comparison).sort_values(['F1-Score','ROC-AUC'],ascending=False).reset_index(drop=True)
    name=cv.loc[0,'Model']
    best=Pipeline([('preprocessor',clone(base)),('classifier',clone(models[name]))])
    best.fit(X_dev,y_dev)
    pred=best.predict(X_test);prob=best.predict_proba(X_test)[:,1]
    metrics={'Best Model':name,**evaluate(y_test,pred,prob)}
    cm=confusion_matrix(y_test,pred,labels=[0,1])
    out.mkdir(parents=True,exist_ok=True)
    cv.to_csv(out/'validation_comparison.csv',index=False)
    pd.DataFrame([metrics]).to_csv(out/'final_test_metrics.csv',index=False)
    pd.DataFrame(cm,index=['actual_stayed','actual_left'],columns=['pred_stayed','pred_left']).to_csv(out/'final_confusion_matrix.csv')
    (out/'classification_report.txt').write_text(classification_report(y_test,pred,target_names=['Stayed (No)','Left (Yes)'],zero_division=0))
    with (out/'metrics.json').open('w') as f:json.dump({'n_train':len(X_train),'n_validation':len(X_val),'n_test':len(X_test),'validation':cv.to_dict('records'),'final':metrics,'confusion_matrix':cm.tolist()},f,indent=2)
    fpr,tpr,_=roc_curve(y_test,prob)
    fig,ax=plt.subplots(figsize=(7.4,5.0))
    ax.plot(fpr,tpr,label=f'{name} (AUC={metrics["ROC-AUC"]:.4f})',linewidth=2.5)
    ax.plot([0,1],[0,1],'k--',label='Random baseline')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Final holdout ROC curve',xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(out/'roc_curve.png',dpi=170);plt.close(fig)
    joblib.dump(best,out/'best_model.joblib')
    print('Selected:',name,'Final metrics:',metrics,'Confusion:',cm.tolist())
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Reproduce the 10,000-row attrition experiment.')
    parser.add_argument('--input',type=Path,required=True,help='Path to 10,000-row .xlsx dataset')
    parser.add_argument('--output-dir',type=Path,default=Path('results/reproduced'))
    parser.add_argument('--random-state',type=int,default=42)
    args=parser.parse_args()
    run(args.input,args.output_dir,args.random_state)
