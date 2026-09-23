# -*- coding: utf-8 -*-
import pickle, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from features_final_multirate import FEATURE_COLUMNS

SEED=42
CSV='amc_dataset_exactpipeline_16features_2Msps_1024.csv'
DIG3='amc_model_digital3_exactpipeline_16features.pkl'
OUT='amc_hierarchical5_exactpipeline_16features.pkl'
RES='amc_hierarchical5_exactpipeline_training_results.txt'

df=pd.read_csv(CSV)
X=df[FEATURE_COLUMNS].copy(); y5=df['label'].astype(str)
ybin=np.where(y5.isin(['AM','FM']),'ANALOG','DIGITAL')

Xtr,Xte,ytr,yte=train_test_split(X,ybin,test_size=.2,random_state=SEED,stratify=ybin)
binary=RandomForestClassifier(n_estimators=400,max_depth=18,min_samples_leaf=2,min_samples_split=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)
binary.fit(Xtr,ytr)
p=binary.predict(Xte)
acc_bin=accuracy_score(yte,p)
rep_bin=classification_report(yte,p,digits=4)
cm_bin=confusion_matrix(yte,p,labels=['ANALOG','DIGITAL'])

m=y5.isin(['AM','FM'])
Xa,ya=X[m],y5[m]
Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=.2,random_state=SEED,stratify=ya)
analog=RandomForestClassifier(n_estimators=300,max_depth=18,min_samples_leaf=2,min_samples_split=2,random_state=SEED,n_jobs=-1)
analog.fit(Xtr,ytr)
p=analog.predict(Xte)
acc_a=accuracy_score(yte,p)
rep_a=classification_report(yte,p,digits=4)
cm_a=confusion_matrix(yte,p,labels=['AM','FM'])

with open(DIG3,'rb') as f: db=pickle.load(f)
digital=db['model']
if list(db['feature_columns'])!=FEATURE_COLUMNS: raise ValueError('Feature mismatch')

idx=np.arange(len(df)); _,tid=train_test_split(idx,test_size=.2,random_state=123,stratify=y5)
Xt=X.iloc[tid].reset_index(drop=True); yt=y5.iloc[tid].to_numpy()
r=binary.predict(Xt); fp=np.empty(len(Xt),dtype=object)
ia=np.where(r=='ANALOG')[0]; idg=np.where(r=='DIGITAL')[0]
if len(ia): fp[ia]=analog.predict(Xt.iloc[ia])
if len(idg): fp[idg]=digital.predict(Xt.iloc[idg])
classes=['AM','FM','BPSK','QPSK','16-QAM']
acc_end=accuracy_score(yt,fp)
rep_end=classification_report(yt,fp,labels=classes,digits=4,zero_division=0)
cm_end=confusion_matrix(yt,fp,labels=classes)

print('='*100); print('HIJERARHIJSKI 5-KLASNI AMC - TRENING'); print('='*100)
print(f'\nSTAGE 1 ANALOG/DIGITAL holdout: {acc_bin*100:.2f}%'); print(rep_bin); print(cm_bin)
print(f'\nSTAGE 2A AM/FM holdout: {acc_a*100:.2f}%'); print(rep_a); print(cm_a)
print(f'\nEND-TO-END sintetički 5-class: {acc_end*100:.2f}%'); print(rep_end); print(cm_end)

bundle={'binary_model':binary,'analog_model':analog,'digital_model':digital,'feature_columns':FEATURE_COLUMNS,'final_classes':classes,'fs':2_000_000,'frame_samples':1024,'architecture':'ANALOG/DIGITAL -> AM/FM or BPSK/QPSK/16-QAM','training_type':'synthetic_only_exact_pipeline_hierarchical','seed':SEED}
with open(OUT,'wb') as f: pickle.dump(bundle,f)
with open(RES,'w',encoding='utf-8') as f:
    f.write(f'Stage1={acc_bin*100:.4f}%\nStage2A={acc_a*100:.4f}%\nEndToEnd={acc_end*100:.4f}%\n\n')
    f.write(rep_end+'\n'+str(cm_end))
print('\nSacuvano:',OUT,RES,sep='\n- ')
