# -*- coding: utf-8 -*-
import os,pickle,h5py,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from features_final_multirate import extract_features,FEATURE_COLUMNS

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL = (
    PROJECT_ROOT
    / "models"
    / "amc_hierarchical5_exactpipeline_16features.pkl"
)

H5 = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "subset_test.h5"
)
OUT='REAL_HIERARCHICAL5_EXACTPIPELINE_RESULTS'; os.makedirs(OUT,exist_ok=True)
MAP={0:'BPSK',1:'QPSK',2:'16-QAM'}; TARGET=['BPSK','QPSK','16-QAM']; FINAL=['AM','FM','BPSK','QPSK','16-QAM']; CH={0:'Clean / LOS',1:'Multipath'}

with open(MODEL,'rb') as f: b=pickle.load(f)
binary,analog,digital=b['binary_model'],b['analog_model'],b['digital_model']; cols=list(b['feature_columns'])
with h5py.File(H5,'r') as f:
    ym=f['y_mod'][:]; yc=f['y_chan'][:]; ys=f['y_snr'][:]
idx=np.where(np.isin(ym,[0,1,2]))[0]
print('='*100); print('REALNI DIGITALNI SIGNALI - HIJERARHIJSKI 5-KLASNI SISTEM'); print('='*100); print('Ukupno:',len(idx))
F=[]; L=[]; S=[]; C=[]
with h5py.File(H5,'r') as f:
    X=f['X']; total=len(idx)
    for k,i in enumerate(idx):
        fr=X[i].astype(np.float32); iq=(fr[:,0]+1j*fr[:,1]).astype(np.complex64); p=np.mean(np.abs(iq)**2)
        if p>1e-12: iq/=np.sqrt(p)
        ft=extract_features(iq,fs=2_000_000); v=np.array([ft[c] for c in cols],float)
        if np.all(np.isfinite(v)):
            F.append(v); L.append(MAP[int(ym[i])]); S.append(int(ys[i])); C.append(int(yc[i]))
        if (k+1)%1000==0 or k==total-1: print(f'Obradjeno {k+1}/{total}')
Xr=pd.DataFrame(np.asarray(F),columns=cols); L=np.asarray(L); S=np.asarray(S); C=np.asarray(C)
r=binary.predict(Xr); md=r=='DIGITAL'; ma=~md
fp=np.empty(len(Xr),dtype=object); idg=np.where(md)[0]; ia=np.where(ma)[0]
if len(idg): fp[idg]=digital.predict(Xr.iloc[idg])
if len(ia): fp[ia]=analog.predict(Xr.iloc[ia])
route=np.mean(md); overall=accuracy_score(L,fp)
print('\n'+'='*100); print('STAGE 1 - ANALOG / DIGITAL'); print('='*100)
print(f'Realni digitalni frejmovi prepoznati kao DIGITAL: {route*100:.2f}%')
print(f'Pogresno usmereni u ANALOG granu: {(1-route)*100:.2f}%')
print('\n'+'='*100); print('KONACNI HIJERARHIJSKI REZULTAT'); print('='*100)
print(f'UKUPNA REAL-WORLD TACNOST: {overall*100:.2f}%')
rows=[]
print('\nTACNOST PO KLASI')
for cls in TARGET:
    m=L==cls; a=accuracy_score(L[m],fp[m]); rr=np.mean(md[m]); n=int(m.sum()); rows.append((cls,a,rr,n)); print(f'{cls}: final={a*100:.2f}% | Stage1->DIGITAL={rr*100:.2f}% | N={n}')
print('\nTACNOST PO SNR-u')
for snr in sorted(np.unique(S)):
    m=S==snr; a=accuracy_score(L[m],fp[m]); rr=np.mean(md[m]); print(f'SNR={snr:2d} dB: final={a*100:.2f}% | Stage1->DIGITAL={rr*100:.2f}% | N={m.sum()}')
print('\nCLEAN VS MULTIPATH')
for ch in sorted(np.unique(C)):
    m=C==ch; a=accuracy_score(L[m],fp[m]); rr=np.mean(md[m]); print(f'{CH[ch]}: final={a*100:.2f}% | Stage1->DIGITAL={rr*100:.2f}% | N={m.sum()}')
cm=confusion_matrix(L,fp,labels=FINAL); cm_df=pd.DataFrame(cm,index=[f'True {x}' for x in FINAL],columns=[f'Pred {x}' for x in FINAL])
print('\nFINALNA CONFUSION MATRIX'); print(cm_df)
rep=classification_report(L,fp,labels=TARGET,digits=4,zero_division=0); print('\nCLASSIFICATION REPORT'); print(rep)
cm_df.to_csv(os.path.join(OUT,'confusion_matrix.csv'))
with open(os.path.join(OUT,'summary.txt'),'w',encoding='utf-8') as f:
    f.write(f'Stage1 DIGITAL routing: {route*100:.4f}%\nFinal real-world accuracy: {overall*100:.4f}%\n\n'+rep+'\n'+cm_df.to_string())
print('\nSacuvano u:',OUT)
