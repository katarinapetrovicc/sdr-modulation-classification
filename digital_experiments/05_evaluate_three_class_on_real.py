# -*- coding: utf-8 -*-
import os
import pickle
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from features_final_multirate import extract_features, FEATURE_COLUMNS

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "amc_model_digital3_exactpipeline_16features.pkl"
)

REAL_H5 = (
    PROJECT_ROOT
    / "data"
    / "real"
    / "subset_test.h5"
)
OUTPUT_DIR="REAL_DIGITAL3_EXACTPIPELINE_RESULTS"
os.makedirs(OUTPUT_DIR,exist_ok=True)

REAL_FS=2_000_000
REAL_CLASS_MAP={0:"BPSK",1:"QPSK",2:"16-QAM"}
TARGET_CLASSES=["BPSK","QPSK","16-QAM"]
CHANNEL_NAMES={0:"Clean / LOS",1:"Multipath"}

with open(MODEL_FILE,"rb") as f:
    bundle=pickle.load(f)

model=bundle["model"]
cols=list(bundle["feature_columns"])
if cols!=FEATURE_COLUMNS:
    raise ValueError("Feature kolone se ne poklapaju.")

with h5py.File(REAL_H5,"r") as f:
    y_mod=f["y_mod"][:]
    y_chan=f["y_chan"][:]
    y_snr=f["y_snr"][:]

indices=np.where(np.isin(y_mod,[0,1,2]))[0]

print("="*100)
print("REALNI DIGITALNI SIGNALI - TROKLASNI MODEL")
print("="*100)
print("Klase modela:",list(model.classes_))
print("Ukupno realnih frejmova:",len(indices))

features=[]; labels=[]; snrs=[]; channels=[]

with h5py.File(REAL_H5,"r") as f:
    X=f["X"]
    total=len(indices)
    for counter,idx in enumerate(indices):
        frame=X[idx].astype(np.float32)
        iq=(frame[:,0]+1j*frame[:,1]).astype(np.complex64)
        p=np.mean(np.abs(iq)**2)
        if p>1e-12:
            iq=iq/np.sqrt(p)
        feats=extract_features(iq,fs=REAL_FS)
        vec=np.array([feats[c] for c in cols],dtype=float)
        if not np.all(np.isfinite(vec)):
            continue
        features.append(vec)
        labels.append(REAL_CLASS_MAP[int(y_mod[idx])])
        snrs.append(int(y_snr[idx]))
        channels.append(int(y_chan[idx]))
        if (counter+1)%1000==0 or counter==total-1:
            print(f"Obradjeno {counter+1}/{total}")

Xreal=pd.DataFrame(np.asarray(features),columns=cols)
labels=np.asarray(labels)
snrs=np.asarray(snrs)
channels=np.asarray(channels)
pred=model.predict(Xreal)

overall=accuracy_score(labels,pred)

print("\n"+"="*100)
print("UKUPNI REZULTAT")
print("="*100)
print(f"UKUPNA REAL-WORLD TACNOST: {overall*100:.2f}%")

print("\nTACNOST PO KLASI")
class_rows=[]
for cls in TARGET_CLASSES:
    m=labels==cls
    a=accuracy_score(labels[m],pred[m])
    n=int(np.sum(m))
    class_rows.append({"class":cls,"accuracy_percent":a*100,"n":n})
    print(f"{cls}: {a*100:.2f}% (N={n})")

print("\nTACNOST PO SNR-u")
snr_rows=[]
for snr in sorted(np.unique(snrs)):
    m=snrs==snr
    a=accuracy_score(labels[m],pred[m])
    n=int(np.sum(m))
    snr_rows.append({"snr_db":int(snr),"accuracy_percent":a*100,"n":n})
    print(f"SNR={snr:2d} dB: {a*100:.2f}% (N={n})")

print("\nCLEAN VS MULTIPATH")
channel_rows=[]
for ch in sorted(np.unique(channels)):
    m=channels==ch
    a=accuracy_score(labels[m],pred[m])
    n=int(np.sum(m))
    channel_rows.append({"channel":int(ch),"channel_name":CHANNEL_NAMES[int(ch)],"accuracy_percent":a*100,"n":n})
    print(f"{CHANNEL_NAMES[int(ch)]}: {a*100:.2f}% (N={n})")

print("\nCONFUSION MATRIX")
cm=confusion_matrix(labels,pred,labels=TARGET_CLASSES)
cm_df=pd.DataFrame(cm,index=[f"True {x}" for x in TARGET_CLASSES],columns=[f"Pred {x}" for x in TARGET_CLASSES])
print(cm_df)

print("\nCLASSIFICATION REPORT")
report=classification_report(labels,pred,labels=TARGET_CLASSES,digits=4,zero_division=0)
print(report)

cm_df.to_csv(os.path.join(OUTPUT_DIR,"confusion_matrix.csv"))
pd.DataFrame(class_rows).to_csv(os.path.join(OUTPUT_DIR,"by_class.csv"),index=False)
pd.DataFrame(snr_rows).to_csv(os.path.join(OUTPUT_DIR,"by_snr.csv"),index=False)
pd.DataFrame(channel_rows).to_csv(os.path.join(OUTPUT_DIR,"by_channel.csv"),index=False)

with open(os.path.join(OUTPUT_DIR,"summary.txt"),"w",encoding="utf-8") as f:
    f.write(f"UKUPNA REAL-WORLD TACNOST: {overall*100:.4f}%\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(cm_df.to_string())

print("\nSacuvano u:",OUTPUT_DIR)
