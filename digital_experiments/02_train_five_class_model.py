# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from features_final_multirate import extract_features, FEATURE_COLUMNS

SEED=42
rng=np.random.default_rng(SEED)

FS_FINAL=2_000_000
FINAL_SAMPLES=1024
SYMBOL_RATE=100_000
SPS_SOURCE=4
FS_SOURCE=SYMBOL_RATE*SPS_SOURCE
UPSAMPLE_FACTOR=5
RRC_ALPHA=0.35
RRC_SPAN_SYMBOLS=10
SNR_VALUES=[20,22,24,26,28,30]
CLASSES=["AM","FM","BPSK","QPSK","16-QAM"]
EXAMPLES_PER_CLASS_PER_SNR=250

DATASET_NAME="amc_dataset_exactpipeline_16features_2Msps_1024.csv"
MODEL_NAME="amc_model_exactpipeline_16features_2Msps_1024.pkl"
RESULTS_NAME="amc_model_exactpipeline_16features_2Msps_1024_results.txt"
IMPORTANCE_NAME="amc_model_exactpipeline_16features_2Msps_1024_importance.csv"

def rrc_taps(beta,sps,span_symbols=10):
    n_taps=span_symbols*sps+1
    t=(np.arange(n_taps)-(n_taps-1)/2)/sps
    h=np.zeros(n_taps,dtype=float)
    for i,ti in enumerate(t):
        if abs(ti)<1e-12:
            h[i]=1+beta*(4/np.pi-1)
        elif beta>0 and abs(abs(ti)-1/(4*beta))<1e-10:
            h[i]=beta/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num=np.sin(np.pi*ti*(1-beta))+4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den=np.pi*ti*(1-(4*beta*ti)**2)
            h[i]=num/den
    h/=np.sqrt(np.sum(h**2)+1e-12)
    return h

RRC=rrc_taps(RRC_ALPHA,SPS_SOURCE,RRC_SPAN_SYMBOLS)

def pulse_shape_400k(symbols):
    x=signal.upfirdn(RRC,symbols,up=SPS_SOURCE)
    d=(len(RRC)-1)//2
    x=x[d:len(x)-d]
    return x.astype(np.complex64)

def to_2msps_crop(iq):
    y=signal.resample_poly(iq,up=UPSAMPLE_FACTOR,down=1).astype(np.complex64)
    if len(y)<FINAL_SAMPLES:
        raise ValueError("Premalo uzoraka posle resampling-a.")
    s=(len(y)-FINAL_SAMPLES)//2
    return y[s:s+FINAL_SAMPLES].astype(np.complex64)

def digital_pipeline(symbols):
    return to_2msps_crop(pulse_shape_400k(symbols))

def nsym():
    return 80

def generate_bpsk():
    b=rng.integers(0,2,nsym())
    return digital_pipeline((2*b-1).astype(np.complex64))

def generate_qpsk():
    idx=rng.integers(0,4,nsym())
    sym=np.exp(1j*(np.pi/4+idx*np.pi/2))
    return digital_pipeline(sym)

def generate_16qam():
    lev=np.array([-3,-1,1,3],dtype=float)
    i=rng.choice(lev,nsym())
    q=rng.choice(lev,nsym())
    return digital_pipeline((i+1j*q)/np.sqrt(10))

def generate_am():
    t=np.arange(FINAL_SAMPLES)/FS_FINAL
    f1=rng.uniform(1e3,12e3); f2=rng.uniform(12e3,30e3)
    p1=rng.uniform(0,2*np.pi); p2=rng.uniform(0,2*np.pi)
    m=0.65*np.sin(2*np.pi*f1*t+p1)+0.35*np.sin(2*np.pi*f2*t+p2)
    m/=np.max(np.abs(m))+1e-12
    mi=rng.uniform(0.25,0.9)
    return (1+mi*m).astype(np.complex64)

def generate_fm():
    t=np.arange(FINAL_SAMPLES)/FS_FINAL
    f1=rng.uniform(2e3,15e3); f2=rng.uniform(10e3,40e3)
    p1=rng.uniform(0,2*np.pi); p2=rng.uniform(0,2*np.pi)
    m=0.6*np.sin(2*np.pi*f1*t+p1)+0.4*np.sin(2*np.pi*f2*t+p2)
    m/=np.max(np.abs(m))+1e-12
    dev=rng.uniform(25e3,75e3)
    ph=2*np.pi*dev*np.cumsum(m)/FS_FINAL
    return np.exp(1j*ph).astype(np.complex64)

def generate_signal(label):
    return {"AM":generate_am,"FM":generate_fm,"BPSK":generate_bpsk,"QPSK":generate_qpsk,"16-QAM":generate_16qam}[label]()

def normalize(iq):
    p=np.mean(np.abs(iq)**2)
    return (iq/np.sqrt(p+1e-12)).astype(np.complex64)

def mild_rf(iq):
    n=len(iq); t=np.arange(n)/FS_FINAL
    foff=rng.uniform(-5e3,5e3)
    ph0=rng.uniform(0,2*np.pi)
    pn=np.cumsum(rng.standard_normal(n)*rng.uniform(0.0005,0.004))
    return (iq*np.exp(1j*(2*np.pi*foff*t+ph0+pn))).astype(np.complex64)

def awgn(iq,snr_db):
    sp=np.mean(np.abs(iq)**2)
    npow=sp/(10**(snr_db/10))
    n=rng.standard_normal(len(iq))+1j*rng.standard_normal(len(iq))
    n*=np.sqrt(npow/2)
    return (iq+n).astype(np.complex64)

print("="*100)
print("KONTROLISANI EXACT-PIPELINE SYNTHETIC MODEL")
print("="*100)
print("Digital: 100 kSym/s -> 4 sps -> 400 kS/s -> RRC 0.35 -> x5 -> 2 MS/s")
print("Final frame:",FINAL_SAMPLES)
print("SNR:",SNR_VALUES)

rows=[]
for label in CLASSES:
    print("\nKLASA:",label)
    for snr in SNR_VALUES:
        print("  SNR",snr,"dB")
        for _ in range(EXAMPLES_PER_CLASS_PER_SNR):
            iq=generate_signal(label)
            iq=mild_rf(iq)
            iq=awgn(iq,snr)
            iq=normalize(iq)
            feats=extract_features(iq,fs=FS_FINAL)
            row={f:feats[f] for f in FEATURE_COLUMNS}
            row["label"]=label
            row["snr_db"]=snr
            rows.append(row)

df=pd.DataFrame(rows)
if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(float)).all():
    raise ValueError("NaN/inf u datasetu")

df.to_csv(DATASET_NAME,index=False)
print("\nShape:",df.shape)
print(df["label"].value_counts().sort_index())

X=df[FEATURE_COLUMNS]
y=df["label"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)

rf=RandomForestClassifier(n_estimators=300,max_depth=18,min_samples_leaf=2,min_samples_split=2,random_state=SEED,n_jobs=-1)
print("\nTreniram Random Forest...")
rf.fit(Xtr,ytr)

pred=rf.predict(Xte)
acc=accuracy_score(yte,pred)
report=classification_report(yte,pred,digits=4)
cm=confusion_matrix(yte,pred,labels=CLASSES)

print("\n"+"="*100)
print("EXACT-PIPELINE SINTETICKI HOLDOUT")
print("="*100)
print(f"Accuracy: {acc*100:.2f}%")
print(report)
print("Confusion matrix, klase:",CLASSES)
print(cm)

meta=df.loc[Xte.index,["label","snr_db"]].copy()
meta["prediction"]=pred
snr_rows=[]
print("\nTACNOST PO SNR-u")
for snr in SNR_VALUES:
    sub=meta[meta["snr_db"]==snr]
    a=np.mean(sub["label"].to_numpy()==sub["prediction"].to_numpy())
    snr_rows.append((snr,a,len(sub)))
    print(f"SNR {snr:>2} dB: {a*100:6.2f}% (n={len(sub)})")

imp=pd.DataFrame({"feature":FEATURE_COLUMNS,"importance":rf.feature_importances_}).sort_values("importance",ascending=False)
imp.to_csv(IMPORTANCE_NAME,index=False)
print("\nFEATURE IMPORTANCE")
print(imp.to_string(index=False))

with open(MODEL_NAME,"wb") as f:
    pickle.dump({
        "model":rf,
        "feature_columns":FEATURE_COLUMNS,
        "fs":FS_FINAL,
        "final_samples":FINAL_SAMPLES,
        "classes":CLASSES,
        "symbol_rate":SYMBOL_RATE,
        "source_sps":SPS_SOURCE,
        "source_fs":FS_SOURCE,
        "upsample_factor":UPSAMPLE_FACTOR,
        "rrc_alpha":RRC_ALPHA,
        "snr_values":SNR_VALUES,
        "seed":SEED,
        "training_type":"synthetic_only_exact_4sps_then_x5",
    },f)

with open(RESULTS_NAME,"w",encoding="utf-8") as f:
    f.write(f"Accuracy: {acc*100:.4f}%\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n"+str(cm))
    f.write("\n\nAccuracy by SNR:\n")
    for snr,a,n in snr_rows:
        f.write(f"{snr} dB: {a*100:.4f}% (n={n})\n")
    f.write("\nFeature importance:\n"+imp.to_string(index=False))

print("\nSacuvano:")
print("-",DATASET_NAME)
print("-",MODEL_NAME)
print("-",RESULTS_NAME)
print("-",IMPORTANCE_NAME)
print("\nSLEDECE: ako je holdout dobar, testirati ovaj model na realnom subset_test.h5.")
