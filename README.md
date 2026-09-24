SDR klasifikacija radio-signala

Ovaj repozitorijum sadrži izvorni kod korišćen u okviru master rada:
„Primena softverski definisanog radija za detekciju, analizu i klasifikaciju radio-signala primenom metoda digitalne obrade signala i mašinskog učenja“
Cilj projekta je detekcija, analiza i klasifikacija radio-signala korišćenjem SDR prijemnika, metoda digitalne obrade signala i algoritama mašinskog učenja.

U radu su analizirane sledeće modulacione klase:
- AM
- FM
- BPSK
- QPSK
- 16-QAM

Za prijem realnih AM i FM signala korišćen je RTL-SDR Blog V3 SDR prijemnik sa RTL2832U čipom i R820T2 RF tjunerom, dok je za proveru rada klasifikatora na realnim digitalnim signalima korišćen javno dostupan SDR skup podataka sa poznatim oznakama modulacionih klasa.

Struktura projekta
sdr-modulation-classification/
│
├── acquisition/
│   ├── am/
│   │   ├── 01_scan_vhf_airband.py
│   │   └── 02_record_am_125925_mhz.py
│   └── fm/
│       ├── 01_scan_vhf_fm_band.py
│       └── 02_record_fm_98_7047_mhz.py
│
├── processing/
│   ├── am/
│   │   └── 01_demodulate_am_125925_mhz.py
│   └── fm/
│       └── 01_analyze_and_demodulate_fm_98_7047_mhz.py
│
├── features/
│   ├── features_final.py
│   └── features_final_multirate.py
│
├── classification/
│   ├── base_model/
│   │   └── 01_train_five_class_classifier.py
│   └── real_am_fm/
│       ├── 01_evaluate_real_am.py
│       └── 02_evaluate_real_fm.py
│
├── experiments/
│   ├── spectral_analysis/
│   │   └── 01_compare_fft_welch_windows.py
│   ├── base_classification/
│   │   ├── 01_run_basic_synthetic_classification.py
│   │   └── 02_check_rf_overfitting.py
│   └── efficiency/
│       └── 01_analyze_segment_length_and_efficiency.py
│
├── digital_experiments/
│   ├── 01_verify_synthetic_generator.py
│   ├── 02_train_five_class_model.py
│   ├── 03_evaluate_five_class_on_real.py
│   ├── 04_train_three_class_digital_model.py
│   ├── 05_evaluate_three_class_on_real.py
│   ├── 06_train_hierarchical_model.py
│   ├── 07_evaluate_hierarchical_on_real.py
│   ├── 08_analyze_domain_gap.py
│   └── 09_real_assisted_training.py
│
├── realtime/
│   └── 01_realtime_fm_classification.py
│
├── data/
│   ├── am/
│   ├── fm/
│   ├── real/
│   └── synthetic/
│
├── models/
├── results/
├── requirements.txt
└── README.md

Opis glavnih delova projekta

Prijem AM signala
Skripte u direktorijumu acquisition/am/ koriste se za skeniranje VHF vazduhoplovnog opsega i snimanje AM signala.
Analizirani opseg je 118–137 MHz, a ciljana frekvencija korišćena u eksperimentima je približno 125.925 MHz.

Prijem FM signala
Skripte u direktorijumu acquisition/fm/ koriste se za skeniranje FM radio-difuznog opsega i snimanje odabranog FM signala.
Analizirani opseg je 87.5–108 MHz.
U završnim eksperimentima analiziran je signal na približno 98.7047 MHz.

Obrada i demodulacija
Direktorijum processing/ sadrži skripte za obradu snimljenih I/Q uzoraka i demodulaciju AM i FM signala.
Obrada obuhvata:
- filtriranje,
- promenu frekvencije uzorkovanja,
- AM demodulaciju,
- FM demodulaciju,
- spektralnu analizu,
- generisanje rezultata za dalju analizu.

Ekstrakcija obeležja
Direktorijum features/ sadrži implementaciju ekstrakcije 16 obeležja korišćenih za klasifikaciju signala.
Datoteka features_final.py koristi se za osnovni klasifikacioni sistem pri frekvenciji uzorkovanja od 500 kS/s.
Datoteka features_final_multirate.py koristi se u eksperimentima sa realnim digitalnim SDR skupom pri frekvenciji uzorkovanja od 2 MS/s.
U oba slučaja koristi se isti skup od 16 obeležja, dok se parametri koji zavise od frekvencije uzorkovanja prilagođavaju odgovarajućem signalu.

Osnovna klasifikacija
U osnovnom eksperimentu korišćeno je pet klasa:
- AM
- FM
- BPSK
- QPSK
- 16-QAM
Za obuku i poređenje korišćeni su algoritmi mašinskog učenja, uključujući:
- Random Forest,
- Support Vector Machine,
- Decision Tree,
- Logistic Regression,
- k-Nearest Neighbors.
Sintetički skup podataka korišćen je za obuku i početnu evaluaciju modela.

Realni AM i FM signali
Skripte u classification/real_am_fm/ koriste se za proveru klasifikacionog modela na realnim AM i FM signalima snimljenim RTL-SDR prijemnikom.
Model korišćen u ovom delu očekuje 16 izdvojenih obeležja.

Eksperimenti sa realnim digitalnim signalima
Direktorijum digital_experiments/ sadrži eksperimente kojima se proverava generalizacija modela sa sintetičkih na realne digitalne SDR signale.
Redosled pokretanja skripti je:
01_verify_synthetic_generator.py
02_train_five_class_model.py
03_evaluate_five_class_on_real.py
04_train_three_class_digital_model.py
05_evaluate_three_class_on_real.py
06_train_hierarchical_model.py
07_evaluate_hierarchical_on_real.py
08_analyze_domain_gap.py
09_real_assisted_training.py
Prvo se proverava ispravnost sintetičkog generatora, zatim se trenira petoklasni model i proverava na realnim podacima.
Nakon toga se analiziraju:
- troklasni digitalni model,
- hijerarhijski klasifikacioni pristup,
- razlika između sintetičkog i realnog domena,
- real-assisted pristup učenju.

Realni digitalni skup podataka
Za evaluaciju na realnim digitalnim signalima korišćen je javno dostupan skup podataka:
Nikolay Belousov and Mikhail Ronkin, "Real-World IQ Dataset for Automatic Radio Modulation Recognition under Multipath Channels", Mendeley Data, Version 2, 2026.
DOI: 10.17632/tjzsbph49x.2
Zvanična stranica skupa:
https://data.mendeley.com/datasets/tjzsbph49x/2
Skup sadrži realne kompleksne I/Q uzorke prikupljene na 2.4 GHz, sa poznatim oznakama modulacione klase, SNR nivoa i uslova propagacije.
U okviru ovog rada korišćene su sledeće digitalne modulacione klase:
- BPSK
- QPSK
- 16-QAM
Za eksperimente korišćen je fajl subset_test.h5.
Veliki .h5 fajl nije uključen u ovaj GitHub repozitorijum.
Nakon preuzimanja skupa, datoteku subset_test.h5 potrebno je smestiti u:
data/real/subset_test.h5

Sintetički podaci
Sintetički podaci korišćeni u eksperimentima smeštaju se u:
data/synthetic/
Određene skripte generišu odgovarajući sintetički skup podataka i čuvaju ga za naredne eksperimente.
AM i FM snimci
Snimljeni I/Q podaci nisu uključeni u repozitorijum zbog njihove veličine.
AM podaci se očekuju u:
data/am/
FM podaci se očekuju u:
data/fm/
Na primer:
data/am/AM_125_925/
data/fm/FM_98_7047/

Modeli
Obučeni modeli čuvaju se u:
models/
Modeli se mogu generisati odgovarajućim skriptama za obuku.
Velike .pkl datoteke nisu neophodno uključene u repozitorijum i mogu se ponovo generisati pokretanjem odgovarajućih skripti.

Rezultati
Rezultati eksperimenata čuvaju se u:
results/
Pojedinačne skripte automatski kreiraju potrebne poddirektorijume ukoliko oni ne postoje.

Real-time klasifikacija
Skripta realtime/01_realtime_fm_classification.py implementira obradu i klasifikaciju FM signala u realnom vremenu.
Sistem radi sa blokovima I/Q uzoraka i zasebnim postupcima prijema i obrade, kako bi se procenila mogućnost primene klasifikacionog modela u real-time uslovima.

Instalacija
Tokom izrade rada korišćen je Python 3.8.10.
Potrebne biblioteke mogu se instalirati naredbom:
pip install -r requirements.txt

Organizacija putanja
Skripte u repozitorijumu ne koriste apsolutne lokalne putanje kao što su C:\Users\....
Putanje do podataka, modela i rezultata formiraju se relativno u odnosu na korenski direktorijum projekta pomoću biblioteke pathlib.
Na ovaj način kod se može pokrenuti na drugom računaru bez izmene korisničkih lokalnih putanja, pod uslovom da su potrebni podaci smešteni u odgovarajuće direktorijume.

Napomena o velikim datotekama
Zbog veličine, određene datoteke nisu uključene u repozitorijum, uključujući:
- realni digitalni HDF5 skup podataka,
- snimljene AM i FM I/Q datoteke,
- određene obučene modele,
- generisane rezultate eksperimenata.
Potrebne datoteke treba preuzeti ili generisati prema uputstvima navedenim u prethodnim odeljcima.
