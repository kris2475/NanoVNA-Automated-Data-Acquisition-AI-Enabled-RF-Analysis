
# NanoVNA Automated Data Acquisition & AI-Enabled RF Analysis

This project demonstrates an **end-to-end workflow** for automated RF measurement using the **NanoVNA Filter Attenuator Test Board** integrated with **Machine Learning (ML)** and **Deep Learning (DL)** models. The workflow leverages the **latest automated acquisition script v2.9** (900 MHz constraint and logging fix) for robust data collection, preprocessing, and ML/DL integration.

---

## 🧭 Executive Summary

This project transforms a low-cost NanoVNA into a **AI-assisted RF analyzer** capable of:

- Automated **S-parameter sweeps** using Python
- Classification of RF circuits (LPF, HPF, BPF, attenuators) using **1D CNNs**
- Regression for predicting component values like attenuation or impedance
- **Calibration integrity monitoring** using Autoencoders
- Logging fixes and 900 MHz sweep cap to improve robustness in v2.9

It combines **hardware control**, **data acquisition**, **feature engineering**, and **AI model training** in one open, reproducible framework.

---

## ⚙️ Hardware Requirements

| Component | Description |
|-----------|-------------|
| NanoVNA (v2/v3) | USB-enabled, supports 2-port sweeps |
| NanoVNA Filter & Attenuator Test Board | LPF, HPF, BPF, Attenuators |
| SMA Male–Male Cables | Equal length recommended |
| Calibration Kit (SOLT) | Short, Open, Load, Thru |
| PC / Raspberry Pi | Python 3.8+ |

---

## 🧠 Software Requirements

```bash
pip install numpy pandas pynanovna matplotlib scikit-learn tensorflow jupyter
```

---

## 📝 Key Features of Acquisition Script v2.9

1. **Sweep Frequency Management**
   - Automatic switching between LOW, WIDE, and HIGH-optimized sweeps
   - LOW: 1–30 MHz for critical low-frequency circuits
   - WIDE: 1–900 MHz for default components
   - C11: 10–600 MHz optimized LPF
   - C12: 100–900 MHz optimized HPF (capped for NanoVNA)

2. **Custom Logging Filter**
   - Suppresses repetitive calibration warnings from pynanovna
   - Prevents recursive logging errors

3. **Sweep Verification**
   - LOW-frequency sweeps are automatically verified
   - Manual intervention prompted if start frequency not applied correctly

4. **Data Averaging & Export**
   - Configurable REPEATS (default = 3) per circuit
   - Computes magnitude (dB) and phase (degrees) for S11 and S21
   - CSV export per circuit plus summary table

5. **Port Handling**
   - Supports both 1-port and 2-port DUTs
   - Adjusts S21 handling for missing data in single-port circuits

---

## 📡 Workflow Overview

### 1. Connect NanoVNA

```python
vna = VNA()
vna.set_sweep(start=WIDE_START_FREQ, stop=WIDE_STOP_FREQ, points=WIDE_POINTS)
```

- Verify USB connection
- Initialize WIDE sweep
- Set `CURRENT_SWEEP` for later verification

### 2. Attach DUT

- 1-port or 2-port coax connection depending on the circuit
- Script prompts user at each step

### 3. Sweep Verification (LOW circuits)

```python
def verify_low_sweep_success(vna_instance, target_start_freq, circuit_number, description):
    # Test sweep, check first frequency, prompt manual adjustment if needed
```

- Ensures accurate LOW frequency sweep
- Returns initial sweep data for accumulation

### 4. Sweep Acquisition & Averaging

- Loop over all circuits
- Apply conditional sweep parameters (LOW, WIDE, C11, C12)
- Repeat `REPEATS` times
- Average real/imag components for S11 and S21

```python
s11_real_avg = s11_real_accum / sweeps_performed
s21_real_avg = s21_real_accum / sweeps_performed
```

- Calculate magnitude and phase
- Save per-circuit CSV

### 5. Summary Table

```python
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(SAVE_FOLDER, 'summary_table_optimized.csv'), index=False)
```

- Includes metadata: Circuit, Ports, Filename, Sweep Range, Points, Repeats, Successful Sweeps

### 6. Disconnect

```python
vna.disconnect()
```
- Clean disconnect at end of script
- Handles AttributeError safely (pynanovna quirk)

---

## 🧩 Deep Learning Integration

### 1. Filter Classification (1D CNN)

```python
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv1D(32, 7, activation='relu', input_shape=input_shape),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 2. Component Value Regression

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = Sequential([
    Dense(128, activation='relu', input_dim=input_dim),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### 3. Calibration Integrity Monitoring (Autoencoder)

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```

- Reconstruction error used for anomaly detection
- Thresholding detects deviations in calibration standards

```python
errors = np.mean(np.square(X_test - reconstructions), axis=1)
anomalies = errors > threshold
```

---

## ⚡ Usage Instructions

1. Connect NanoVNA via USB
2. Place the DUT as prompted
3. Run the script: `python nanovna_acquisition_v2_9.py`
4. CSVs stored in `nanovna_data/`
5. Summary CSV shows all sweeps and metadata

---

## 🧰 Tools & Libraries

- **Python 3.8+**
- **pynanovna**
- **numpy, pandas**
- **scikit-learn, tensorflow**
- **matplotlib**

---

**Author:** K. Seunarine  
**License:** MIT  
**Last Updated:** 21 October 2025
