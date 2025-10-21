# Leveraging Deep Learning for Advanced RF Component Analysis with the NanoVNA Test Board

The integration of low-cost vector network analysis (VNA) with modern data science techniques offers a powerful new paradigm for radio frequency (RF) engineering.  

This project outlines a complete workflow — from **data acquisition** to **deep learning model deployment** — focused on utilizing **Machine Learning (ML)** and **Deep Learning (DL)** to characterize, classify, and monitor components on the **NanoVNA Filter Attenuator VNA RF Test Board Tester Demo Kit**.

By combining affordable measurement hardware with advanced AI techniques, this framework maximizes the utility of the test board as a versatile data source for complex **S-parameter** measurements.

---

## I. Project Foundation: Data Acquisition and Preprocessing

Accurate and automated data collection is the cornerstone of any successful ML project. The Python ecosystem provides robust tools for automating the NanoVNA.

### A. Data Acquisition Automation

**Hardware Setup**
- Connect the NanoVNA to the host PC via USB.  
- Attach the VNA ports:
  - CH0 (Port 1) and CH1 (Port 2) to the RF Test Board using SMA cables.  
- Perform a **full 2-port SOLT calibration** (Short, Open, Load, Thru) at the reference plane to remove cable and connector parasitics.

**Tooling**
- Use a Python library such as `pynanovna` or equivalent scripts to control the VNA.

**Sweep Settings**
- Frequency points: 201 to 401  
- Frequency range: **1 MHz – 1 GHz** (or as relevant for your DUT)

**Data Capture**
- Measure complex S-parameters:
  - `frequency`, `S11 Real`, `S11 Imag`, `S21 Real`, `S21 Imag`
- Save results in `.csv` or Touchstone `.s2p` format.

---

### Example: Data Acquisition Script

```python
import pandas as pd
from pynanovna import NanoVNA

vna = NanoVNA()

# Example sweep configuration
vna.set_start(1e6)    # 1 MHz
vna.set_stop(1e9)     # 1 GHz
vna.set_points(401)

# Capture S-parameters
freqs, s11, s21 = vna.capture()

# Save as CSV
df = pd.DataFrame({
    "Frequency (Hz)": freqs,
    "S11 Real": s11.real,
    "S11 Imag": s11.imag,
    "S21 Real": s21.real,
    "S21 Imag": s21.imag
})

df.to_csv("nanovna_data.csv", index=False)
print("Data saved to nanovna_data.csv")
```

---

### B. Feature Engineering and Normalization

**Feature Vector (X):**

The most robust input features are derived from complex S21 and S11 data:

```
|S21|_dB   = 20 * log10(|S21|)
Phase_deg = arg(S21) * (180 / π)
```

Concatenate magnitude and phase into a single 1D array per measurement.

**Normalization:**
- Use **Min-Max Scaling (0–1)** or **Z-score normalization**.

**Labeling (Y):**
- Assign clear labels such as `"LPF 400MHz"`, `"10 dB Attenuator"`, etc.

---

## II. Project 1: Filter Type Classification (Deep Learning)

Train a model to automatically identify the **type of RF circuit** (e.g., LPF, HPF, BPF, Attenuator).

### A. 1D Convolutional Neural Network (1D-CNN)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(32, 7, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
# cnn = build_cnn((401, 2), 4)
# cnn.summary()
```

**Training Example:**

```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

# Example: X (samples, 401, 2), y (labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

cnn.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.15)
cnn.evaluate(X_test, y_test)
```

---

## III. Project 2: Component Value Regression

Predict continuous component values (e.g., attenuation in dB, resistance in Ω).

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_regression_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1)  # Continuous output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Example usage:
# reg_model = build_regression_model(802)
# reg_model.fit(X_train, y_train, epochs=100, validation_split=0.2)
```

---

## IV. Project 3: Calibration Integrity Monitoring (Autoencoder)

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Example usage:
# ae = build_autoencoder(802)
# ae.fit(X_train, X_train, epochs=50, batch_size=32)
```

**Reconstruction Error Thresholding Example:**
```python
reconstructions = ae.predict(X_test)
errors = np.mean(np.square(X_test - reconstructions), axis=1)
threshold = np.mean(errors) + 3 * np.std(errors)
anomalies = errors > threshold
print(f"Detected {np.sum(anomalies)} anomalies.")
```

---

## 🧰 Tools & Libraries

- Python 3.9+  
- PyNanoVNA (or equivalent)  
- NumPy, Pandas, Scikit-learn  
- TensorFlow / PyTorch  
- Matplotlib / Plotly

---

**Author:** K Seunarine  
**License:** MIT  
**Last Updated:** 21 October 2025
