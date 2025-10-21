# Leveraging Deep Learning for Advanced RF Component Analysis with the NanoVNA Test Board

The integration of low-cost vector network analysis (VNA) with modern data science techniques offers a powerful new paradigm for radio frequency (RF) engineering.  

This project outlines a complete workflow — from **data acquisition** to **deep learning model deployment** — focused on utilizing **Machine Learning (ML)** and **Deep Learning (DL)** to characterize, classify, and monitor components on the **NanoVNA Filter Attenuator VNA RF Test Board Tester Demo Kit**.

By combining affordable measurement hardware with advanced AI techniques, this framework maximizes the utility of the test board as a versatile data source for complex **S-parameter** measurements.

---

## I. Project Foundation: Data Acquisition and Preprocessing

Accurate and automated data collection is the cornerstone of any successful ML project. The Python ecosystem provides robust tools for automating the NanoVNA.

### A. Data Acquisition Automation

**Hardware Setup:**
- Connect the NanoVNA to the host PC via USB.  
- Attach the VNA ports:
  - CH0 (Port 1) and CH1 (Port 2) to the RF Test Board using SMA cables.  
- Perform a **full 2-port SOLT calibration (Short, Open, Load, Thru)** at the reference plane to remove cable and connector parasitics.

**Tooling:**
- Use a Python library such as [`pynanovna`](https://github.com) or equivalent scripts to control the VNA.

**Sweep Settings:**
- High-resolution frequency sweep: 201–401 points.  
- Frequency range: **1 MHz – 1 GHz** (or relevant for your DUT).

**Data Capture:**
- Measure complex S-parameters:  
  - `frequency`, `S11 Real`, `S11 Imag`, `S21 Real`, `S21 Imag`  
- Save results as `.csv` or Touchstone `.s2p` files.

### B. Feature Engineering and Normalization

**Feature Vector (X):**
The most robust input features are derived from complex `S21` and `S11` data:

\[
\text{Mag}_{dB} = 20 \cdot \log_{10}(|S_{21}|)
\]
\[
\text{Phase}_{deg} = \arg(S_{21}) \cdot \frac{180}{\pi}
\]

Concatenate magnitude and phase into a 1D array per measurement.

**Normalization:**
- Apply **Min-Max Scaling (0–1)** or **Z-score normalization**.

**Labeling (Y):**
- Assign clear labels such as `"LPF 400MHz"`, `"10 dB Attenuator"`, etc.

---

## II. Project 1: Filter Type Classification (Deep Learning)

Train a model to automatically identify the **type of RF circuit** (e.g., LPF, HPF, BPF, Attenuator).

### A. ML Task and Data Strategy

- **Task:** Classification (predict a discrete category).  
- **Data:** Focus on `S21` sweeps of all filters and attenuators.

**Data Augmentation (Critical):**
Generate diverse data by:
- Shifting the VNA start/stop frequency slightly.  
- Varying temperature conditions.  
- Measuring on different days.

**Goal:** ~50–100 unique sweeps per filter type.

### B. 1D Convolutional Neural Network (1D-CNN) Architecture

| Layer Type | Configuration | Purpose |
|-------------|---------------|----------|
| Input Layer | X vector (e.g., 401 points × 2 channels) | Receives S21 Mag and Phase |
| Conv1D | 32 filters, kernel size 7, ReLU | Learns local spectral features |
| MaxPooling1D | Pool size 2 | Downsampling for invariance |
| Conv1D | 64 filters, kernel size 5, ReLU | Learns abstract filter signatures |
| Flatten | — | Converts 2D feature maps into 1D vector |
| Dense (Hidden) | 128 neurons, ReLU | Feature combination |
| Output Layer | N neurons, Softmax | Class probabilities |

### C. Training and Evaluation

- **Loss Function:** `Categorical Crossentropy`  
- **Metrics:** Accuracy, Precision, Recall  
- **Data Split:**  
  - Train: 70%  
  - Validation: 15%  
  - Test: 15%

---

## III. Project 2: Component Value Regression (R/L/C and Attenuation)

Predict continuous component values (e.g., attenuation in dB, resistance in Ω) from their frequency response.

### A. ML Task and Data Strategy

- **Task:** Regression  
- **Data:**  
  - Use `S11` for RLC circuits (reflection).  
  - Use `S21` for attenuators (transmission).  
- **Labels:** Continuous values (e.g., `2.98 dB`, `49.5 Ω`).

### B. Prediction Model Sketch

Use a simple **Feed-forward Neural Network (FNN)**.

**Feature Reduction (Optional):**
Extract key KPIs:
- Mean `S21` magnitude (passband)
- 3 dB cutoff frequency
- Minimum return loss (`S11_min`)

**Model:**  
Input → Hidden (x2, ReLU) → Output  

**Training:**
- **Loss:** MSE or MAE  
- **Metrics:** R² Score, RMSE

---

## IV. Project 3: Calibration Integrity Monitoring (Anomaly Detection)

Monitor the integrity of the VNA calibration standards (Short, Open, Load, Thru).

### A. ML Task and Data Strategy

- **Task:** Anomaly Detection  
- **Model:** Autoencoder  
- **Data:** Collect a clean baseline of `S11` for the **50 Ω Load**.

### B. Autoencoder Workflow

**Training:**
- Model learns to reconstruct nominal (healthy) 50 Ω sweeps.

**Testing/Monitoring:**
- Faulty or degraded measurements yield **higher reconstruction errors**.

**Alerting:**
- Define an **Anomaly Threshold** based on nominal error.  
- If reconstruction error > threshold → trigger alert.

---

## 🧠 Summary

| Project | Task | Model | Objective |
|----------|------|--------|------------|
| 1 | Classification | 1D-CNN | Identify circuit type |
| 2 | Regression | FNN | Predict component value |
| 3 | Anomaly Detection | Autoencoder | Monitor calibration health |

---

## 🧰 Tools & Libraries

- **Python**: Data collection & processing  
- **PyNanoVNA** (or equivalent): Instrument control  
- **NumPy / Pandas**: Data manipulation  
- **Matplotlib / Plotly**: Visualization  
- **TensorFlow / PyTorch**: Deep Learning models  
- **Scikit-learn**: Feature scaling & metrics

---

## 📈 Future Directions

- Extend dataset using multiple NanoVNAs and boards  
- Deploy trained models for **real-time classification**  
- Integrate results into a **web-based RF dashboard**

---

**Author:** K Seunarine  
**License:** MIT  
**Last Updated:** 21 October 2025
