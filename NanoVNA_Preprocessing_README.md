# Preprocessing Data for NanoVNA Analysis

---

## 1. Introduction: The Need for Preprocessing

This document outlines the **preprocessing** workflow for the raw data collected from **NanoVNA Vector Network Analyzer (VNA)** experiments.

The core purpose of this phase is to transform real-world, highly variable frequency sweep measurements into a **clean, robust, and unified dataset**. This homogenization is critical to ensure that downstream analysis, particularly **machine learning model training**, is reliable. The goal is to make sure the model focuses on the intrinsic characteristics of the circuits rather than measurement artifacts or data inconsistencies.

---

## 2. Input Data

The preprocessing scripts expect a collection of raw S-parameter measurements, typically sourced from the VNA.

* **Format**: Data must be in **Comma Separated Value (`.csv`)** format.
* **Source**: Each `.csv` file should contain the results of a single frequency sweep for a specific circuit under test.
* **Expected Columns**: Files are expected to contain the core VNA measurements:
    * `Frequency` (Hz)
    * `S11 Magnitude` (Reflection coefficient magnitude)
    * `S11 Phase` (Reflection coefficient phase)
    * `S21 Magnitude` (Transmission coefficient magnitude - *Optional*)
    * `S21 Phase` (Transmission coefficient phase - *Optional*)

---

## 3. Preprocessing Steps

The preprocessing logic is primarily implemented in the **`NanoVNA_circuits.ipynb`** Jupyter notebook and includes the following essential steps:

### 3.1. Handling Inconsistent S-Parameter Data (One-Port vs. Two-Port)

A major challenge in raw VNA data is the **inconsistent presence of S21 columns** (Transmission Coefficient). This inconsistency arises because:

* **One-Port Circuits** (e.g., shorts, opens, terminations, or simple antennas) are measured using only Port 1 and will typically **only provide `S11` data**.
* **Two-Port Circuits** (e.g., filters, attenuation networks, amplifiers) require measurements across both Port 1 and Port 2, providing both **`S11` and `S21` data**.

**Standardization Action:** To create a consistent feature space for machine learning, the script must:

1.  Identify measurements missing `S21` columns.
2.  **Impute Missing Columns**: For these one-port measurements, the columns (`S21 Magnitude` and `S21 Phase`) are explicitly created and populated with a placeholder value (e.g., **zeroes or NaN values**). This ensures every measurement, regardless of the circuit type, occupies the same dimensional space for the model.

### 3.2. Standardization and Normalization

**Why Standardization?** Raw VNA measurements are susceptible to **measurement drift** and noise caused by:
* **Instrument Variability**: Minor non-linearity or slight component drift across the VNA's frequency range or over time.
* **Environmental Factors**: Temperature changes, cable movement, or slight variations in setup between different measurement sessions.

**Standardization Actions:** To mitigate these biases and ensure that feature scales don't unduly influence the model, the data undergoes:

1.  **Baseline Referencing**: Measurements are often standardized against a known baseline (e.g., subtracting a calibration sweep) to remove fixture-specific artifacts.
2.  **Feature Scaling**: Data is then scaled (e.g., Min-Max scaling or Z-score normalization) to ensure that features with naturally large values (like S-parameter magnitude) do not unfairly dominate the features with smaller values (like phase) in the downstream model training process. The result is new normalized columns (e.g., `S11 Magnitude Normalized`).

### 3.3. Feature Engineering and Labeling

* **Feature Engineering**: The script calculates derived features, most notably converting the S-parameters from the native magnitude/phase representation to **Real/Imaginary (Cartesian) components**. This provides the model with a more complete mathematical description of the complex impedance.
* **Labeling**: A categorical label (the circuit type or configuration) is assigned to each sweep measurement, typically by parsing the filenames and mapping them to known circuit classes.

---

## 4. How to Run

1.  **Placement**: Ensure all raw NanoVNA sweep files (`*.csv`) are located in the designated input directory (e.g., `data/raw/`).
2.  **Execution**: Open the **`NanoVNA_circuits.ipynb`** notebook in a Jupyter environment and execute all cells sequentially.

---

## 5. Output Data

Upon successful completion, the script saves the processed data into the designated output directory (e.g., `data/processed/`).

* **Primary Output**: A single, consolidated file (e.g., `processed_nanovna_dataset.csv`) containing the full dataset.
* **Content**: This output file includes the raw frequency and S-parameter data, along with all newly calculated features, normalized values, and the associated, consistent circuit labels.
