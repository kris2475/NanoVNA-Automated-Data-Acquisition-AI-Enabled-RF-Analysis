# 📡 NanoVNA Automation and Coaxial Cable Analyser with TDR Instructions

This repository provides Python scripts for **automating NanoVNA data capture** and performing **software-based Time Domain Reflectometry (TDR)** to analyse coaxial cables. The workflow automatically records complex S-parameters and derives each cable’s **Electrical Length** and **Velocity Factor (VF)** for characterisation and quality assessment.

---

## 🧩 Project Overview

The workflow comprises two primary stages:

### 1. Data Acquisition (`nanovna_capture.py`)
- Connects to the NanoVNA, performs configurable frequency sweeps, and stores complex S-parameter data in a single CSV file.
- The latest script prompts the user to enter the number of cables (sweeps) interactively.

### 2. Data Analysis (`cable_analysis.py`)
- Loads the raw sweep data, performs an **Inverse Fast Fourier Transform (IFFT)** to generate a TDR trace, and calculates each cable’s electrical length and velocity factor.

---

## ⚙️ Prerequisites

### 1. System Requirements
- Python 3.x
- NanoVNA connected via USB and powered on

### 2. Required Libraries
Install the required dependencies using:
```bash
pip install pynanovna numpy pandas scipy
```

### 3. Calibration (CRITICAL)
- Perform a **full SOLT (Short, Open, Load, Thru) calibration** at the reference plane of Port 1 before measuring cables.

---

## 🧠 Script 1: Data Acquisition — `nanovna_capture.py`

Automates multiple sweeps and saves all S-parameter data to a labelled CSV file.

### Configuration
- The latest script prompts for the following inputs:
  - Number of sweeps (cables)
  - Start frequency (MHz)
  - Stop frequency (MHz)
  - Number of points per sweep
- Defaults can be accepted by pressing Enter.

### Connection Setup
- Connect **one end of the cable** to **Port 1 (CH0)** of the NanoVNA.
- Leave the **far end of the cable open**.
- **Do not connect Port 2** during reflection measurement.

### Usage
```bash
python nanovna_capture.py
```
- Follow prompts to attach each cable and press ENTER for each sweep.

### Output
- Generates a timestamped CSV file, e.g.:
```
nanovna_batch_sweep_20251017_180100.csv
```
- Each row contains:
```
Sample_ID, Frequency (Hz), S11 Mag (dB), S11 Phase (deg),
S21 Mag (dB), S21 Phase (deg), S11 Real, S11 Imag, S21 Real, S21 Imag
```

---

## 🔬 Script 2: Data Analysis — `cable_analysis.py`

Processes the captured data to compute cable properties through software TDR analysis.

### Key Concepts
- **Time Domain Reflectometry (TDR):** Converts S11 frequency-domain data into a time-domain reflection trace using IFFT.
- **Velocity Factor (VF):** Ratio of signal speed in the cable to the speed of light (0–1). Indicates cable type and quality.

### Preparation
- Ensure each cable was measured with **open circuit termination**.
- Set the physical length of your cables in the script:
```python
PHYSICAL_LENGTH_METERS = 1.0
```

### Usage
```bash
python cable_analysis.py nanovna_batch_sweep_20251017_180100.csv
```

### Output
- Generates `cable_tdr_summary.csv` containing:
```
Sample_ID, Physical_Length_Input (m), Calculated_Electrical_Length (m), Calculated_Velocity_Factor (VF)
```

### Method Summary
1. Load S11 complex data from the CSV file.
2. Perform an IFFT to obtain the time-domain TDR trace.
3. Locate the reflection peak corresponding to the open end.
4. Compute:
   - **Electrical Length:** L_e = (C * t)/2
   - **Velocity Factor:** VF = L_e / L_physical
5. Save results to a summary CSV.

### Example Output
| Sample_ID | Physical_Length_Input (m) | Calculated_Electrical_Length (m) | Calculated_Velocity_Factor (VF) |
|-----------|----------------------------|---------------------------------|--------------------------------|
| 1         | 1.0                        | 0.67                            | 0.67                           |
| 2         | 1.0                        | 0.80                            | 0.80                           |
| 3         | 1.0                        | 0.69                            | 0.69                           |

---

## 🧭 Next Steps
- **Compare Cable Types:** Use VF to differentiate cables.
- **Quality Assurance:** Identify degraded or water-damaged cables by unusually low VF readings.

---

## 🧰 References
- NanoVNA Official Documentation
- pynanovna Library
- SciPy FFT Documentation

---

## 📜 Licence
Released under the MIT Licence — free to use, modify, and distribute with attribution.

**Author:** K Seunarine  
**Last Updated:** 17 October 2025

