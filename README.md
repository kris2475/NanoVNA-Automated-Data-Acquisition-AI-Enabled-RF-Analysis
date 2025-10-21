# 🧭 NanoVNA Automated Data Acquisition Script (v2.9)

## 🧩 Introduction

Modern RF and microwave research often demands rapid, repeatable, and accurate acquisition of **S-parameter data** across multiple circuits. Manual data collection using a **Vector Network Analyser (VNA)** can be time-consuming, inconsistent, and susceptible to human error — especially when multiple devices or frequency ranges are involved.

The **NanoVNA Automated Data Acquisition Script** was developed to address these challenges for users of the **NanoVNA** platform. It provides a fully automated method for sweeping, averaging, and storing calibrated measurement data across a predefined list of circuits, each with its own optimised frequency range.

Version **2.9** introduces important refinements that improve measurement reliability, simplify logging, and align sweep limits with NanoVNA hardware capabilities.

---

## 🧾 Executive Summary

The **NanoVNA Automated Data Acquisition Script (v2.9)** is a Python-based automation utility for laboratory and research environments. It communicates directly with a connected NanoVNA to:

- Select and execute appropriate **frequency sweep profiles** for each circuit type.  
- Perform **multiple sweeps** and calculate averaged S-parameters for enhanced stability.  
- Save processed data into well-structured **CSV files** for every circuit tested.  
- Generate a **summary table** of all sweeps performed.

Version 2.9 adds:

- A **hard limit of 900 MHz** for all WIDE and HIGH sweeps — consistent with NanoVNA hardware limits.  
- A **logging system improvement**, suppressing repetitive and non-critical warnings for a cleaner terminal output.

Together, these changes deliver a **robust, hands-free data acquisition workflow** that ensures reproducible, high-quality results suitable for documentation, analysis, and research publication.

---

## 🚀 Overview

This Python script automates the process of collecting S-parameter data from a connected **NanoVNA (Vector Network Analyser)**.  
It iterates through a predefined list of electronic circuits, applies optimal sweep ranges, averages multiple readings, and saves the resulting data in a structured format for easy review.

---

### 🆕 Key Updates in Version 2.9

- **900 MHz Frequency Cap**  
  All *WIDE* and optimised *HIGH* sweeps now stop at **900 MHz**, maintaining compatibility with NanoVNA hardware constraints.

- **Reliable Logging Fix**  
  Internal logging has been improved to filter out repetitive `pynanovna` warnings such as *“No calibration has been applied”*, resulting in clearer terminal output.

---

## ✨ Features

- **Intelligent Sweep Management**
  - **LOW Sweep:** 1 MHz – 30 MHz for low-frequency resonant circuits (e.g. Circuits 5 & 6).  
  - **C11 Sweep:** 10 MHz – 600 MHz, optimised for the 400 MHz Low-Pass Filter.  
  - **C12 Sweep:** 100 MHz – 900 MHz, optimised for the 500 MHz High-Pass Filter.  
  - **WIDE Sweep:** 1 MHz – 900 MHz (default).

- **Data Averaging**  
  Performs several sweeps (default: 3) and averages the complex S-parameters (real/imaginary) to reduce noise and measurement variability.

- **Low-Frequency Verification**  
  Checks and corrects cases where the NanoVNA fails to initialise the low start frequency (a known issue on some models).

- **Automated Data Storage**  
  Saves all results automatically into the `nanovna_data/` folder, generating individual CSV files and a combined summary table.

---

## 💻 Prerequisites

### Hardware
- A **NanoVNA** device connected to the computer via USB.

### Software
- **Python 3.8 or newer**
- Required Python libraries:
  ```bash
  pip install pynanovna numpy pandas
  ```

---

## ⚙️ Configuration

All primary configuration options are located at the top of the script and can be modified to suit your hardware or experimental needs.

| Constant | Description | Default Value |
|-----------|--------------|----------------|
| `SAVE_FOLDER` | Directory where all CSV files are stored. | `nanovna_data` |
| `REPEATS` | Number of sweeps performed for averaging. | `3` |
| `WIDE_STOP_FREQ` | Maximum frequency used in any sweep. | `900e6` (900 MHz) |
| `CIRCUITS` | List of circuits and port counts. | *(See script for full list)* |

### Example Circuit Definitions

Circuits are defined using a tuple format:  
`('Circuit_Name', <Port_Count>)`

```python
CIRCUITS = [
    # Uses LOW_SWEEP
    ('Circuit_5_6_5MHz_Ceramic_Notch_Filter', 2),
    # Uses C11_SWEEP
    ('Circuit_11_400MHz_Low_Pass_Filter', 2),
    # Uses WIDE_SWEEP (default)
    ('Circuit_7_RC_Series_Circuit', 1),
]
```

---

## ▶️ Usage

1. **Run the script:**
   ```bash
   python your_script_name.py
   ```

2. **Follow on-screen prompts:**
   The script guides you through each circuit measurement:
   ```
   ➡️ Attach input and output coax to Circuit 11: ‘400 MHz Low-Pass Filter’ (Step 11 of 20, 2-port).
   Press Enter when ready…
   ```

3. **Connect and proceed:**  
   Attach the specified circuit to the NanoVNA, then press **Enter** to begin acquisition.

4. **Low-frequency check (Circuits 5 & 6 only):**  
   If the VNA cannot set the 1 MHz start frequency, the script pauses and prompts you to adjust it manually before continuing.

5. **Completion:**  
   After all circuits are tested, the script disconnects from the NanoVNA and displays a summary of the collected data.

---

## 📂 Output Data

All data are saved automatically in the `nanovna_data/` directory.

### 🔹 Circuit Data Files (`.csv`)
Each circuit produces a uniquely named file, for example:
```
Circuit_12_500MHz_High_Pass_Filter_c12_sweep.csv
```

| Column | Description |
|---------|--------------|
| `frequency_Hz` | Frequency points (Hz). |
| `S11_real`, `S11_imag` | Averaged real and imaginary parts of S₁₁. |
| `S11_mag_dB`, `S11_phase_deg` | Magnitude (dB) and phase (°) of S₁₁. |
| `S21_real`, `S21_imag` | Averaged real and imaginary parts of S₂₁ (2-port circuits only). |
| `S21_mag_dB`, `S21_phase_deg` | Magnitude (dB) and phase (°) of S₂₁ (2-port circuits only). |

### 🔹 Summary File

A single summary file is generated:
```
summary_table_optimised.csv
```
This file lists sweep parameters, circuit identifiers, and measurement details for all tested circuits.

---

## 🧰 Troubleshooting

| Issue | Possible Cause / Solution |
|--------|-----------------------------|
| **Device not detected** | Ensure the NanoVNA is powered on and recognised as a USB serial device before execution. |
| **Low-frequency initialisation error** | Manually set the start frequency to 1 MHz via the NanoVNA interface when prompted. |
| **Permission denied (Linux)** | Add your user to the `dialout` group to permit serial access:<br>`sudo usermod -aG dialout $USER` |

---

## 🧪 Recommended Workflow

1. Calibrate the NanoVNA before commencing automated measurements.  
2. Verify circuit connections prior to confirming each sweep.  
3. After completion, review both the circuit CSV files and the summary table to confirm data quality and integrity.

---

## 📜 Licence

This project is distributed under the **MIT Licence**.  
You are free to use, adapt, and share this script for academic, research, or educational purposes, provided that appropriate attribution is given.

---

## 📘 Project Details

- **Author:** *[Your Name]*  
- **Version:** 2.9  
- **Last Updated:** October 2025  
- **Output Directory:** `nanovna_data/`

---

> *“Precision through automation — empowering consistent RF measurement workflows.”*
> 

