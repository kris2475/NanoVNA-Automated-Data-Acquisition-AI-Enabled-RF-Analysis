# 🧭 NanoVNA Automated Data Acquisition Script (v2.9)

## 🧩 Overview

Collecting **S-parameter measurements** for RF and microwave circuits can be tedious and error-prone, especially with multiple circuits or wide frequency ranges. Manual operation of a **Vector Network Analyzer (VNA)** slows research workflows and introduces inconsistencies.

The **NanoVNA Automated Data Acquisition Script** automates frequency sweeps, averaging, and data storage for multiple circuits. Each circuit uses its optimized frequency range to ensure high-quality, reproducible measurements.

Version **2.9** improves stability, cleans up logging, and prevents exceeding the NanoVNA's hardware frequency limits.

---

## 🧾 Summary

This Python-based tool:

- Automatically configures and runs **frequency sweeps** per circuit.
- Performs multiple sweeps and averages S-parameters to reduce noise.
- Saves measurements in structured **CSV files** per circuit.
- Generates a **summary table** consolidating all results.

**Version 2.9 updates:**

- Maximum frequency for all *WIDE* and *HIGH* sweeps capped at **900 MHz**.
- Logging suppresses repetitive non-critical messages for clearer terminal output.

---

## 🚀 Features

- **Custom Sweep Profiles:**  
  - **LOW:** 1–30 MHz (for low-frequency resonators, Circuits 5 & 6)  
  - **C11:** 10–600 MHz (optimized for 400 MHz Low-Pass Filter)  
  - **C12:** 100–900 MHz (optimized for 500 MHz High-Pass Filter)  
  - **WIDE:** 1–900 MHz (default)  

- **Averaged Measurements:** Multiple sweeps (default: 3) to reduce variability.
- **Low-Frequency Verification:** Detects and corrects 1 MHz start frequency initialization issues.
- **Automated Data Storage:** Saves CSV files per sweep and a summary table in `sweeps/`.

---

## 💻 Requirements

### Hardware
- **NanoVNA** connected via USB.

### Software
- Python **3.8+**
- Python packages: `pynanovna`, `numpy`, `pandas`, `matplotlib`, `pyserial`

Install missing packages automatically or via:
```bash
pip install pynanovna numpy pandas matplotlib pyserial
```

---

## ⚙️ Configuration

Main settings are at the top of the script:

| Setting | Description | Default |
|---------|-------------|---------|
| `SAVE_FOLDER` | Folder for CSV output | `sweeps` |
| `SWEEP_AVERAGE_COUNT` | Number of sweeps per averaged measurement | `3` |
| `SWEEPS_PER_BATCH` | Number of sweeps per circuit in a batch | `100` |
| `MAX_POINTS` | Points per sweep | `1024` |
| `circuits` | Circuit names and port counts | *(see script)* |

**Example circuit definition:**  
```python
circuits = [
    ('Circuit_5_6_5MHz_Ceramic_Notch_Filter', 2),
    ('Circuit_11_400MHz_Low_Pass_Filter', 2),
    ('Circuit_7_RC_Series_Circuit', 1),
]
```

---

## ▶️ Usage

1. Run the script:
```bash
python your_script_name.py
```

2. Follow the prompts for each circuit:
```
➡️ Attach input/output to Circuit 11 (2-port). Press Enter…
```

3. Connect the specified circuit to the NanoVNA.

4. For circuits 5 & 6, manually confirm 1 MHz start if prompted.

5. After all sweeps, the NanoVNA disconnects and a summary table is displayed.

---

## 📂 Output

All data are saved in `sweeps/`.

### Circuit CSV Files
Example:
```
Circuit_12_500MHz_High_Pass_Filter_Sweep_001.csv
```

| Column | Description |
|--------|-------------|
| `frequency_Hz` | Sweep points (Hz) |
| `S11_real`, `S11_imag` | Averaged real & imaginary parts |
| `S11_mag_dB`, `S11_phase_deg` | Magnitude (dB) & phase (°) |
| `S21_real`, `S21_imag` | 2-port S21 real & imaginary |
| `S21_mag_dB`, `S21_phase_deg` | 2-port S21 magnitude & phase |

### Summary Table
`summary_table_optimised.csv` – Consolidates sweep metadata and circuit info.

---

## 🧰 Troubleshooting

| Problem | Solution |
|---------|---------|
| NanoVNA not detected | Ensure USB connection and device power. |
| Low-frequency error | Set start frequency to 1 MHz manually. |
| Linux permission denied | Add user to `dialout`: `sudo usermod -aG dialout $USER` |

---

## 🧪 Recommended Workflow

1. Calibrate NanoVNA before starting automated measurements.
2. Confirm connections for each circuit.
3. Review CSV and summary data for completeness and quality.

---

## 📜 License

MIT License – free to use, adapt, and share for academic, research, or educational purposes with proper attribution.

---

## 📘 Project Info

- **Author:** K Seunarine  
- **Version:** 2.9  
- **Last Updated:** October 2025  
- **Output Folder:** `sweeps/`

> *“Precision through automation — empowering consistent RF measurement workflows.”*
