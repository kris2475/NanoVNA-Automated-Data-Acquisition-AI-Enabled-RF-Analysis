# 📡 NanoVNA Automation and Coaxial Cable Analyser

This repository provides Python scripts for **automating NanoVNA data capture** and performing **software-based Time Domain Reflectometry (TDR)** to analyse coaxial cables. The workflow automatically records complex S-parameters and derives each cable’s **Electrical Length** and **Velocity Factor (VF)** for characterisation and quality assessment.

---

## 🧩 Project Overview

The workflow comprises two primary stages:

1. **Data Acquisition (`nanovna_capture.py`)**  
   Connects to the NanoVNA, performs configurable frequency sweeps, and stores complex S-parameter data in a single CSV file.

2. **Data Analysis (`cable_analysis.py`)**  
   Loads the raw sweep data, performs an Inverse Fast Fourier Transform (IFFT) to generate a TDR trace, and calculates each cable’s electrical length and velocity factor.

---

## ⚙️ Prerequisites

### 1. System Requirements
- **Python 3.x**
- **NanoVNA** connected via USB and powered on

### 2. Required Libraries
Install the required dependencies using:
```bash
pip install pynanovna numpy pandas scipy
```

### 3. Calibration (CRITICAL)
For accurate results, carry out a full **SOLT (Short, Open, Load, Thru)** calibration on your NanoVNA for the desired frequency range **before** data collection.

---

## 🧠 Script 1: Data Acquisition — `nanovna_capture.py`

Automates multiple sweeps and saves all S-parameter data to a labelled CSV file.

### Configuration
Edit the following variables in the script:
```python
NUM_SWEEPS = 10      # Number of cable samples
START_FREQ = 100e6   # Start frequency (Hz)
STOP_FREQ = 1000e6   # Stop frequency (Hz)
POINTS = 101         # Data points per sweep
```

### Usage
Run the script and follow on-screen instructions:
```bash
python nanovna_capture.py
```

You will be prompted to connect a new cable between each sweep.

### Output
A timestamped CSV file is generated, e.g.:
```
nanovna_batch_sweep_20251017_180100.csv
```

Each row contains:
```
Sample_ID, Frequency (Hz), S11 Mag (dB), S11 Phase (deg),
S21 Mag (dB), S21 Phase (deg), S11 Real, S11 Imag, S21 Real, S21 Imag
```

---

## 🔬 Script 2: Data Analysis — `cable_analysis.py`

Processes the captured data to compute cable properties through software TDR analysis.

### Key Concepts

- **Time Domain Reflectometry (TDR):**  
  Converts S11 frequency-domain data into a time-domain reflection trace using IFFT to identify discontinuities and the cable’s electrical length.

- **Velocity Factor (VF):**  
  The ratio of signal propagation speed through the cable to the speed of light (0–1). Indicates cable type and quality.

---

### Preparation
- Ensure each cable was measured with an **open circuit termination** (nothing connected at the far end).  
- Set the **known physical length** within the script:
  ```python
  PHYSICAL_LENGTH_METERS = 1.0
  ```

---

### Usage
Run the script with your CSV file:
```bash
python cable_analysis.py nanovna_batch_sweep_20251017_180100.csv
```

### Output
Creates a results file:
```
cable_tdr_summary.csv
```
containing:
```
Sample_ID,
Physical_Length_Input (m),
Calculated_Electrical_Length (m),
Calculated_Velocity_Factor (VF)
```

---

## 🧮 Method Summary

1. Load S11 complex data from the CSV file.  
2. Perform an IFFT to obtain the time-domain TDR trace.  
3. Locate the reflection peak corresponding to the open end.  
4. Compute:
   - **Electrical Length:**  
     \[
     L_e = \frac{C \cdot t}{2}
     \]
   - **Velocity Factor:**  
     \[
     VF = \frac{L_e}{L_{physical}}
     \]
5. Save the calculated parameters to a summary file.

---

## 🧾 Example Output

| Sample_ID | Physical_Length_Input (m) | Calculated_Electrical_Length (m) | Calculated_Velocity_Factor (VF) |
|------------|---------------------------|----------------------------------|----------------------------------|
| 1          | 1.0                       | 0.67                             | 0.67                             |
| 2          | 1.0                       | 0.80                             | 0.80                             |
| 3          | 1.0                       | 0.69                             | 0.69                             |

---

## 🧭 Next Steps

After analysis, you can:

- **Compare Cable Types:**  
  Differentiate between cable types based on calculated **VF** values.

- **Perform Quality Assurance:**  
  Identify degraded or water-damaged cables by detecting unusually low **VF** readings.

---

## 🧰 References

- [NanoVNA Official Documentation](https://nanovna.com/)  
- [pynanovna Library](https://pypi.org/project/pynanovna/)  
- [SciPy FFT Documentation](https://docs.scipy.org/doc/scipy/reference/fft.html)

---

## 📜 Licence

Released under the **MIT Licence** — free to use, modify, and distribute with attribution.

---

**Author:** K Seunarine  
**Last Updated:** 17 October 2025
