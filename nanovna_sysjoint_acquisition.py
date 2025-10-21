"""
====================================================================================================
NanoVNA Automated Data Acquisition Script for SYSJOINT RF Test Board
====================================================================================================

Author       : K Seunarine
Date         : 21 October 2025
Version      : 1.1
License      : MIT License
Contact      : [Your Email Here]

Purpose:
--------
This script automates the acquisition of S-parameter measurements (S11 and S21) from the SYSJOINT 
NanoVNA Filter Attenuator RF Test Board containing 18 circuits. It supports both 1-port and 2-port 
circuits, automatically prompts the user for coax connections, computes magnitude and phase, and 
saves data in CSV format ready for machine learning or deep learning workflows.

Key Features:
-------------
1. **Supports 1-port and 2-port circuits**:
   - 1-port circuits (e.g., resistors, capacitors, RLC circuits) → S11 only.
   - 2-port circuits (e.g., filters, attenuators, Thru) → S21 + S11.

2. **Interactive prompts**:
   - Guides the user to attach/detach coax cables correctly for each circuit.

3. **Data processing**:
   - Computes magnitude in dB and phase in degrees for all measured S-parameters.

4. **Data saving**:
   - Each circuit's data saved as a separate CSV.
   - Summary table generated with sweep parameters, circuit name, and file paths.

5. **Repeatable measurements**:
   - Supports multiple sweeps per circuit with averaging for improved data quality.

6. **ML-ready data**:
   - Ensures consistent column naming and format for easy use in ML/DL pipelines.

Dependencies:
-------------
- Python 3.8+
- pynanovna
- numpy
- pandas
- os
- time

Usage Instructions:
-------------------
1. Connect the NanoVNA to your PC via USB and identify the COM/serial port.
2. Perform a full 1-port or 2-port SOLT calibration using the included standards.
3. Run this script. Follow prompts to attach coax for each circuit.
4. The script will save CSV files in the folder 'nanovna_data' and generate a summary table.
5. After all circuits are measured, the NanoVNA is automatically disconnected.

====================================================================================================
"""

import pandas as pd
import numpy as np
from pynanovna import NanoVNA
import os
import time

# --------------------------
# CONFIGURATION
# --------------------------
PORT = 'COM3'  # Replace with your NanoVNA COM port
SAVE_FOLDER = 'nanovna_data'
os.makedirs(SAVE_FOLDER, exist_ok=True)

START_FREQ = 1e6   # 1 MHz
STOP_FREQ = 1e9    # 1 GHz
POINTS = 401
REPEATS = 3        # Number of sweeps per circuit for averaging

# --------------------------
# CIRCUIT DEFINITIONS
# --------------------------
# (Circuit Name, Port Type: 1 or 2)
CIRCUITS = [
    ('RLC_Series_Parallel_1', 1),
    ('RLC_Series_Parallel_2', 1),
    ('33_Ohm_Resistor', 1),
    ('75_Ohm_Resistor', 1),
    ('6_5MHz_Ceramic_Notch_Filter', 2),
    ('10_7MHz_Ceramic_Notch_Filter', 2),
    ('RC_Series_Circuit', 1),
    ('LC_Series_Circuit', 1),
    ('Capacitor', 1),
    ('Inductor', 1),
    ('400MHz_Low_Pass_Filter', 2),
    ('500MHz_High_Pass_Filter', 2),
    ('Short_Circuit', 1),
    ('Open_Circuit', 1),
    ('50_Ohm_Load', 1),
    ('Thru_Circuit', 2),
    ('10dB_Attenuation_Circuit', 2),
    ('3dB_Attenuation_Circuit', 2)
]

# --------------------------
# CONNECT TO NANOVNA
# --------------------------
vna = NanoVNA()
vna.connect(port=PORT)
vna.set_sweep(start=START_FREQ, stop=STOP_FREQ, points=POINTS)

# --------------------------
# DATA ACQUISITION LOOP
# --------------------------
summary = []

for idx, (name, ports) in enumerate(CIRCUITS, start=1):
    if ports == 1:
        prompt = f"\n➡️ Attach coax to '{name}' (Circuit {idx}, 1-port). Press Enter when ready..."
    else:
        prompt = f"\n➡️ Attach input and output coax to '{name}' (Circuit {idx}, 2-port). Press Enter when ready..."
    
    input(prompt)

    # Initialize accumulation arrays for averaging
    freq_accum = None
    s11_real_accum = None
    s11_imag_accum = None
    s21_real_accum = None
    s21_imag_accum = None

    # Perform repeated sweeps
    for rep in range(REPEATS):
        data = vna.capture_sweep()

        if freq_accum is None:
            freq_accum = data['frequency']
            s11_real_accum = data['S11'].real.copy()
            s11_imag_accum = data['S11'].imag.copy()
            if ports == 2:
                s21_real_accum = data['S21'].real.copy()
                s21_imag_accum = data['S21'].imag.copy()
        else:
            s11_real_accum += data['S11'].real
            s11_imag_accum += data['S11'].imag
            if ports == 2:
                s21_real_accum += data['S21'].real
                s21_imag_accum += data['S21'].imag

        print(f"   Sweep {rep+1}/{REPEATS} complete")
        time.sleep(0.5)

    # Average results
    s11_real_avg = s11_real_accum / REPEATS
    s11_imag_avg = s11_imag_accum / REPEATS
    if ports == 2:
        s21_real_avg = s21_real_accum / REPEATS
        s21_imag_avg = s21_imag_accum / REPEATS

    # Prepare DataFrame
    df = pd.DataFrame({
        'frequency_Hz': freq_accum,
        'S11_real': s11_real_avg,
        'S11_imag': s11_imag_avg
    })
    df['S11_mag_dB'] = 20 * np.log10(np.sqrt(df['S11_real']**2 + df['S11_imag']**2))
    df['S11_phase_deg'] = np.degrees(np.arctan2(df['S11_imag'], df['S11_real']))

    if ports == 2:
        df['S21_real'] = s21_real_avg
        df['S21_imag'] = s21_imag_avg
        df['S21_mag_dB'] = 20 * np.log10(np.sqrt(df['S21_real']**2 + df['S21_imag']**2))
        df['S21_phase_deg'] = np.degrees(np.arctan2(df['S21_imag'], df['S21_real']))

    # Save CSV
    filename = f"{name}_sweep.csv"
    filepath = os.path.join(SAVE_FOLDER, filename)
    df.to_csv(filepath, index=False)

    # Update summary
    summary.append({
        'Circuit': name,
        'Ports': ports,
        'Filename': filename,
        'Start_Freq_Hz': START_FREQ,
        'Stop_Freq_Hz': STOP_FREQ,
        'Points': POINTS,
        'Repeats': REPEATS
    })

    print(f"✅ Sweep for '{name}' saved as {filename}")
    time.sleep(1)

# Save summary table
summary_df = pd.DataFrame(summary)
summary_csv = os.path.join(SAVE_FOLDER, 'summary_table.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"\n📄 All sweeps completed. Summary saved as '{summary_csv}'")

# Disconnect NanoVNA
vna.disconnect()
print("🔌 NanoVNA disconnected. Data acquisition finished!")
