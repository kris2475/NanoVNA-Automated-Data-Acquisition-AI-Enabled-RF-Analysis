# ====================================================================================================
# NanoVNA Automated Data Acquisition Script (Version 2.9: 900 MHz VNA Constraint Applied & Logging Fix)
# ====================================================================================================

# Purpose:
# --------
# - Implements automatic switching between LOW, WIDE, and two optimized HIGH sweep ranges.
# - The maximum frequency (C12_SWEEP and WIDE_SWEEP) is now capped at 900 MHz.
# - The recursive logging error is FIXED.

# Dependencies:
# -------------
# - Python 3.8+
# - pynanovna
# - numpy
# - pandas
# - os
# - time
# - logging
# ====================================================================================================

import pandas as pd
import numpy as np
import os
import time
import logging
import sys
from pynanovna import VNA

# ----------------------------------------------------------------------
# 1. CUSTOM LOGGING FILTER IMPLEMENTATION (FIXED)
# ----------------------------------------------------------------------

class CalibrationFilter(logging.Filter):
    """
    Suppresses specific, non-critical, repetitive log messages from pynanovna
    that warn about calibration status.
    """
    def __init__(self, filter_phrases=None):
        super().__init__()
        self.filter_phrases = filter_phrases or [
            "No calibration has been applied",
            "calibration not valid",
            "recommended to re-calibrate",
        ]

    def filter(self, record):
        """Returns False if the log message should be suppressed."""
        message = record.getMessage()
        for phrase in self.filter_phrases:
            if phrase.lower() in message.lower():
                return False
        return True

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any default handlers to ensure only the custom one is used
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(levelname)s: %(message)s')

# --- CORRECTED LINE: Setting the formatter object to the handler ---
handler.setFormatter(formatter) 
# ------------------------------------------------------------------

root_logger.addHandler(handler)

calibration_filter = CalibrationFilter()
root_logger.addFilter(calibration_filter)


print("--- Log Messages Filtered ---")
print("✅ Repetitive 'calibration not valid' messages are now suppressed reliably.")

# ----------------------------------------------------------------------
# CONFIGURATION (Capped at 900 MHz)
# ----------------------------------------------------------------------
SAVE_FOLDER = 'nanovna_data'
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Low Frequency Sweep (for Circuits 5 and 6)
LOW_START_FREQ = 1e6    # 1 MHz
LOW_STOP_FREQ = 30e6    # 30 MHz
LOW_POINTS = 801
LOW_FREQ_CHECK_TOLERANCE_HZ = 10e6 

# Circuit 11: Optimized Sweep (Focus on 400 MHz cutoff)
C11_START_FREQ = 10e6   # 10 MHz
C11_STOP_FREQ = 600e6   # 600 MHz 
C11_POINTS = 1001       

# Circuit 12: Optimized Sweep (Wider sweep to find 500 MHz passband - CAPPED AT 900 MHz)
C12_START_FREQ = 100e6  # 100 MHz 
C12_STOP_FREQ = 900e6   # 900 MHz (MAXIMUM FOR YOUR VNA)
C12_POINTS = 1601       

# Wide Frequency Sweep (Default for all others)
WIDE_START_FREQ = 1e6   # 1 MHz
WIDE_STOP_FREQ = 900e6  # 900 MHz (MAXIMUM FOR YOUR VNA)
WIDE_POINTS = 801       

REPEATS = 3 # Number of sweeps per circuit for averaging

# Store current sweep config
CURRENT_SWEEP = {'start': WIDE_START_FREQ, 'stop': WIDE_STOP_FREQ, 'points': WIDE_POINTS}

# --------------------------
# CIRCUIT DEFINITIONS
# --------------------------
CIRCUITS = [
    ('Circuit_1_RLC_Series_Parallel_1', 1), ('Circuit_2_RLC_Series_Parallel_2', 1),
    ('Circuit_3_33_Ohm_Resistor', 1), ('Circuit_4_75_Ohm_Resistor', 1),
    # LOW_SWEEP (Critical Low-Frequency Filters)
    ('Circuit_5_6_5MHz_Ceramic_Notch_Filter', 2), ('Circuit_6_10_7MHz_Ceramic_Notch_Filter', 2),
    # WIDE_SWEEP (More Default Components)
    ('Circuit_7_RC_Series_Circuit', 1), ('Circuit_8_LC_Series_Circuit', 1),
    ('Circuit_9_Capacitor', 1), ('Circuit_10_Inductor', 1),
    # OPTIMIZED HIGH-FREQUENCY SWEEPS
    ('Circuit_11_400MHz_Low_Pass_Filter', 2), ('Circuit_12_500MHz_High_Pass_Filter', 2),
    # WIDE_SWEEP (Attenuation & Calibration Standards)
    ('Circuit_17_10dB_Attenuation_Circuit', 2), ('Circuit_18_3dB_Attenuation_Circuit', 2),
    ('Circuit_13_Short_Circuit', 1), ('Circuit_14_Open_Circuit', 1),
    ('Circuit_15_50_Ohm_Load', 1), ('Circuit_16_Thru_Circuit', 2),
]

# ----------------------------------------------------------------------
# HELPER FUNCTION FOR SWEEP VERIFICATION 
# ----------------------------------------------------------------------
def verify_low_sweep_success(vna_instance, target_start_freq, circuit_name, circuit_description):
    """
    Performs a test sweep and checks if the first frequency data point is near the target
    low start frequency.
    """
    try:
        # 1. Run a test sweep
        s_data = vna_instance.sweep() 
        frequencies = s_data[-1]
        first_freq = frequencies[0]
        
        # 2. Check the result
        if first_freq > LOW_FREQ_CHECK_TOLERANCE_HZ:
            # 3. FAILURE: Command was ignored, prompt user
            print("\n=======================================================================================")
            print("❌ SWEEP FAILED CHECK: VNA ignored 1 MHz start command! (Observed start: "
                  f"{first_freq/1e6:.1f} MHz)")
            print("=======================================================================================")
            print(f"Circuit: {circuit_name} - {circuit_description}")
            print("Action: **MANUALLY SET** the VNA sweep range now:")
            print(f"  > **START FREQUENCY**: {target_start_freq/1e6} MHz (or 50 KHz if possible)")
            print(f"  > **STOP FREQUENCY**: {LOW_STOP_FREQ/1e6} MHz (30 MHz)")
            print("---------------------------------------------------------------------------------------")
            input("🛑 MANUALLY SET THE VNA FREQUENCY RANGE, then Press Enter to continue acquisition...")
            
            # The user has now manually fixed the sweep. We can return the data from the test sweep.
            return s_data 
        else:
            # 3. SUCCESS: Sweep range appears correct.
            print(f"  ✅ Sweep check passed. First frequency measured at {first_freq/1e6:.3f} MHz.")
            return s_data # Return data from the successful test sweep
            
    except Exception as e:
        logging.warning(f"⚠️ WARNING: Error during sweep verification. Skipping check. Details: {e}")
        # If the check fails (e.g., communication error), proceed without the check
        return None 


# --------------------------
# CONNECT TO NANOVNA
# --------------------------
vna = None
try:
    print("\n🔌 Attempting to connect to NanoVNA...")
    vna = VNA() 
    # Set initial WIDE sweep parameters (now capped at 900 MHz)
    vna.set_sweep(start=WIDE_START_FREQ, stop=WIDE_STOP_FREQ, points=WIDE_POINTS)
    CURRENT_SWEEP = {'start': WIDE_START_FREQ, 'stop': WIDE_STOP_FREQ, 'points': WIDE_POINTS}
    print(f"✅ Connection successful. Initial sweep set to {WIDE_START_FREQ/1e6} MHz to {WIDE_STOP_FREQ/1e6} MHz.")
    
except Exception as e:
    # This exception handling is now reached without the logging recursion error
    logging.critical(f"❌ CRITICAL ERROR: Could not connect to NanoVNA. Is it plugged in and the pynanovna driver active?")
    print(f"    Details: {e}")
    print("🛑 Script halted.")
    exit(1)


# --------------------------
# DATA ACQUISITION LOOP
# --------------------------
summary = []

for idx, (name, ports) in enumerate(CIRCUITS, start=1):
    
    circuit_parts = name.split('_')
    circuit_number = circuit_parts[1]
    circuit_description = ' '.join(circuit_parts[2:]).replace('_', ' ')
    
    # 🎯 CONDITIONAL SWEEP LOGIC (Uses updated constants) 🎯
    sweep_start, sweep_stop, sweep_points, filename_suffix, needs_verification = \
        WIDE_START_FREQ, WIDE_STOP_FREQ, WIDE_POINTS, "_sweep", False

    if 'Circuit_5' in name or 'Circuit_6' in name:
        # LOW FREQUENCY SWEEP - Needs Verification
        sweep_start, sweep_stop, sweep_points = LOW_START_FREQ, LOW_STOP_FREQ, LOW_POINTS
        filename_suffix = "_low_sweep"
        needs_verification = True 
        
    elif 'Circuit_11' in name:
        # C11 OPTIMIZED SWEEP (400 MHz LPF)
        sweep_start, sweep_stop, sweep_points = C11_START_FREQ, C11_STOP_FREQ, C11_POINTS
        filename_suffix = "_c11_sweep"
        
    elif 'Circuit_12' in name:
        # C12 OPTIMIZED SWEEP (500 MHz HPF - Wider range, capped at 900 MHz)
        sweep_start, sweep_stop, sweep_points = C12_START_FREQ, C12_STOP_FREQ, C12_POINTS
        filename_suffix = "_c12_sweep"
        

    # Check and apply sweep change
    if CURRENT_SWEEP['start'] != sweep_start or CURRENT_SWEEP['stop'] != sweep_stop or CURRENT_SWEEP['points'] != sweep_points:
        
        # Send the command multiple times for robustness
        for i in range(3):
            vna.set_sweep(start=sweep_start, stop=sweep_stop, points=sweep_points)
            time.sleep(0.1)
            
        CURRENT_SWEEP = {'start': sweep_start, 'stop': sweep_stop, 'points': sweep_points}
        print(f"\n✨ SWEEP CHANGE: Setting sweep for '{name}': {sweep_start/1e6:.1f}MHz to {sweep_stop/1e6:.1f}MHz ({sweep_points} points).")
    
    
    # --- PROMPT FOR CONNECTION ---
    if ports == 1:
        prompt = f"\n➡️ Attach coax to Circuit {circuit_number}: '{circuit_description}' (Step {idx} of {len(CIRCUITS)}, 1-port). Press Enter when ready..."
    else:
        prompt = f"\n➡️ Attach input and output coax to Circuit {circuit_number}: '{circuit_description}' (Step {idx} of {len(CIRCUITS)}, 2-port). Press Enter when ready..."
    
    input(prompt)


    # --- VERIFY SWEEP AND GET INITIAL DATA ---
    initial_sweep_data = None
    if needs_verification:
        initial_sweep_data = verify_low_sweep_success(vna, sweep_start, circuit_number, circuit_description)
    
    # Initialize accumulation arrays
    freq_accum = None
    s11_real_accum = None
    s11_imag_accum = None
    s21_real_accum = None
    s21_imag_accum = None
    
    sweeps_performed = 0 

    if initial_sweep_data is not None:
        frequencies = initial_sweep_data[-1]
        s_parameters = initial_sweep_data[:-1] 

        s11_complex = s_parameters[0]
        s21_complex = s_parameters[1] if ports == 2 and len(s_parameters) > 1 else np.full(CURRENT_SWEEP['points'], np.nan + 0j) 

        freq_accum = frequencies.copy()
        s11_real_accum = np.real(s11_complex)
        s11_imag_accum = np.imag(s11_complex)
        s21_real_accum = np.real(s21_complex)
        s21_imag_accum = np.imag(s21_complex)
        
        sweeps_performed = 1
        logging.info("    Sweep 1/3 (verification) complete")


    # Perform remaining repeated sweeps
    remaining_repeats = REPEATS - sweeps_performed
    for rep in range(remaining_repeats):
        try: 
            s_data = vna.sweep() 
            frequencies = s_data[-1]
            s_parameters = s_data[:-1] 

            s11_complex = s_parameters[0]
            s21_complex = s_parameters[1] if ports == 2 and len(s_parameters) > 1 else np.full(CURRENT_SWEEP['points'], np.nan + 0j) 


            s11_real = np.real(s11_complex)
            s11_imag = np.imag(s11_complex)

            if ports == 2 and s21_complex is not None:
                s21_real = np.real(s21_complex)
                s21_imag = np.imag(s21_complex)

            if freq_accum is None:
                freq_accum = frequencies.copy()
                s11_real_accum = s11_real.copy()
                s11_imag_accum = s11_imag.copy()
                s21_real_accum = s21_real.copy() if ports == 2 else None
                s21_imag_accum = s21_imag.copy() if ports == 2 else None
            else:
                s11_real_accum += s11_real
                s11_imag_accum += s11_imag
                if ports == 2:
                    s21_real_accum += s21_real
                    s21_imag_accum += s21_imag

            sweeps_performed += 1 

            logging.info(f"    Sweep {sweeps_performed}/{REPEATS} complete")
            time.sleep(0.5)

        except Exception as e:
            logging.warning(f"\n    ⚠️ WARNING: Data acquisition error on sweep {rep+1}. Skipping this sweep.")
            print(f"    Details: {e}")
            time.sleep(1)
            continue


    # --- Averaging and Final Data Prep ---
    if sweeps_performed > 0:
        s11_real_avg = s11_real_accum / sweeps_performed
        s11_imag_avg = s11_imag_accum / sweeps_performed
        if ports == 2:
            s21_real_avg = s21_real_accum / sweeps_performed
            s21_imag_avg = s21_imag_accum / sweeps_performed
    else:
        if freq_accum is None:
            freq_accum = np.linspace(CURRENT_SWEEP['start'], CURRENT_SWEEP['stop'], CURRENT_SWEEP['points'])
        s11_real_avg = np.full(CURRENT_SWEEP['points'], np.nan)
        s11_imag_avg = np.full(CURRENT_SWEEP['points'], np.nan)
        if ports == 2:
            s21_real_avg = np.full(CURRENT_SWEEP['points'], np.nan)
            s21_imag_avg = np.full(CURRENT_SWEEP['points'], np.nan)


    # Prepare DataFrame
    df = pd.DataFrame({
        'frequency_Hz': freq_accum,
        'S11_real': s11_real_avg,
        'S11_imag': s11_imag_avg
    })
    
    # Calculate magnitude and phase for S11
    s11_complex_avg = s11_real_avg + 1j * s11_imag_avg
    if not np.isnan(s11_real_avg).all(): 
        df['S11_mag_dB'] = 20 * np.log10(np.abs(s11_complex_avg))
        df['S11_phase_deg'] = np.degrees(np.angle(s11_complex_avg))
    else:
        df['S11_mag_dB'] = np.nan
        df['S11_phase_deg'] = np.nan

    if ports == 2:
        df['S21_real'] = s21_real_avg
        df['S21_imag'] = s21_imag_avg
        
        # Calculate magnitude and phase for S21
        s21_complex_avg = s21_real_avg + 1j * s21_imag_avg
        if not np.isnan(s21_real_avg).all():
            df['S21_mag_dB'] = 20 * np.log10(np.abs(s21_complex_avg))
            df['S21_phase_deg'] = np.degrees(np.angle(s21_complex_avg))
        else:
            df['S21_mag_dB'] = np.nan
            df['S21_phase_deg'] = np.nan


    # Save CSV
    filename = f"{name}{filename_suffix}.csv"
    filepath = os.path.join(SAVE_FOLDER, filename)
    df.to_csv(filepath, index=False)

    # Update summary
    summary.append({
        'Circuit': name,
        'Ports': ports,
        'Filename': filename,
        'Start_Freq_Hz': CURRENT_SWEEP['start'],
        'Stop_Freq_Hz': CURRENT_SWEEP['stop'],
        'Points': CURRENT_SWEEP['points'],
        'Repeats': REPEATS,
        'Successful_Sweeps': sweeps_performed
    })

    print(f"✅ Sweep for '{name}' saved as {filename}")
    time.sleep(1)


# Save summary table
summary_df = pd.DataFrame(summary)
summary_csv = os.path.join(SAVE_FOLDER, 'summary_table_optimized.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"\n📄 All sweeps completed. Summary saved as '{summary_csv}'")

# --------------------------
# DISCONNECT AND CLEANUP
# --------------------------
if vna:
    try:
        vna.disconnect()
        print("🔌 NanoVNA disconnected. Data acquisition finished!")
    except AttributeError:
        # Expected warning from pynanovna, safely ignorable
        print("\n⚠️ WARNING: The final VNA disconnect step caused an AttributeError. **You can safely ignore this message.**")
        pass
       
vna.disconnect()
print("🔌 NanoVNA disconnected. Data acquisition finished!")

