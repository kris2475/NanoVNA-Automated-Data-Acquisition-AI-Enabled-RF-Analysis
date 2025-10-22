import subprocess
import sys
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re 

# ---------------------------
# 1️⃣ Ensure dependencies and pynanovna are present (UNCHANGED)
# ---------------------------
required_packages = ["pyserial", "numpy", "pandas", "matplotlib"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

try:
    import pynanovna
    from pynanovna import VNA
except ImportError:
    print("Installing pynanovna from GitHub...")
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "git+https://github.com/PICC-Group/pynanovna.git"], check=True)
    import pynanovna
    from pynanovna import VNA


# ----------------------------------------------------------------------
# 2️⃣ CUSTOM LOGGING FILTER IMPLEMENTATION (UNCHANGED)
# ----------------------------------------------------------------------

class CalibrationFilter(logging.Filter):
    """ Suppresses specific, non-critical, repetitive log messages. """
    def __init__(self, filter_phrases=None):
        super().__init__()
        self.filter_phrases = filter_phrases or [
            "No calibration has been applied",
            "calibration not valid",
            "recommended to re-calibrate",
        ]

    def filter(self, record):
        message = record.getMessage()
        for phrase in self.filter_phrases:
            if phrase.lower() in message.lower():
                return False 
        return True 

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) 
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

calibration_filter = CalibrationFilter()
root_logger.addFilter(calibration_filter)

print("--- Log Messages Filtered ---")
print("✅ Repetitive 'calibration not valid' messages are now suppressed reliably.")


# ---------------------------
# 3️⃣ Configuration & Resumption Logic (UNCHANGED)
# ---------------------------
MAX_POINTS = 1024
SWEEP_AVERAGE_COUNT = 3 
SWEEPS_PER_BATCH = 100 # Run 100 new sweeps per circuit
SAVE_FOLDER = "sweeps"
os.makedirs(SAVE_FOLDER, exist_ok=True)


def get_next_sweep_number(circuit_name, data_dir):
    """ Finds the highest existing sweep number for safe resumption. """
    pattern = re.compile(
        rf"{re.escape(circuit_name)}_Sweep_(\d+)\.csv$", 
        re.IGNORECASE
    )
    max_sweep_num = 0
    for filename in os.listdir(data_dir):
        match = pattern.match(filename)
        if match:
            current_sweep_num = int(match.group(1))
            if current_sweep_num > max_sweep_num:
                max_sweep_num = current_sweep_num
                
    return max_sweep_num + 1

def generate_filename(circuit_name, sweep_num):
    """ Generates the full, correctly padded file path. """
    padded_sweep_num = f"{sweep_num:03d}"
    filename = (
        f"{circuit_name}_Sweep_{padded_sweep_num}.csv"
    )
    return os.path.join(SAVE_FOLDER, filename)


# ---------------------------
# 4️⃣ Setup and Full SOLT Calibration (UNCHANGED)
# ---------------------------
def set_and_calibrate(vna):
    START_FREQ = 1_000_000     
    STOP_FREQ = 900_000_000
    
    print(f"✨ Setting calibration sweep to {START_FREQ/1e6}MHz - {STOP_FREQ/1e6}MHz ({MAX_POINTS} points)...")
    vna.set_sweep(START_FREQ, STOP_FREQ, MAX_POINTS) 
    print("✅ Sweep range set.")
    
    print("\n✨ Starting FULL two-port SOLT calibration process...")
    
    # 1. SHORT standard
    input("🛑 CONNECT SHORT STANDARD to Port 0, then press Enter to measure...")
    vna.calibration_step('short') 
    print("✅ Short standard measured.")
    
    # 2. OPEN standard
    input("🛑 CONNECT OPEN STANDARD to Port 0, then press Enter to measure...")
    vna.calibration_step('open')
    print("✅ Open standard measured.")
    
    # 3. LOAD standard
    input("🛑 CONNECT LOAD STANDARD (50 Ohm) to Port 0, then press Enter to measure...")
    vna.calibration_step('load')
    print("✅ Load standard measured.")
    
    # 4. THROUGH standard (REQUIRED for S21 measurements)
    input("🛑 CONNECT CABLE BETWEEN Port 0 AND Port 1 (THRU), then press Enter to measure...")
    vna.calibration_step('through')
    print("✅ Through standard measured.")

    # Calculate and apply calibration
    vna.calibrate() 
    
    print("✨ Calculating and applying SOLT calibration corrections...")
    print("✅ Full SOLT Calibration complete and applied.")

# ---------------------------
# 5️⃣ Define circuits (UNCHANGED)
# ---------------------------
circuits = [
    ('Circuit_1_RLC_Series_Parallel_1', 1),
    ('Circuit_2_RLC_Series_Parallel_2', 1),
    ('Circuit_3_33_Ohm_Resistor', 1),
    ('Circuit_4_75_Ohm_Resistor', 1),
    ('Circuit_5_6_5MHz_Ceramic_Notch_Filter', 2),
    ('Circuit_6_10_7MHz_Ceramic_Notch_Filter', 2),
    ('Circuit_7_RC_Series_Circuit', 1),
    ('Circuit_8_LC_Series_Circuit', 1),
    ('Circuit_9_Capacitor', 1),
    ('Circuit_10_Inductor', 1),
    ('Circuit_11_400MHz_Low_Pass_Filter', 2),
    ('Circuit_12_500MHz_High_Pass_Filter', 2),
    ('Circuit_13_Short_Circuit', 1),
    ('Circuit_14_Open_Circuit', 1),
    ('Circuit_15_50_Ohm_Load', 1),
    ('Circuit_16_Thru_Circuit', 2),
    ('Circuit_17_10dB_Attenuation_Circuit', 2),
    ('Circuit_18_3dB_Attenuation_Circuit', 2),
]

# ---------------------------
# 6️⃣ Acquire sweep (UNCHANGED)
# ---------------------------
def acquire_data(vna):
    all_s11_sweeps = []
    all_s21_sweeps = []
    freqs = None
    
    for i in range(SWEEP_AVERAGE_COUNT): # Uses 3 for averaging
        sweep_data = vna.sweep() 
        
        if isinstance(sweep_data, tuple) and len(sweep_data) == 3:
            freqs, s11_params, s21_params = sweep_data 
            all_s11_sweeps.append(s11_params)
            all_s21_sweeps.append(s21_params)
        else:
            raise ValueError(f"Unexpected sweep return format. Expected 3 elements, got {len(sweep_data) if isinstance(sweep_data, tuple) else 'non-tuple'}")
            
        logging.info(f"Sweep {i+1}/{SWEEP_AVERAGE_COUNT} acquired (Averaging)")
        time.sleep(0.5)
        
    avg_s11 = np.mean(all_s11_sweeps, axis=0)
    avg_s21 = np.mean(all_s21_sweeps, axis=0)
    
    return freqs, avg_s11, avg_s21

# ---------------------------
# 7️⃣ Main routine (LOOPS SWAPPED for per-circuit batch)
# ---------------------------
def main():
    global MAX_POINTS, SWEEPS_PER_BATCH
    
    try:
        logging.info("\n🔌 Attempting to connect to NanoVNA...")
        vna = VNA()
        logging.info("✅ VNA successfully initialized.")
    except Exception as e:
        logging.critical(f"❌ ERROR: Failed to initialize VNA. Is it connected and powered on? Error: {e}")
        sys.exit(1)

    # 1. MANDATORY CALIBRATION PROMPT (Once per script run)
    set_and_calibrate(vna) 

    # 2. RESUMPTION CHECK (Find starting sweep number for all circuits)
    next_sweep_starts = {}
    print("\n--------------------------------------------------------------")
    print("Scanning existing sweeps to determine safe starting numbers...")
    for name, _ in circuits:
        start_num = get_next_sweep_number(name, SAVE_FOLDER)
        next_sweep_starts[name] = start_num
        print(f"  {name}: Resuming at Sweep {start_num}")
    print("--------------------------------------------------------------\n")

    # --- 3. SWEEP AND SAVE LOOP (The Per-Circuit Batch) ---
    # Outer loop: Iterate through all 18 circuits (This is now the pause point)
    for name, ports in circuits:
        
        # Calculate the next batch's absolute start and end numbers
        start_num = next_sweep_starts[name]
        end_num = start_num + SWEEPS_PER_BATCH - 1
        
        print(f"\n==============================================================")
        print(f"  PREPARING CIRCUIT: {name} ({ports}-Port)")
        print(f"  Targeting {SWEEPS_PER_BATCH} new sweeps (Files {start_num} to {end_num}).")
        print(f"==============================================================")

        # --- Circuit Connection Prompt (Waits for user) ---
        if ports == 1:
            prompt = f"🛑 CONNECT {name} to CH0, then press Enter to start the {SWEEPS_PER_BATCH} automated sweeps..."
        else: # ports == 2
            prompt = f"🛑 CONNECT {name} to CH0 and CH1, then press Enter to start the {SWEEPS_PER_BATCH} automated sweeps..."
        input(prompt)
        # --- End Prompt ---
        
        
        # --- Custom Sweep Setting (Once per circuit) ---
        # Default to the wide, calibrated range (1MHz to 900MHz)
        START_FREQ = 1_000_000     
        STOP_FREQ = 900_000_000
        
        # Custom ranges for specific circuits
        if "6_5MHz_Ceramic_Notch_Filter" in name:
            START_FREQ = 1_000_000
            STOP_FREQ = 15_000_000 
        elif "10_7MHz_Ceramic_Notch_Filter" in name:
            START_FREQ = 5_000_000
            STOP_FREQ = 20_000_000 
        elif "400MHz_Low_Pass_Filter" in name:
            START_FREQ = 100_000_000
            STOP_FREQ = 600_000_000 
        elif "500MHz_High_Pass_Filter" in name:
            START_FREQ = 300_000_000
            STOP_FREQ = 800_000_000 

        # Apply the custom (or default) sweep for the current circuit
        logging.info(f"Setting sweep range to {START_FREQ/1e6}MHz - {STOP_FREQ/1e6}MHz.")
        vna.set_sweep(START_FREQ, STOP_FREQ, MAX_POINTS)
        # --- End Custom Sweep Setting ---

        
        # Inner loop: Run the desired number of new sweeps (e.g., 100 times)
        # THIS LOOP DOES NOT PAUSE
        for i in range(SWEEPS_PER_BATCH):
            
            absolute_sweep_num = start_num + i
            file_path = generate_filename(name, absolute_sweep_num) 
            
            try:
                # ACQUIRE DATA (3x average)
                freqs, s11_avg, s21_avg = acquire_data(vna) 
            except Exception as e:
                logging.error(f"❌ Error acquiring data for {name} (Sweep {absolute_sweep_num}): {e}")
                continue

            # --- Save Data ---
            df = pd.DataFrame({
                'frequency_Hz': freqs,
                'S11_real': np.real(s11_avg),
                'S11_imag': np.imag(s11_avg),
                'S11_mag_dB': 20 * np.log10(np.abs(s11_avg)),
                'S11_phase_deg': np.degrees(np.angle(s11_avg))
            })
            
            if ports == 2:
                df['S21_real'] = np.real(s21_avg)
                df['S21_imag'] = np.imag(s21_avg)
                df['S21_mag_dB'] = 20 * np.log10(np.abs(s21_avg))
                df['S21_phase_deg'] = np.degrees(np.angle(s21_avg))
                
            df.to_csv(file_path, index=False)
            print(f"  -> Saved Sweep {absolute_sweep_num}/{end_num} as {os.path.basename(file_path)}")
            # --- End Save ---

        print(f"\n✅ SUCCESSFULLY COMPLETED {SWEEPS_PER_BATCH} sweeps for {name}. READY FOR NEXT CIRCUIT.")

    print(f"\n🎉 Successfully captured all requested sweeps for all {len(circuits)} circuits!")
    
    try:
        vna.disconnect()
        print("🔌 NanoVNA disconnected.")
    except Exception as e:
        logging.warning(f"⚠️ Could not cleanly disconnect VNA: {e}")


if __name__ == "__main__":
    main()
