import pynanovna
import numpy as np
import time
import csv

# ====================================================================
# SCRIPT CONFIGURATION
# Adjust these values to set the frequency range, resolution, and sweep count.
# ====================================================================
NUM_SWEEPS = 10     # Total number of sweeps/samples to collect.
START_FREQ = 100e6  # Start Frequency in Hz (e.g., 100 MHz)
STOP_FREQ = 1000e6  # Stop Frequency in Hz (e.g., 1 GHz)
POINTS = 101        # Number of data points per sweep (resolution)

# Output file name includes a unique timestamp for the entire batch
OUTPUT_FILE = f"nanovna_batch_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# ====================================================================
# HELPER FUNCTION
# Converts complex S-parameter (Real + Imaginary) into Magnitude (dB) and Phase (degrees).
# ====================================================================
def s_to_mag_phase(s_complex):
    # Magnitude (dB) = 20 * log10(|S|)
    mag_db = 20 * np.log10(np.abs(s_complex))
    # Phase (degrees)
    phase_deg = np.angle(s_complex, deg=True)
    return mag_db, phase_deg

# ====================================================================
# MAIN AUTOMATION LOGIC
# Connects, sets sweep, loops through N measurements, and saves the file.
# ====================================================================
print("Starting NanoVNA Multi-Sweep Automation...")
vna = None # Initialize vna variable

try:
    # 1. Initialize and Connect to the NanoVNA
    vna = pynanovna.VNA()
    print("Connection successful.")

    # 2. Set Sweep Parameters once for the entire batch
    print(f"Setting sweep: {START_FREQ/1e6:.1f} MHz to {STOP_FREQ/1e6:.1f} MHz, {POINTS} points.")
    vna.set_sweep(START_FREQ, STOP_FREQ, POINTS)
    time.sleep(0.5)

    # Prepare CSV file: Open it and write the header row only once
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # NOTE: Added 'Sample_ID' column for ML labeling
        header = ["Sample_ID", "Frequency (Hz)", "S11 Mag (dB)", "S11 Phase (deg)", 
                  "S21 Mag (dB)", "S21 Phase (deg)",
                  "S11 Real", "S11 Imag", "S21 Real", "S21 Imag"]
        writer.writerow(header)

    # 3. Loop through the desired number of sweeps
    for sweep_id in range(1, NUM_SWEEPS + 1):
        print(f"\n--- Ready for Sweep {sweep_id} of {NUM_SWEEPS} ---")
        
        # --- PAUSE FOR MANUAL INTERACTION ---
        # This is CRITICAL: Wait for the user to change the physical sample/DUT.
        input(f"ACTION REQUIRED: Attach Sample ID {sweep_id} to ports 1 and 2. PRESS ENTER to start the sweep.")
        
        # 4. Perform the Sweep and Capture Data
        print(f"Sweeping Sample ID {sweep_id}...")
        s11_complex, s21_complex, frequencies = vna.sweep()

        # 5. Append Data to the CSV file
        with open(OUTPUT_FILE, 'a', newline='') as f: # Use 'a' for append mode
            writer = csv.writer(f)
            
            for i in range(POINTS):
                f_hz = frequencies[i]
                
                # Convert complex S-parameters
                s11_mag_db, s11_phase_deg = s_to_mag_phase(s11_complex[i])
                s21_mag_db, s21_phase_deg = s_to_mag_phase(s21_complex[i])
                
                # Construct the row, starting with the Sample_ID
                row = [sweep_id, f_hz, s11_mag_db, s11_phase_deg, s21_mag_db, s21_phase_deg,
                       s11_complex[i].real, s11_complex[i].imag,
                       s21_complex[i].real, s21_complex[i].imag]
                writer.writerow(row)
            
        print(f"✅ Data for Sample ID {sweep_id} appended successfully.")
        
    print(f"\n✨ MULTI-SWEEP BATCH COMPLETE! All data saved to {OUTPUT_FILE}")

except Exception as e:
    # Catch any connection or communication errors
    print(f"❌ An error occurred. Ensure the NanoVNA is connected, powered on, and the pynanovna library is installed.")
    print(f"Error details: {e}")

finally:
    # Clean up the connection if it was established
    if vna and vna.connected:
        print("Disconnected from NanoVNA.")
