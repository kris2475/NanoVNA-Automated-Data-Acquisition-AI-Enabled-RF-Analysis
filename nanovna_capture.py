import pynanovna
import numpy as np
import time
import csv

# ====================================================================
# USER CONFIGURABLE DEFAULTS
# These can be adjusted directly in the script if desired.
# ====================================================================
DEFAULT_START_FREQ = 100e6   # 100 MHz
DEFAULT_STOP_FREQ = 1000e6   # 1 GHz
DEFAULT_POINTS = 101         # Number of data points per sweep

# ====================================================================
# HELPER FUNCTION
# Converts complex S-parameters into Magnitude (dB) and Phase (degrees)
# ====================================================================
def s_to_mag_phase(s_complex):
    mag_db = 20 * np.log10(np.abs(s_complex))
    phase_deg = np.angle(s_complex, deg=True)
    return mag_db, phase_deg

# ====================================================================
# MAIN SCRIPT
# ====================================================================
print("\n📡 NanoVNA Multi-Sweep Automation Script")
print("=========================================")
print("This script performs multiple frequency sweeps on a NanoVNA.")
print("Each sweep corresponds to one cable or DUT (Device Under Test).")
print("Ensure your NanoVNA is calibrated at the reference plane before proceeding.\n")

try:
    # --- User input for sweep configuration ---
    num_sweeps = int(input("Enter the number of sweeps (e.g. 2 for two cables): "))
    start_freq = float(input(f"Enter start frequency in MHz [default {DEFAULT_START_FREQ/1e6:.1f}]: ") or DEFAULT_START_FREQ/1e6) * 1e6
    stop_freq = float(input(f"Enter stop frequency in MHz [default {DEFAULT_STOP_FREQ/1e6:.1f}]: ") or DEFAULT_STOP_FREQ/1e6) * 1e6
    points = int(input(f"Enter number of points per sweep [default {DEFAULT_POINTS}]: ") or DEFAULT_POINTS)

    # --- Prepare output file ---
    output_file = f"nanovna_batch_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"\nOutput file: {output_file}\n")

    # --- Connect to NanoVNA ---
    print("Connecting to NanoVNA...")
    vna = pynanovna.VNA()
    print("✅ Connection successful.")

    print(f"Setting sweep parameters: {start_freq/1e6:.1f} MHz to {stop_freq/1e6:.1f} MHz, {points} points.")
    vna.set_sweep(start_freq, stop_freq, points)
    time.sleep(0.5)

    # --- Create CSV header ---
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Sample_ID", "Frequency (Hz)", "S11 Mag (dB)", "S11 Phase (deg)",
                  "S21 Mag (dB)", "S21 Phase (deg)",
                  "S11 Real", "S11 Imag", "S21 Real", "S21 Imag"]
        writer.writerow(header)

    # --- Main sweep loop ---
    for sweep_id in range(1, num_sweeps + 1):
        print(f"\n--- Sweep {sweep_id} of {num_sweeps} ---")
        input(f"Attach Sample ID {sweep_id} to ports 1 and 2, then press ENTER to continue...")

        print("Sweeping...")
        s11_complex, s21_complex, frequencies = vna.sweep()

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(points):
                s11_mag_db, s11_phase_deg = s_to_mag_phase(s11_complex[i])
                s21_mag_db, s21_phase_deg = s_to_mag_phase(s21_complex[i])
                row = [
                    sweep_id,
                    frequencies[i],
                    s11_mag_db, s11_phase_deg,
                    s21_mag_db, s21_phase_deg,
                    s11_complex[i].real, s11_complex[i].imag,
                    s21_complex[i].real, s21_complex[i].imag
                ]
                writer.writerow(row)

        print(f"✅ Data for Sample ID {sweep_id} saved.")

    print(f"\n✨ All {num_sweeps} sweeps complete! Data saved to {output_file}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Ensure the NanoVNA is connected, powered on, and the pynanovna library is installed.")

finally:
    try:
        if 'vna' in locals() and vna.connected:
            vna.disconnect()
            print("Disconnected from NanoVNA.")
    except Exception:
        pass
