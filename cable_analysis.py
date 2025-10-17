import pandas as pd
import numpy as np
import sys
from scipy.fft import ifft

# --- Constants ---
C = 299792458.0  # Speed of light (m/s)
PHYSICAL_LENGTH_METERS = 1.0  # Adjust for your measured cable length
SYSTEM_IMPEDANCE_OHMS = 50.0

# --- Helper Function: Complex S11 to TDR Trace (IFFT) ---
def calculate_tdr_trace(s11_complex, frequencies):
    """
    Converts frequency-domain S11 data to a time-domain reflection trace using IFFT.
    """
    N_orig = len(s11_complex)
    N_fft = 2**np.ceil(np.log2(N_orig)).astype(int) * 4  # Use 4× resolution
    s11_padded = np.zeros(N_fft, dtype=complex)
    s11_padded[:N_orig] = s11_complex
    tdr_trace = ifft(s11_padded)
    f_max = frequencies[-1]
    time_step = 1.0 / f_max / N_fft
    time_axis = np.arange(N_fft) * time_step
    return np.real(tdr_trace), time_axis

# --- Main Analysis Function ---
def analyse_cable_data(input_file):
    print(f"Analysing file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return

    summary_data = []

    for sample_id, group in df.groupby('Sample_ID'):
        print(f"Processing Sample ID {sample_id}...")
        group = group.sort_values(by='Frequency (Hz)')
        s11_complex = group['S11 Real'] + 1j * group['S11 Imag']
        frequencies = group['Frequency (Hz)'].values

        tdr_trace, time_axis = calculate_tdr_trace(s11_complex.values, frequencies)
        half_point = len(tdr_trace) // 2
        peak_index = np.argmax(tdr_trace[:half_point])
        round_trip_time = time_axis[peak_index]
        electrical_length = (C * round_trip_time) / 2.0
        velocity_factor = electrical_length / PHYSICAL_LENGTH_METERS if PHYSICAL_LENGTH_METERS > 0 else 0.0

        summary_data.append({
            'Sample_ID': sample_id,
            'Physical_Length_Input (m)': PHYSICAL_LENGTH_METERS,
            'Calculated_Electrical_Length (m)': electrical_length,
            'Calculated_Velocity_Factor (VF)': velocity_factor
        })

    output_df = pd.DataFrame(summary_data)
    output_filename = 'cable_tdr_summary.csv'
    output_df.to_csv(output_filename, index=False)

    print(f"\n✨ Analysis Complete! Summary saved to {output_filename}")
    print(f"--- VF Range: {output_df['Calculated_Velocity_Factor (VF)'].min():.4f} to {output_df['Calculated_Velocity_Factor (VF)'].max():.4f} ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cable_analysis.py <input_csv_file>")
        print("Example: python cable_analysis.py nanovna_batch_sweep_20251017_180100.csv")
    else:
        if PHYSICAL_LENGTH_METERS <= 0:
            print(f"⚠️ WARNING: PHYSICAL_LENGTH_METERS = {PHYSICAL_LENGTH_METERS}. Please set the correct cable length.")
        analyse_cable_data(sys.argv[1])

