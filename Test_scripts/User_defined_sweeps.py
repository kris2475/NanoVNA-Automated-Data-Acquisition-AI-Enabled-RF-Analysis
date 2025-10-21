import pandas as pd
import numpy as np
import sys
import csv
from scipy.fft import ifft

# --- Constants ---
C = 299792458.0  # Speed of light (m/s)
SYSTEM_IMPEDANCE_OHMS = 50.0
# Global variables to be loaded from CSV metadata
PHYSICAL_LENGTH_METERS = 0.0
VELOCITY_FACTOR = 0.0

# --- Helper Function: Read Metadata ---
def read_metadata_from_csv(input_file):
    """
    Reads commented metadata (lines starting with '#') from the CSV header
    to extract cable properties and sweep settings.
    Returns: (dict of metadata, number of skip rows)
    """
    metadata = {}
    skip_rows = 0
    
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].startswith('#'):
                # Stop when we hit the first non-comment or empty line (the header)
                break
            
            # Line is a comment, parse the key: value
            line = row[0].lstrip('#').strip()
            if ':' in line:
                key, value = [part.strip() for part in line.split(':', 1)]
                
                # Clean up keys for consistent dictionary access
                key = key.replace(' ', '_').replace('(', '').replace(')', '')
                
                try:
                    # Attempt to convert to float/int if possible
                    if '.' in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
            
            skip_rows += 1
            
    return metadata, skip_rows

# --- Helper Function: Complex S11 to TDR Trace (IFFT) ---
def calculate_tdr_trace(s11_complex, frequencies):
    """
    Converts frequency-domain S11 data to a time-domain step response trace using IFFT.
    """
    N_orig = len(s11_complex)
    
    # 4x zero padding for higher resolution on the time axis
    N_fft = 2**np.ceil(np.log2(N_orig)).astype(int) * 4
    
    # 1. Zero-pad the S11 data for IFFT
    s11_padded = np.zeros(N_fft, dtype=complex)
    s11_padded[:N_orig] = s11_complex
    
    # 2. Perform the IFFT to get the impulse response
    tdr_impulse = ifft(s11_padded)

    # 3. Calculate the time axis (Critical for correct scaling)
    f_start = frequencies[0]
    f_stop = frequencies[-1]
    bandwidth = f_stop - f_start
    
    # The time span must be corrected by a factor of 2.0 because VNA data is
    # single-sided (band-limited) data, not baseband data.
    time_span = (N_orig / bandwidth) / 2.0 # Applied final factor of 2 correction
    time_step = time_span / N_fft
    time_axis = np.arange(N_fft) * time_step
    
    # 4. Convert impulse response to step response via cumulative sum (integration)
    # This is essential when the sweep starts above 0 Hz.
    tdr_step_response = np.cumsum(np.real(tdr_impulse))
    
    # Normalize the trace to start at zero
    tdr_step_response -= tdr_step_response[0]
    
    return tdr_step_response, time_axis

# --- Main Analysis Function ---
def analyse_cable_data(input_file):
    global PHYSICAL_LENGTH_METERS, VELOCITY_FACTOR

    print(f"Analysing file: {input_file}")

    # 1. Read metadata from the header
    metadata, skip_rows = read_metadata_from_csv(input_file)
    
    try:
        PHYSICAL_LENGTH_METERS = metadata['CABLE_LENGTH_M']
        VELOCITY_FACTOR = metadata['VELOCITY_FACTOR']
        print(f"--- Metadata Loaded: Length={PHYSICAL_LENGTH_METERS:.3f}m, VF={VELOCITY_FACTOR:.3f} ---")
    except KeyError:
        print("\nFATAL ERROR: CABLE_LENGTH_M or VELOCITY_FACTOR was missing in the CSV header.")
        print("Please ensure you used the latest data collection script and provided inputs.")
        return

    # 2. Read the main data, skipping the metadata lines
    try:
        # Use skiprows to skip the commented metadata lines
        df = pd.read_csv(input_file, skiprows=skip_rows)
    except pd.errors.ParserError as e:
        print(f"FATAL ERROR: Failed to parse data rows after skipping {skip_rows} rows.")
        print(f"Details: {e}")
        return
    
    summary_data = []

    # 3. Group by Sample_ID and process each sweep
    for sample_id, group in df.groupby('Sample_ID'):
        print(f"Processing Sample ID {sample_id}...")
        
        # S11 data is needed as a complex number array
        s11_complex = group['S11 Real'] + 1j * group['S11 Imag']
        frequencies = group['Frequency (Hz)'].values

        # Calculate TDR trace and time axis
        tdr_trace, time_axis = calculate_tdr_trace(s11_complex.values, frequencies)
        
        # Find the peak corresponding to the cable end (impedance discontinuity)
        # We search the first half of the trace to avoid wrap-around effects from the IFFT
        half_point = len(tdr_trace) // 2
        peak_index = np.argmax(np.abs(tdr_trace[:half_point])) # Use absolute value for robust detection
        
        # Calculate derived metrics
        round_trip_time = time_axis[peak_index]
        electrical_length = (C * round_trip_time) / 2.0
        
        # The calculated VF using the user's *input* physical length
        velocity_factor_calc = electrical_length / PHYSICAL_LENGTH_METERS if PHYSICAL_LENGTH_METERS > 0 else 0.0

        # Output warning if VF is non-physical
        if not (0.0 < velocity_factor_calc <= 1.0):
            print(f"WARNING: Sample {sample_id} calculated VF ({velocity_factor_calc:.4f}) is outside the physical range [0, 1].")
            print("This usually indicates: 1) NO SOL CALIBRATION was performed, or 2) the input cable length is incorrect.")

        summary_data.append({
            'Sample_ID': sample_id,
            'Physical_Length_Input (m)': PHYSICAL_LENGTH_METERS,
            'Calculated_Electrical_Length (m)': electrical_length,
            'Calculated_Velocity_Factor (VF)': velocity_factor_calc
        })

    # 4. Save Summary
    output_df = pd.DataFrame(summary_data)
    output_filename = 'cable_tdr_summary.csv'
    output_df.to_csv(output_filename, index=False)

    print(f"\n✨ Analysis Complete! Summary saved to {output_filename}")
    print(f"--- VF Range: {output_df['Calculated_Velocity_Factor (VF)'].min():.4f} to {output_df['Calculated_Velocity_Factor (VF)'].max():.4f} ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cable_analysis.py <input_csv_file>")
    else:
        analyse_cable_data(sys.argv[1])


