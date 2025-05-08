import os
import glob
import pandas as pd
from scipy.signal import savgol_filter

# ----------------------------
# Configuration
# ----------------------------

# Define input and output directories as pairs
input_output_dirs = {
    '/users/eb2019/scratch/swimmers/ES': '/users/eb2019/scratch/sg_data/11_2/ES',
    '/users/eb2019/scratch/swimmers/WTS': '/users/eb2019/scratch/sg_data/11_2/WTS',
    '/users/eb2019/scratch/swimmers/EDS': '/users/eb2019/scratch/sg_data/11_2/EDS'
}

# Savitzky–Golay filter parameters
window_length = 11  # Must be odd.
polyorder = 2  # Polynomial order for the filter.

# ----------------------------
# Processing Files for Each Directory Pair
# ----------------------------

for input_directory, output_directory in input_output_dirs.items():
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get a list of all .txt files in the input directory.
    txt_files = glob.glob(os.path.join(input_directory, '*.txt'))

    for file_path in txt_files:
        try:
            # Check the first line to determine if it is a header
            with open(file_path, 'r') as file:
                first_line = file.readline().strip()  # Read first line
                second_line = file.readline().strip()  # Read second line for structure check

            # Determine if the first line is a header (contains text rather than numeric data)
            skip_rows = 1 if any(c.isalpha() for c in first_line) else 0

            # Read the file with the correct skip_rows value
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=skip_rows)

            # Assign column names if they were removed
            df.columns = ['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth']

            # Apply the Savitzky–Golay filter
            df['x_smooth_sg'] = savgol_filter(df['x'].values, window_length, polyorder)
            df['y_smooth_sg'] = savgol_filter(df['y'].values, window_length, polyorder)
            df['z_smooth_sg'] = savgol_filter(df['z'].values, window_length, polyorder)

            # Construct an output file name
            base_name = os.path.basename(file_path)
            output_file = os.path.join(output_directory, base_name.replace('.txt', '_sg.txt'))

            # Save the new file **without a header**
            df.to_csv(output_file, index=False, sep='\t', header=False)

            

        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")


