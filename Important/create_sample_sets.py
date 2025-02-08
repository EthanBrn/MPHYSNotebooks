import os
import pandas as pd
import random

def create_sample_sets(csv_dirs, groups, num_rows_per_set, output_dir_sg, output_dir_cubic):
    """
    Creates sample sets exported as CSV's for ('WTS', 'ES'), ('WTS', 'EDS'), ('EDS', 'ES'). 
    Made to be put through machine learning classifier. 
    
    CSV Naming Scheme:
    - Each file must include a bacterial type : 'WTS', 'ES', or 'EDS'.
    - Files are stored in  directories: one for SG (Savitzky-Golay) and one for CUBIC (Cubic Spline).
    
    Parameters:
    - csv_dirs: Dictionary containing paths to SG and CUBIC directories.
    - groups: List of bacterial type pairs.
    - num_rows_per_set: Total number tracks used per set.
    - output_dir_sg: Where to store SG sample sets
    - output_dir_cubic: Where to store Cubic sample sets
    """

    # Ensure output directories exist, create them if they don't
    if not os.path.exists(output_dir_sg):
        os.makedirs(output_dir_sg)
    if not os.path.exists(output_dir_cubic):
        os.makedirs(output_dir_cubic)
    
    # Get lists of CSV files for each bacterial type in each smoothing type
    sg_files = {
        btype: [os.path.join(csv_dirs['SG'], f) for f in os.listdir(csv_dirs['SG']) if btype in f] 
        for btype in ['WTS', 'ES', 'EDS']
    }
    cubic_files = {
        btype: [os.path.join(csv_dirs['CUBIC'], f) for f in os.listdir(csv_dirs['CUBIC']) if btype in f] 
        for btype in ['WTS', 'ES', 'EDS']
    }
    
    # Iterate over both smoothing methods (SG and CUBIC)
    for smoothing, file_dict, output_dir in zip(['SG', 'CUBIC'], [sg_files, cubic_files], [output_dir_sg, output_dir_cubic]):
        for group in groups:
            group_name = f"{group[0]}_{group[1]}_{smoothing}"  # Format the output file name
            sample_data = []
            
            # Select equal number of rows from each bacterial type in the group
            for btype in group:
                # Select 2 random files for each bacterial type (ensuring variety)
                selected_files = random.sample(file_dict[btype], min(len(file_dict[btype]), 2))
                
                # Read and concatenate the selected files
                all_data = pd.concat([pd.read_csv(f) for f in selected_files], ignore_index=True)
                
                # Randomly sample rows, ensuring equal representation from both types
                sampled_data = all_data.sample(n=min(num_rows_per_set // 2, len(all_data)), random_state=42, replace=False)
                
                # Store data
                sample_data.append(sampled_data)
            
            # Combine both bacterial type samples into one dataset
            combined_df = pd.concat(sample_data, ignore_index=True)
            
            # Define output file path
            output_file = os.path.join(output_dir, f"{group_name}.csv")
            
            # Save the dataset
            combined_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")



# Dictionary containing paths to original datasets
csv_dirs = {
    "SG": "/users/eb2019/FINALDATA/CSVS/Full_data/SG",  # Directory containing SG-smoothed data
    "CUBIC": "/users/eb2019/FINALDATA/CSVS/Full_data/Cubic"  # Directory containing CUBIC-smoothed data
}

# Bacterial type groups for classification
groups = [('WTS', 'ES'), ('WTS', 'EDS'), ('EDS', 'ES')]

# Number of rows to sample per dataset (evenly split between two bacterial types)
num_rows_per_set = 4000  

# Output directories for processed datasets
output_dir_sg = "/users/eb2019/FINALDATA/CSVS/8_2_25_Cubic_vs_SG_samples/SG"
output_dir_cubic = "/users/eb2019/FINALDATA/CSVS/8_2_25_Cubic_vs_SG_samples/CUBIC"

# Generate the sample sets
create_sample_sets(csv_dirs, groups, num_rows_per_set, output_dir_sg, output_dir_cubic)
