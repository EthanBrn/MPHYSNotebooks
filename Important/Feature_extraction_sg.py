import os
import shutil
import pandas as pd 
import numpy as np
import csv
from scipy.stats import linregress

# Function to calculate the mean squared displacement (MSD) for different time intervals (tau)
def mean_squared_displacement_optimised(file_path):
    # Read the CSV file with whitespace as a delimiter and assign column names
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
    
    # Extract relevant columns as numpy arrays
    x_sg = np.array(data['x_sg'])
    y_sg = np.array(data['y_sg'])
    z_sg = np.array(data['z_sg'])
    time = np.array(data['time'])
    
    msd_by_tau = {}  # Dictionary to store MSD values for different tau values

    # Loop over different time intervals (tau)
    for tau in range(1, len(time)):
        # Calculate the displacement in x, y, and z directions
        dx = x_sg[tau:] - x_sg[:-tau]
        dy = y_sg[tau:] - y_sg[:-tau]
        dz = z_sg[tau:] - z_sg[:-tau]
        
        # Calculate squared displacement and store the mean value
        squared_displacements = dx**2 + dy**2 + dz**2
        msd_by_tau[round(time[tau] - time[0], 3)] = np.mean(squared_displacements)

    return msd_by_tau

# Function to find the gradient and intercept of the MSD curve on a log-log scale
def find_gradient_intercept(file_path):
    msd_by_tau = mean_squared_displacement_optimised(file_path)
    
    # Convert dictionary keys (tau) and values (MSD) into numpy arrays
    tau_arr = np.array(list(msd_by_tau.keys()))
    msd_arr = np.array(list(msd_by_tau.values()))
    
    # Determine the upper limit for tau values to be used in regression 20th of the data deemed suffcient 
    tau_max = tau_arr.max()
    tau_limit = tau_max / 20
    mask = tau_arr <= tau_limit  # Only consider tau values within this limit
    
    # Filter the tau and MSD arrays based on the mask
    limited_tau = tau_arr[mask]
    limited_msd = msd_arr[mask]
    
    # Apply logarithm scale
    log_tau = np.log(limited_tau)
    log_msd = np.log(limited_msd)
    
    # Perform linear regression in log-log scale
    try:
        slope, intercept, _, _, _ = linregress(log_tau, log_msd)
        return slope, intercept
    except ValueError:
        return np.nan, np.nan  # Return NaN values if regression fails

# Function to calculate curvature along bacterial tracks over time
def calc_curvature_arr(file_path):
    # Read the data file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
    
    if len(data) <= 2:
        return np.array([])  # Return an empty array if there are not enough points
    
    v_arr = np.zeros(len(data))  # Initialize velocity array
    t_arr = np.zeros(len(data))  # Initialize time array    
    T_arr = np.zeros((len(data)-2, 4))  # Array for tangent vectors with associated time

    for i in range(1, len(data)-1):
        # Extract necessary data columns
        x_sg = np.array(data['x_sg'])  
        y_sg = np.array(data['y_sg'])
        z_sg = np.array(data['z_sg'])
        time = np.array(data['time'])
    
        # Define next and previous points for central difference calculation
        t_next = time[i+1]       
        x_next = x_sg[i+1]
        y_next = y_sg[i+1]
        z_next = z_sg[i+1]
    
        t_prev = time[i-1]
        x_prev = x_sg[i-1]
        y_prev = y_sg[i-1]
        z_prev = z_sg[i-1]
        
        # Calculate the directional vector
        dir_vector = np.array([x_next - x_prev, y_next - y_prev, z_next - z_prev])
        T_vector = dir_vector / np.linalg.norm(dir_vector)  # Normalize to get unit tangent vector
        T_arr[i-1] = np.array([time[i], *T_vector])  
    
        # Calculate velocity magnitude
        r_diff = np.sqrt((x_next-x_prev)**2 + (y_next-y_prev)**2 + (z_next-z_prev)**2)
        dt = t_next - t_prev
        v = r_diff / dt
        v_arr[i] = v
    
    # Time values corresponding to tangent vectors
    time_for_T = time[1:-1] 
    
    # Remove first and last two velocity points to match the rate of change of tangent vectors
    v_arr = v_arr[2:-2]    
    
    # Initialize the rate of change of tangent vector array
    dT_arr = np.zeros((len(T_arr)-2, 4))  
    
    for i in range(1, len(T_arr)-1):
        # Calculate the derivative of the tangent vector using the central difference
        dT = (T_arr[i+1, 1:] - T_arr[i-1, 1:]) / (T_arr[i+1, 0] - T_arr[i-1, 0])   
        dT_arr[i-1] = np.array([T_arr[i, 0], *dT])
    
    # Initialize curvature array
    curvature_arr = np.zeros(len(dT_arr))
    
    for i in range(len(v_arr)):
        # Extract x, y, z components of dT/dt
        dT_segment = dT_arr[i, 1:]    
        dT_magnitude = np.linalg.norm(dT_segment)  # Calculate magnitude of dT/dt
        v_mag = v_arr[i]  # Get velocity magnitude
        
        # Calculate curvature as |dT/dt| / |v|
        curvature = dT_magnitude / v_mag  
        curvature_arr[i] = curvature   
    
    # Time values corresponding to curvature values
    times_for_curvature = time[2:-2]   
    
    return curvature_arr, times_for_curvature  # Return curvature and associated time array

def calc_acc(file_path):
    # Read the data from the file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
    
    # If there are not enough data points, return an empty array
    if len(data) <= 2:
        return np.array([])
    
    # Initialize arrays for velocity, time, and acceleration
    v_arr = np.zeros(len(data))
    t_arr = np.zeros(len(data)-1)
    a_arr = np.zeros(len(data)-2)
    
    for i in range(1, len(data)-1):
        # Extract smoothed x, y, z coordinates and time
        x_sg = np.array(data['x_sg'])
        y_sg = np.array(data['y_sg'])
        z_sg = np.array(data['z_sg'])
        time = np.array(data['time'])
    
        # Get next and previous time and position values
        t_next = time[i+1]
        x_next = x_sg[i+1]
        y_next = y_sg[i+1]
        z_next = z_sg[i+1]
    
        t_prev = time[i-1]
        x_prev = x_sg[i-1]
        y_prev = y_sg[i-1]
        z_prev = z_sg[i-1]
        
        # Calculate the radial displacement between previous and next points
        r_diff = np.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2 + (z_next - z_prev)**2)
        dt = t_next - t_prev
    
        # Calculate velocity using central difference method
        v_arr[i] = r_diff / dt
    
    # Extract relevant time values and velocity values
    t_arr = time[1:-1]
    v_arr = v_arr[1:-1]
    
    # Calculate acceleration using central difference method
    for i in range(1, len(v_arr) - 1):
        dv = v_arr[i + 1] - v_arr[i - 1]
        dt = time[i + 1] - time[i - 1]
        a_arr[i] = dv / dt
    
    # Extract relevant time values for acceleration
    t_for_acc = time[2:-2] 
    a_arr = a_arr[1:-1]   
    
    return a_arr, t_for_acc


def calc_velo_arr(file_path):
    # Read the data from the file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
    
    # Extract smoothed x, y, z coordinates and time
    x_sg = np.array(data['x_sg'])
    y_sg = np.array(data['y_sg'])
    z_sg = np.array(data['z_sg'])
    time = np.array(data['time'])
    
    # If there are not enough data points, return an empty array
    if len(data) <= 2:
        return np.array([])
    
    # Initialize velocity array
    v_arr = np.zeros(len(data)-2)
    
    for i in range(1, len(data)-1):
        # Get next and previous time and position values
        t_next = time[i+1]
        x_next = x_sg[i+1]
        y_next = y_sg[i+1]
        z_next = z_sg[i+1]
    
        t_prev = time[i-1]
        x_prev = x_sg[i-1]
        y_prev = y_sg[i-1]
        z_prev = z_sg[i-1]
    
        # Calculate the radial displacement between previous and next points
        r_diff = np.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2 + (z_next - z_prev)**2)
        dt = t_next - t_prev
    
        # Calculate velocity using central difference method
        v = r_diff / dt
        v_arr[i-1] = v
    
    return v_arr


def calculate_path_length(file_path):
    # Read the data from the file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
    
    # Extract smoothed x, y, z coordinates
    x_sg = np.array(data['x_sg'])
    y_sg = np.array(data['y_sg'])
    z_sg = np.array(data['z_sg'])
    
    # Calculate total path length as sum of distances between consecutive points
    path_length = np.sum(np.sqrt(np.diff(x_sg)**2 + np.diff(y_sg)**2 + np.diff(z_sg)**2))
    
    return path_length

def calculate_straight_distance(file_path):
    # Read the data from the .txt file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
    
    # Extract starting and ending coordinates
    start_point = data.iloc[0][['x_sg', 'y_sg', 'z_sg']].to_numpy()
    end_point = data.iloc[-1][['x_sg', 'y_sg', 'z_sg']].to_numpy()
    
    # Calculate the straight-line distance
    straight_distance = np.linalg.norm(end_point - start_point)
    
    return straight_distance

def calculate_tortuosity(file_path):
    # Calculate path length and straight-line distance
    path_length = calculate_path_length(file_path)
    straight_distance = calculate_straight_distance(file_path)
    
    # Avoid division by zero
    if straight_distance == 0:
        return np.inf  # Infinite tortuosity if straight distance is zero
    
    # Tortuosity calculation
    tortuosity = path_length / straight_distance
    return tortuosity

def reorientation_events_per_second(file_path):
    # Read the data from file
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
    
    # Extract position and time columns
    x_sg = np.array(data['x_sg'])
    y_sg = np.array(data['y_sg'])
    z_sg = np.array(data['z_sg'])
    time = data['time']
    
    positions = np.stack((x_sg, y_sg, z_sg), axis=1)
    
    # Calculate direction vectors between consecutive points
    dir_vectors = np.diff(positions, axis=0)
    
    # Calculate magnitudes of direction vectors
    mags = np.linalg.norm(dir_vectors, axis=1)
    
    # Avoid division by zero
    non_zero_mags = mags > 0  
    unit_dir = np.zeros_like(dir_vectors)  
    unit_dir[non_zero_mags] = dir_vectors[non_zero_mags] / mags[non_zero_mags, np.newaxis]
    
    # Calculate dot products k steps ahead
    k = 5
    dot_products = [np.dot(unit_dir[i], unit_dir[i + k]) for i in range(len(unit_dir) - k) if i + k < len(unit_dir)]
    
    # Threshold for detecting changes in direction
    change_in_dir_threshold = 0.99
    consecutive_low_values = 3
    
    low_sequences = []
    current_sequence = []
    
    # Identify sequences where dot products are below threshold
    for dp in dot_products:
        if dp < change_in_dir_threshold:
            current_sequence.append(dp)
        else:
            if len(current_sequence) >= consecutive_low_values:
                low_sequences.append(current_sequence)
            current_sequence = []  # Reset for the next sequence
    
    # Capture any remaining sequence
    if len(current_sequence) >= consecutive_low_values:
        low_sequences.append(current_sequence)
    
    # Calculate events per second
    events_per_sec = len(low_sequences) / (np.max(time) - np.min(time))
    return events_per_sec

def extract_features_to_array(file_path, min_length=10):
    # Extract various movement features from the bacterial track file
    msd_by_tau = mean_squared_displacement_optimised(file_path)
    velocity = calc_velo_arr(file_path)
    acceleration = calc_acc(file_path)
    reorientations_per_sec = reorientation_events_per_second(file_path)
    curvature = calc_curvature_arr(file_path)
    tortuosity = calculate_tortuosity(file_path)
    path_length = calculate_path_length(file_path)
    slope, intercept = find_gradient_intercept(file_path)
    
    # Skip tracks that are too short
    if len(msd_by_tau) < min_length or len(velocity) < min_length:
        return None  
    
    # Ensure valid feature arrays or replace empty ones with NaN
    msd_arr = np.array(list(msd_by_tau.values())) if len(msd_by_tau) > 0 else np.nan
    log_velocity = np.log(velocity[velocity > 0]) if len(velocity) > 0 else np.nan
    
    # Return exactly 19 extracted features
    features = [
        slope if slope else np.nan,
        intercept if intercept else np.nan,
        np.mean(log_velocity) if len(log_velocity) > 0 else np.nan,
        np.std(log_velocity) if len(log_velocity) > 0 else np.nan,
        np.max(log_velocity) if len(log_velocity) > 0 else np.nan,
        np.min(log_velocity) if len(log_velocity) > 0 else np.nan,
        np.mean(acceleration) if len(acceleration) > 0 else np.nan, 
        np.std(acceleration) if len(acceleration) > 0 else np.nan,
        np.max(acceleration) if len(acceleration) > 0 else np.nan, 
        np.min(acceleration) if len(acceleration) > 0 else np.nan,
        reorientations_per_sec if reorientations_per_sec else np.nan,  
        np.mean(curvature) if len(curvature) > 0 else np.nan,
        np.std(curvature) if len(curvature) > 0 else np.nan,
        np.max(curvature) if len(curvature) > 0 else np.nan, 
        tortuosity if tortuosity else np.nan, 
        path_length if path_length else np.nan
    ]
    
    return features

def write_features_to_csv(directory_path, min_length=10, output_file='ES_sg.csv'):
    # Ensure the output directory exists
    save_path = os.getcwd()
    output_path = os.path.join(save_path, output_file)
    
    # Open CSV file for writing extracted features
    with open(output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header row
        csv_writer.writerow([
            'track_type', 'filename',
            'gradient', 'intercept', 'mean_log_velocity', 'stddev_log_velocity',
            'max_log_velocity', 'min_log_velocity', 'mean_acceleration', 'stddev_acceleration',
            'max_acceleration', 'min_acceleration', 'reorientations_per_sec', 'mean_curvature',
            'stddev_curvature', 'max_curvature', 'tortuosity', 'path_length'
        ])
        
        # Iterate through files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Ensure it's a valid file
            if os.path.isfile(file_path):
                try:
                    features = extract_features_to_array(file_path, min_length)
                    if features is not None:  # Skip invalid tracks
                        track_type = filename.split('_')[0]  # Extract type from filename
                        csv_writer.writerow([track_type, filename] + features)
                except:
                    continue  # Ignore problematic files

directory_path = 'sg_data/11_3/ES'
write_features_to_csv(directory_path)
