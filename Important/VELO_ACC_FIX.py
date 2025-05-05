import os
import pandas as pd 
import numpy as np
import csv
from scipy.stats import linregress  # (Kept in case you want to add regression later)


def calculate_velocity_array(file_path):
    """
    Reads the file and calculates the speed between two points using vectorized central differences.
    Speed is computed as the magnitude of the displacement divided by the time difference.
    
    Returns:
      - speeds: An array of computed speeds at central time points (i.e. time[1:-1]).
      - t_central: The corresponding central time values.
    """
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth'])
    if len(data) < 3:
        return np.array([]), np.array([])
    
    # Extract arrays from the DataFrame
    time = data['time'].values
    x = data['x_smooth'].values
    y = data['y_smooth'].values
    z = data['z_smooth'].values
    
    # Compute central differences for indices 1 through N-2:
    dt = time[2:] - time[:-2]  # time differences
    dx = x[2:] - x[:-2]
    dy = y[2:] - y[:-2]
    dz = z[2:] - z[:-2]
    
    # Speed is computed as the magnitude of the displacement divided by the time difference.
    speeds = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    
    # The speeds correspond to the central time points (i.e., indices 1 to N-2)
    t_central = time[1:-1]
    
    return speeds, t_central

def calculate_acceleration_array(file_path):
    """
    Calculates acceleration as the time derivative of speed using vectorized central differences.
    It reuses the velocity array from calculate_velocity_array to ensure consistent time alignment.
    
    Negative acceleration indicates that the speed is decreasing (slowing down),
    while positive acceleration indicates that the speed is increasing (speeding up).
    
    Returns:
      - acceleration: An array of acceleration values computed at the central time points 
                      (i.e., from the speed array).
      - t_acc: The corresponding central time stamps for the acceleration values.
    """
    speeds, t_central = calculate_velocity_array(file_path)
    if speeds.size < 3:
        return np.array([]), np.array([])
    
    # Compute acceleration using central differences on the speed array:
    # For indices 1 to N-2 in the speeds array, we compute:
    # acceleration = (speeds[i+1] - speeds[i-1]) / (t_central[i+1] - t_central[i-1])
    dt_speed = t_central[2:] - t_central[:-2]
    ds = speeds[2:] - speeds[:-2]
    acceleration = ds / dt_speed
    
    # The acceleration is computed at the central time points of the speed array:
    t_acc = t_central[1:-1]
    
    return acceleration, t_acc
def extract_velocity_acceleration_features(file_path, min_length=10):
    """
    Extracts features from the file limited to velocity and acceleration.
    Features include:
      - Mean, standard deviation, maximum, and minimum of log(velocity) (only for positive speeds)
      - Mean, standard deviation, maximum, and minimum of acceleration
    Returns:
      - A list of 8 features, or None if the track is too short.
    """
    # Get the speed (velocity magnitude) and acceleration arrays.
    v, _ = calculate_velocity_array(file_path)
    a, _ = calculate_acceleration_array(file_path)
    
    # Check that there are enough data points for each.
    if v.size < min_length or a.size < min_length:
        return None
    
    # Compute log(velocity) for positive speeds
    log_velocity = np.log(v[v > 0]) if np.any(v > 0) else np.array([])
    
    features = [
        np.mean(log_velocity) if log_velocity.size > 0 else np.nan,
        np.std(log_velocity) if log_velocity.size > 0 else np.nan,
        np.max(log_velocity) if log_velocity.size > 0 else np.nan,
        np.min(log_velocity) if log_velocity.size > 0 else np.nan,  # Log velocity features
        np.mean(a) if a.size > 0 else np.nan,
        np.std(a) if a.size > 0 else np.nan,
        np.max(a) if a.size > 0 else np.nan,
        np.min(a) if a.size > 0 else np.nan,  # Acceleration features
    ]
    
    return features


def write_velocity_acceleration_features_to_csv(directory_path, save_path, min_length=10):
    """
    Iterates through files in the specified directory, extracts velocity and acceleration features
    from each file, and writes them to a CSV file.
    
    CSV columns: track_type, filename, mean_speed, std_speed, max_speed, min_speed,
                 mean_acceleration, std_acceleration, max_acceleration, min_acceleration
    """
    with open(save_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'track_type', 'filename',
            'mean_speed', 'std_speed', 'max_speed', 'min_speed',
            'mean_acceleration', 'std_acceleration', 'max_acceleration', 'min_acceleration'
        ])
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    features = extract_velocity_acceleration_features(file_path, min_length)
                    if features is not None:
                        # Assuming track type is encoded in the filename as the prefix before an underscore.
                        track_type = filename.split('_')[0]
                        csv_writer.writerow([track_type, filename] + features)
                except Exception as e:
                    # Optionally log errors:
                    # print(f"Error processing {filename}: {e}")
                    continue

# Example usage:
directory_path = '/users/eb2019/scratch/interpolated'
save_path = '/users/eb2019/new_csvs/SE_quick_v_a.csv'
write_velocity_acceleration_features_to_csv(directory_path, save_path)
