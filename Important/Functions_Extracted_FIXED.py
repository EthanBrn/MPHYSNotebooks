import os
import pandas as pd 
import numpy as np
import csv
from scipy.stats import linregress  # (Kept in case you want to add regression later)


import numpy as np
import pandas as pd
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


def calc_curvature_arr(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, 
                       names=['time', 'x_smooth', 'y_smooth', 'z_smooth'])
    if len(data) <= 2:
        return np.array([])
    v_arr = np.zeros(len(data)) #init velocity array 
    t_arr = np.zeros(len(data))  #init time array    
    T_arr = np.zeros((len(data)-2, 4))  # 2D array with time + 3 components t, x,y,z

    for i in range(1, len(data)-1):

    
        x_smooth = np.array(data['x_smooth'])  #Extracting data
        y_smooth = np.array(data['y_smooth'])
        z_smooth = np.array(data['z_smooth'])
        time = np.array(data['time'])
    
        t_next = time[i+1]       #Define points for central difference
        x_next = x_smooth[i+1]
        y_next = y_smooth[i+1]
        z_next = z_smooth[i+1]
    
        t_prev = time[i-1]
        x_prev = x_smooth[i-1]
        y_prev = y_smooth[i-1]
        z_prev = z_smooth[i-1]
        dir_vector = np.array([x_next - x_prev, y_next - y_prev, z_next - z_prev])
        T_vector = dir_vector / np.linalg.norm(dir_vector)
        T_arr[i-1] = np.array([time[i], *T_vector])  
    
        r_diff = np.sqrt((x_next-x_prev)**2+(y_next-y_prev)**2+(z_next-z_prev)**2)
        dt = t_next - t_prev
        v = r_diff/ (dt)

        v_arr[i] = v
    time_for_T = time[1:-1] # Time associated with the T vectors, lost first and last point 
        
    v_arr = v_arr[2:-2]    #velocity for final calculation. Los first and last two points
        

    dT_arr = np.zeros((len(T_arr)-2, 4))  #init dT array, loses frist and last point from T_arr
    
    for i in range(1, len(T_arr)-1):
        dT = (T_arr[i+1, 1:] - T_arr[i-1, 1:]) / (T_arr[i+1, 0] - T_arr[i-1, 0])   #central diff for dT/dt 
        dT_arr[i-1] = np.array([T_arr[i, 0], *dT])#Put associated t x, y ,z in array
        
    curvature_arr = np.zeros(len(dT_arr))    #init curve array 
    
    for i in range(len(v_arr)):
        
        dT_segment = dT_arr[i, 1:]    #extract only x y z 
        dT_magnitude = np.linalg.norm(dT_segment)  #mag of x y z vector 
        v_mag = v_arr[i] #veloctity associated with this point 
        
        curvature = dT_magnitude / v_mag  #calculate curvature
        curvature_arr[i] = curvature   #put in array 
    times_for_curvature = time[2:-2]   #Time associated with the curvature array.
    
    return curvature_arr , times_for_curvature

def reorientation_events_per_second(file_path):
    """
    Estimates reorientation events per second by counting sequences where the dot product between
    unit direction vectors (separated by k steps) falls below a threshold.
    """
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x_smooth', 'y_smooth', 'z_smooth'])
    # Ensure time ordering
    data.sort_values(by='time', inplace=True)
    
    time = data['time'].values
    x = data['x_smooth'].values
    y = data['y_smooth'].values
    z = data['z_smooth'].values
    positions = np.column_stack((x, y, z))
    
    # Compute differences between consecutive positions and their magnitudes
    dir_vectors = np.diff(positions, axis=0)
    mags = np.linalg.norm(dir_vectors, axis=1)
    
    # Compute unit direction vectors, avoiding division by zero
    unit_dir = np.zeros_like(dir_vectors)
    nonzero = mags > 0
    unit_dir[nonzero] = dir_vectors[nonzero] / mags[nonzero, None]
    
    # Calculate dot products for vectors k steps apart
    k = 5
    dot_products = [np.dot(unit_dir[i], unit_dir[i+k]) for i in range(len(unit_dir) - k)]
    
    change_in_dir_threshold = 0.99
    consecutive_low_values = 3
    low_sequences = []
    current_sequence = []
    
    for dp in dot_products:
        if dp < change_in_dir_threshold:
            current_sequence.append(dp)
        else:
            if len(current_sequence) >= consecutive_low_values:
                low_sequences.append(current_sequence)
            current_sequence = []
    if len(current_sequence) >= consecutive_low_values:
        low_sequences.append(current_sequence)
    
    events_per_sec = len(low_sequences) / (time[-1] - time[0])
    return events_per_sec


def calculate_path_length(file_path):
    """
    Calculates the total path length as the sum of Euclidean distances between consecutive points.
    """
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x_smooth', 'y_smooth', 'z_smooth'])
    x = data['x_smooth'].values
    y = data['y_smooth'].values
    z = data['z_smooth'].values
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    return path_length

def calculate_straight_distance(file_path):
    """
    Calculates the straight-line distance between the first and last point.
    """
    data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                       names=['time', 'x_smooth', 'y_smooth', 'z_smooth'])
    start_point = data.iloc[0][['x_smooth', 'y_smooth', 'z_smooth']].values
    end_point = data.iloc[-1][['x_smooth', 'y_smooth', 'z_smooth']].values
    straight_distance = np.linalg.norm(end_point - start_point)
    return straight_distance

def calculate_tortuosity(file_path):
    """
    Calculates tortuosity as the ratio of the path length to the straight-line distance.
    """
    path_length = calculate_path_length(file_path)
    straight_distance = calculate_straight_distance(file_path)
    if straight_distance == 0:
        return np.inf
    return path_length / straight_distance


def extract_features_to_array(file_path, min_length=10):
    """
    Extracts features from the file and returns them as an array.
    Returns None if the track is too short.
    Features include:
      - Mean, standard deviation, max, and min of log(velocity)
      - Mean, standard deviation, max, and min of acceleration
      - Reorientation events per second
      - Mean, standard deviation, and max curvature
      - Tortuosity and path length
    """
    v, _ = calculate_velocity_array(file_path)
    a, _ = calculate_acceleration_array(file_path)
    reorientations_per_sec = reorientation_events_per_second(file_path)
    curvature, _ = calc_curvature_arr(file_path)
    tortuosity = calculate_tortuosity(file_path)
    path_length = calculate_path_length(file_path)
    
    # Check that the track has a sufficient number of velocity points
    if len(v) < min_length:
        return None
    
    # Compute log(velocity) for positive velocities
    log_velocity = np.log(v[v > 0]) if np.any(v > 0) else np.array([])
    
    features = [
        np.mean(log_velocity) if log_velocity.size > 0 else np.nan,
        np.std(log_velocity) if log_velocity.size > 0 else np.nan,
        np.max(log_velocity) if log_velocity.size > 0 else np.nan,
        np.min(log_velocity) if log_velocity.size > 0 else np.nan,
        np.mean(a) if a.size > 0 else np.nan,
        np.std(a) if a.size > 0 else np.nan,
        np.max(a) if a.size > 0 else np.nan,
        np.min(a) if a.size > 0 else np.nan,
        reorientations_per_sec if reorientations_per_sec is not None else np.nan,
        np.mean(curvature) if curvature.size > 0 else np.nan,
        np.std(curvature) if curvature.size > 0 else np.nan,
        np.max(curvature) if curvature.size > 0 else np.nan,
        tortuosity if tortuosity is not None else np.nan,
        path_length if path_length is not None else np.nan
    ]
    return features

def write_features_to_csv(directory_path, save_path, min_length=10):
    """
    Iterates through files in the specified directory, extracts features from each file,
    and writes them to a CSV file.
    """
    with open(save_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'track_type', 'filename',
            'mean_log_velocity', 'stddev_log_velocity',
            'max_log_velocity', 'min_log_velocity',
            'mean_acceleration', 'stddev_acceleration',
            'max_acceleration', 'min_acceleration',
            'reorientations_per_sec',
            'mean_curvature', 'stddev_curvature', 'max_curvature',
            'tortuosity', 'path_length'
        ])
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                try:
                    features = extract_features_to_array(file_path, min_length)
                    if features is not None:
                        track_type = filename.split('_')[0]  # Assumes track type is part of the filename
                        csv_writer.writerow([track_type, filename] + features)
                except Exception as e:
                    # Optionally log errors: print(f"Error processing {filename}: {e}")
                    continue

# Example usage:
directory_path = '/users/eb2019/scratch/swimmers/tests'
save_path = '/users/eb2019/new_csvs/test_new.csv'
write_features_to_csv(directory_path, save_path)
