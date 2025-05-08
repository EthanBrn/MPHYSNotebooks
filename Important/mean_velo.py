import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory containing the files
directory = "/users/eb2019/FINALDATA/swimmers/WTS"  # Change this to your actual directory path
mean_velocities = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    # Ensure it is a file
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, delim_whitespace=True, header=None,
                           names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth', 'x_sg', 'y_sg', 'z_sg'])
        
        # Extract smoothed x, y, z coordinates and time
        x_smooth = np.array(data['x_smooth'])
        y_smooth = np.array(data['y_smooth'])
        z_smooth = np.array(data['z_smooth'])
        time = np.array(data['time'])
        
        # If there are not enough data points, skip file
        if len(data) <= 2:
            continue
        
        # Initialize velocity array
        v_arr = np.zeros(len(data)-2)
        
        for i in range(1, len(data)-1):
            # Get next and previous time and position values
            t_next = time[i+1]
            x_next = x_smooth[i+1]
            y_next = y_smooth[i+1]
            z_next = z_smooth[i+1]
        
            t_prev = time[i-1]
            x_prev = x_smooth[i-1]
            y_prev = y_smooth[i-1]
            z_prev = z_smooth[i-1]
        
            # Calculate the radial displacement between previous and next points
            r_diff = np.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2 + (z_next - z_prev)**2)
            dt = t_next - t_prev
        
            # Calculate velocity using central difference method
            v_arr[i-1] = r_diff / dt
        
        # Calculate mean velocity and store it
        mean_velocity = np.mean(v_arr)
        mean_velocities.append(mean_velocity)

# Plot histogram of mean velocities
plt.figure(figsize=(8, 6))
plt.hist(mean_velocities, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Mean Velocity")
plt.ylabel("Frequency")
plt.title("Histogram of Mean Velocities")
plt.savefig('wts')
plt.grid(True)
plt.show()
