import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Suppress RuntimeWarnings if desired
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Update global font sizes for clarity in plots
matplotlib.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def mean_squared_displacement_accurate(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, 
                       names=['time', 'x', 'y', 'z', 'x_smooth', 'y_smooth', 'z_smooth'])

    x_smooth = data['x_smooth'].values
    y_smooth = data['y_smooth'].values
    z_smooth = data['z_smooth'].values
    time = data['time'].values

    msd_by_tau = {}

    for tau in range(1, len(time)):
        dx = x_smooth[tau:] - x_smooth[:-tau]
        dy = y_smooth[tau:] - y_smooth[:-tau]
        dz = z_smooth[tau:] - z_smooth[:-tau]
        squared_displacements = dx**2 + dy**2 + dz**2

        # Vectorised time differences for current tau
        tau_times = time[tau:] - time[:-tau]
        avg_tau = np.round(np.mean(tau_times), 3)

        msd_by_tau[avg_tau] = np.mean(squared_displacements)

    return msd_by_tau


def find_gradient_intercept_2(file_path):
    msd_by_tau = mean_squared_displacement_accurate(file_path)
    tau_arr = np.array(list(msd_by_tau.keys()))
    msd_arr = np.array(list(msd_by_tau.values()))
    
    tau_max = tau_arr.max()
    tau_limit = tau_max / 10
    mask = tau_arr <= tau_limit
    limited_tau = tau_arr[mask]
    limited_msd = msd_arr[mask]
    
    valid_mask = (limited_tau > 0) & (limited_msd > 0)
    if not np.any(valid_mask):
        return np.nan, np.nan
    limited_tau = limited_tau[valid_mask]
    limited_msd = limited_msd[valid_mask]
    
    log_tau = np.log(limited_tau)
    log_msd = np.log(limited_msd)
    
    try:
        slope, intercept, _, _, _ = linregress(log_tau, log_msd)
        return slope, intercept
    except Exception:
        return np.nan, np.nan

def process_directory_parallel(directory_path, max_workers=None):
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(find_gradient_intercept_2, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                slope, intercept = future.result()
                filename_only = os.path.basename(file)
                results.append([filename_only, slope, intercept])
            except Exception as e:
                print(f"Error processing {file}: {e}")
                results.append([os.path.basename(file), np.nan, np.nan])
    return pd.DataFrame(results, columns=["filename", "gradient", "intercept"])

def plot_2d_histogram(array, title, filename, dpi=300):
    out_dir = os.path.dirname(filename)
    os.makedirs(out_dir, exist_ok=True)

    valid = np.isfinite(array).all(axis=1)
    valid_array = array[valid]
    if valid_array.size == 0:
        print("No valid data to plot for:", title)
        return

    gradients = valid_array[:, 0]
    intercepts = valid_array[:, 1]

    plt.figure(figsize=(8, 6))
    plt.hist2d(gradients, intercepts, bins=50, cmap='viridis', norm=LogNorm())
    cbar = plt.colorbar(label='Log Count')
    cbar.ax.tick_params(labelsize=12)

    plt.xlabel('Gradient', fontsize=14)
    plt.ylabel('Intercept', fontsize=14)
    plt.title(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    max_workers = None

    # Process directories
    df1 = process_directory_parallel('/users/eb2019/scratch/RAW/WT_planktonic_final/', max_workers)
    df2 = process_directory_parallel('/users/eb2019/scratch/RAW/evolved+disk_planktonic_final/', max_workers)
    df3 = process_directory_parallel('/users/eb2019/scratch/RAW/evolved_planktonic_final/', max_workers)

    # Save gradient data to CSV
    df1.to_csv('/users/eb2019/scratch/Gradients/WT_planktonic_gradients2.csv', index=False)
    df2.to_csv('/users/eb2019/scratch/Gradients/evolved_disk_planktonic_gradients2.csv', index=False)
    df3.to_csv('/users/eb2019/scratch/Gradients/evolved_planktonic_gradients2.csv', index=False)

    # Plot 2D histograms
    plot_2d_histogram(df1[["gradient", "intercept"]].values, 'WT Planktonic Final',
                      '/users/eb2019/scratch/Plots/2D_hist_WT2.png', dpi=600)
    plot_2d_histogram(df2[["gradient", "intercept"]].values, 'Evolved+Disk Planktonic Final',
                      '/users/eb2019/scratch/Plots/2D_hist_evolved_disk2.png', dpi=600)
    plot_2d_histogram(df3[["gradient", "intercept"]].values, 'Evolved Planktonic Final',
                      '/users/eb2019/scratch/Plots/2D_hist_evolved2.png', dpi=600)
