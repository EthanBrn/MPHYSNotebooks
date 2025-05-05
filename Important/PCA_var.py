import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_variance(csv_path, output_dir, n_components=10):
    """
    Reads a dataset, applies PCA, and plots the variance explained by each principal component.

    - Standardizes the data before PCA.
    - Computes the variance explained by the top `n_components`.
    - Saves the variance plot as a PNG.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load dataset
    df = pd.read_csv(csv_path).dropna()

    # Drop non-numeric columns (e.g., 'track_type', 'filename')
    X = df.select_dtypes(include=[np.number])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Variance explained by each component
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage

    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    # Plot variance explained
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, label="Individual Explained Variance")
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='--', color='r', label="Cumulative Variance")

    plt.xlabel("Principal Components")
    plt.ylabel("Variance Explained (%)")
    plt.title("PCA Explained Variance Plot")
    plt.legend()
    plt.grid()

    # Save plot
    output_path = os.path.join(output_dir, "pca_variance_plot.png")
    plt.savefig(output_path)
    plt.close()

    print(f"PCA variance plot saved at: {output_path}")

# Example usage
csv_path = "/users/eb2019/FINALDATA/CSVS/Full_data/Cubic/WTS_CUBIC.csv"
output_dir = "/users/eb2019/FINALDATA/PCA_Plots"

plot_pca_variance(csv_path, output_dir, n_components=10)
