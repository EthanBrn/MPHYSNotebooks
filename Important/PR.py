import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score

# Function to evaluate different decision thresholds for a Gradient Boosting classifier
def evaluate_decision_thresholds(csv_dir, output_csv, plot_dir, num_runs=3, thresholds=np.arange(0.01, 1.0, 0.01)):
    results = []  # Store results for each file
    
    # Ensure the plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Loop through all CSV files in the directory
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(csv_dir, file)
            df = pd.read_csv(file_path)

            # Skip files without the 'track_type' column
            if 'track_type' not in df.columns:
                print(f"Skipping {file} (no 'track_type' column)")
                continue

            # Preprocessing
            df = df.dropna()  # Confirm missing values removed
            X = df.drop(columns=[col for col in ['track_type', 'filename'] if col in df.columns])  # Features
            y = df['track_type']  # Target variable

            # Encode categorical target variable into numeric values
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Normalise feature values between 0 and 1
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

            # Storage for averaging across multiple runs
            precision_dict = {t: [] for t in thresholds}
            recall_dict = {t: [] for t in thresholds}
            accuracy_dict = {t: [] for t in thresholds}
            roc_auc_list = []  # Store ROC AUC scores
            total_run_time = 0  # Track total runtime

            for run in range(num_runs):  # Multiple runs for better average
                # Split dataset into training and test sets (stratified to preserve class balance)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=run
                )

                # Define Gradient Boosting Classifier with fixed hyperparameters
                gb = GradientBoostingClassifier(
                    learning_rate=0.1,  # Learning rate for boosting
                    max_depth=3,  # Tree depth to prevent overfitting
                    min_samples_leaf=5,  # Minimum samples per leaf
                    min_samples_split=5,  # Minimum samples needed to split
                    n_estimators=50  # Number of boosting stages
                )

                # Train the model and record runtime
                start_time = time.time()
                gb.fit(X_train, y_train)
                end_time = time.time()
                total_run_time += (end_time - start_time)

                # Get predicted probabilities for the positive class
                y_probs = gb.predict_proba(X_test)[:, 1]

                # Calculate precision, recall, and associated thresholds
                precision, recall, threshold_values = precision_recall_curve(y_test, y_probs)

                # Calculate ROC AUC score
                roc_auc = roc_auc_score(y_test, y_probs)
                roc_auc_list.append(roc_auc)

                # Evaluate model performance across different decision thresholds
                for threshold in thresholds:
                    y_pred = (y_probs >= threshold).astype(int)  # Convert probabilities to binary labels
                    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

                    # Find closest threshold index in Calculated precision-recall curve
                    closest_idx = np.searchsorted(threshold_values, threshold, side="right") - 1
                    precision_val = precision[closest_idx] if closest_idx >= 0 else np.nan
                    recall_val = recall[closest_idx] if closest_idx >= 0 else np.nan

                    # Store results for averaging later
                    precision_dict[threshold].append(precision_val)
                    recall_dict[threshold].append(recall_val)
                    accuracy_dict[threshold].append(accuracy)

            # Calculate average values across multiple runs
            avg_precision = {t: np.nanmean(precision_dict[t]) for t in thresholds}
            avg_recall = {t: np.nanmean(recall_dict[t]) for t in thresholds}
            avg_accuracy = {t: np.nanmean(accuracy_dict[t]) for t in thresholds}
            avg_roc_auc = np.mean(roc_auc_list)
            avg_run_time = total_run_time / num_runs

            # Save results to the list
            for threshold in thresholds:
                results.append({
                    "File": file,
                    "Threshold": threshold,
                    "Precision": avg_precision[threshold],
                    "Recall": avg_recall[threshold],
                    "Accuracy": avg_accuracy[threshold],
                    "ROC_AUC": avg_roc_auc,
                    "Run_Time_Seconds": avg_run_time
                })

            # Plot Precision, Recall, Accuracy, and ROC AUC against Threshold
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, list(avg_precision.values()), label="Precision", marker="o", markersize=2)
            plt.plot(thresholds, list(avg_recall.values()), label="Recall", marker="s", markersize=2)
            plt.plot(thresholds, list(avg_accuracy.values()), label="Accuracy", marker="d", markersize=2)
            plt.axhline(avg_roc_auc, color="r", linestyle="--", label=f"ROC AUC: {avg_roc_auc:.3f}")

            plt.xlabel("Decision Threshold")
            plt.ylabel("Score")
            plt.title(f"Precision, Recall, Accuracy, and ROC AUC vs. Threshold ({file})")
            plt.legend()
            plt.grid(True)

            # Save plot
            plot_path = os.path.join(plot_dir, f"{file}_Threshold_Curve.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved Precision-Recall-Accuracy-ROC plot for {file} at {plot_path}")

            print(f"Processed {file}: Avg Time = {avg_run_time:.2f} sec")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Threshold evaluation results saved to {output_csv}")

# Directories for input and output
output_dir_cubic = "/users/eb2019/FINALDATA/CSVS/cubic_samples_4000"
output_csv_cubic = "/users/eb2019/FINALDATA/GB_thresholds_avg.csv"
output_plot_dir = "/users/eb2019/FINALDATA/Plots_2"

# Use function
print("\nEvaluating Gradient Boosting Decision Thresholds with Averaging:")
evaluate_decision_thresholds(output_dir_cubic, output_csv_cubic, output_plot_dir)
