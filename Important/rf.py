import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score

def evaluate_decision_thresholds(csv_dir, output_csv, plot_dir, num_runs=3, thresholds=np.arange(0.01, 1.0, 0.01)):
    results = []
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(csv_dir, file)
            df = pd.read_csv(file_path)

            if 'track_type' not in df.columns:
                print(f"Skipping {file} (no 'track_type' column)")
                continue

            # Preprocessing
            df = df.dropna()
            X = df.drop(columns=[col for col in ['track_type', 'filename', 'path_length', 'intercept', 'gradient'] if col in df.columns])  
            y = df['track_type']  

            # Encode labels as integers
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # MinMax scale features
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

            # Storage for averaging across runs
            precision_dict = {t: [] for t in thresholds}
            recall_dict = {t: [] for t in thresholds}
            accuracy_dict = {t: [] for t in thresholds}
            roc_auc_list = []
            total_run_time = 0

            for run in range(num_runs):
                # Train-test split (reshuffled each run)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=run
                )

                # Define fixed Random Forest parameters
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    min_samples_split=5,
                    random_state=run
                )

                start_time = time.time()
                rf.fit(X_train, y_train)
                end_time = time.time()
                total_run_time += (end_time - start_time)

                # Get predicted probabilities for positive class
                y_probs = rf.predict_proba(X_test)[:, 1]

                # Compute precision, recall, and thresholds
                precision, recall, threshold_values = precision_recall_curve(y_test, y_probs)

                # Compute ROC AUC
                roc_auc = roc_auc_score(y_test, y_probs)
                roc_auc_list.append(roc_auc)

                for threshold in thresholds:
                    y_pred = (y_probs >= threshold).astype(int)
                    accuracy = accuracy_score(y_test, y_pred)

                    closest_idx = np.searchsorted(threshold_values, threshold, side="right") - 1
                    precision_val = precision[closest_idx] if closest_idx >= 0 else np.nan
                    recall_val = recall[closest_idx] if closest_idx >= 0 else np.nan

                    precision_dict[threshold].append(precision_val)
                    recall_dict[threshold].append(recall_val)
                    accuracy_dict[threshold].append(accuracy)

            # Compute averages
            avg_precision = {t: np.nanmean(precision_dict[t]) for t in thresholds}
            avg_recall = {t: np.nanmean(recall_dict[t]) for t in thresholds}
            avg_accuracy = {t: np.nanmean(accuracy_dict[t]) for t in thresholds}
            avg_roc_auc = np.mean(roc_auc_list)
            avg_run_time = total_run_time / num_runs

            # Save results
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

            # Plot Precision, Recall, Accuracy, and ROC AUC vs. Decision Threshold
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

# Directories
output_dir_cubic = "/users/eb2019/scratch/CSVS/cubic_samples_4000"
output_csv_cubic = "/users/eb2019/scratch/RF_thresholds_avg.csv"
output_plot_dir = "/users/eb2019/scratch/Plots_2"

# Run evaluation with Precision-Recall-Accuracy-ROC curves
print("\nEvaluating Random Forest Decision Thresholds with Averaging:")
evaluate_decision_thresholds(output_dir_cubic, output_csv_cubic, output_plot_dir)
