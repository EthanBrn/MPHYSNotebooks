import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def classify_track_type(csv_dir, output_csv):
    """
    Performs machine learning classification on the 'track_type' column for each CSV in the directory using an MLP Classifier.
    Outputs a CSV with accuracy metrics and confusion matrices.

    Parameters:
    - csv_dir: Directory containing the CSV files to process.
    - output_csv: Path to save the output CSV containing accuracy and confusion matrices.
    """
    results = []  # Store results for all files

    # Iterate over all CSV files in the specified directory
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):  # Ensure it's a CSV file
            file_path = os.path.join(csv_dir, file)
            df = pd.read_csv(file_path)  # Load data

            # Skip files that do not contain the 'track_type' column
            if 'track_type' not in df.columns:
                print(f"Skipping {file} (no 'track_type' column)")
                continue
            
            # Data Preprocessing
            df = df.dropna()  # Remove any missing values
            X = df.drop(columns=[col for col in ['track_type', 'filename'] if col in df.columns])  # Features
            y = df['track_type']  # Target variable

            # Train-test split: 80% training, 20% testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            # Train MLP Classifier
            model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate performance
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
            conf_matrix = confusion_matrix(y_test, y_pred)  # Generate confusion matrix

            # Flatten confusion matrix if more than 2 classes
            conf_matrix_flat = conf_matrix.flatten() if conf_matrix.shape[0] <= 2 else conf_matrix.ravel()
            
            # Store results for this file
            result_entry = {'File': file, 'Accuracy': accuracy}
            for i, value in enumerate(conf_matrix_flat):
                result_entry[f'CM_{i}'] = value  # Store confusion matrix values

            results.append(result_entry)  # Append to results list
            print(f"Processed {file}: Accuracy = {accuracy:.4f}")

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Classification results saved to {output_csv}")

# Directories containing SG and CUBIC smoothed bacterial track data
output_dir_sg = "/users/eb2019/FINALDATA/CSVS/8_2_25_Cubic_vs_SG_samples/SG"
output_dir_cubic = "/users/eb2019/FINALDATA/CSVS/8_2_25_Cubic_vs_SG_samples/CUBIC"

# Output files for accuracy results
output_csv_sg = "/users/eb2019/FINALDATA/CSVS/8_2_25_Cubic_vs_SG_samples/SG_results.csv"
output_csv_cubic = "/users/eb2019/FINALDATA/CSVS/8_2_25_Cubic_vs_SG_samples/CUBIC_results.csv"

# Perform classification on both SG and CUBIC datasets and save reports
print("\nSG Classification Results:")
classify_track_type(output_dir_sg, output_csv_sg)

print("\nCUBIC Classification Results:")
classify_track_type(output_dir_cubic, output_csv_cubic)
