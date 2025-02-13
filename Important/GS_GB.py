import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

def grid_search_gradient_boosting(csv_dir, output_csv):
    """
    Performs Grid Search to optimize Gradient Boosting Classifier hyperparameters on multiple datasets.

    - Loads and preprocesses data from CSV files in `csv_dir`
    - Uses Grid Search to find the best hyperparameters for Gradient Boosting
    - Evaluates model performance using 5-fold cross-validation
    - Saves the best parameters and accuracy for each dataset to `output_csv`
    """
    results = []  # List to store results for each dataset
    
    for file in os.listdir(csv_dir):  # Iterate over all CSV files in the directory
        if file.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(csv_dir, file)
            df = pd.read_csv(file_path)

            # Ensure dataset contains 'track_type' column (label column)
            if 'track_type' not in df.columns:
                print(f"Skipping {file} (no 'track_type' column)")
                continue  # Skip files without a target variable

            # Preprocessing
            df = df.dropna()  # Remove any rows with missing values
            
            # Select feature columns (excluding 'track_type' and 'filename' if present)
            X = df.drop(columns=[col for col in ['track_type', 'filename'] if col in df.columns])  
            y = df['track_type']  

            # Encode categorical labels into numerical values
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Standardize features (mean = 0, std = 1) to improve model performance
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Split dataset into training and testing sets (80% training, 20% testing)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # Expanded hyperparameter grid for optimization
            param_grid = {
                "n_estimators": [50, 100, 200, 300, 500],  # Number of boosting iterations
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],  # Step size for boosting
                "max_depth": [3, 5, 7, 9],  # Depth of each decision tree
                "min_samples_split": [2, 5, 10, 20],  # Min samples to split a node
                "min_samples_leaf": [1, 3, 5, 10],  # Min samples per leaf node
            }

            # Initialize the base Gradient Boosting Classifier
            gb = GradientBoostingClassifier()

            # Start timing the Grid Search process
            start_time = time.time()

            # Grid Search with 5-fold cross-validation to find the best hyperparameters
            grid_search = GridSearchCV(
                estimator=gb,  # Model to optimize
                param_grid=param_grid,  # Hyperparameter search space
                scoring="accuracy",  # Optimize for accuracy
                cv=5,  # 5-fold cross-validation
                n_jobs=-1,  # Use all available CPU cores for parallel processing
                verbose=1  # Show progress updates
            )
            grid_search.fit(X_train, y_train)  # Fit the model on training data
            
            # Stop timing after the search is complete
            end_time = time.time()
            run_time = end_time - start_time

            # Retrieve best accuracy and best hyperparameters
            best_accuracy = grid_search.best_score_
            best_params = grid_search.best_params_

            # Store results for the dataset
            result_entry = {
                "File": file,
                "Best_Accuracy": best_accuracy,  # Best accuracy from cross-validation
                "Best_Params": best_params,  # Best hyperparameter set
                "Run_Time_Seconds": run_time  # Total time taken for Grid Search
            }

            results.append(result_entry)  # Append results to list
            print(f"Processed {file}: Best Accuracy = {best_accuracy:.4f}, Time = {run_time:.2f} sec")

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Grid search results saved to {output_csv}")

# Directory containing bacterial swimming data
output_dir_cubic = "/users/eb2019/FINALDATA/CSVS/cubic_samples_4000"
output_csv_cubic = "/users/eb2019/FINALDATA/Gridsearches/GB_2.csv"

# Run Grid Search with expanded hyperparameter grid
print("\nPerforming Grid Search on Gradient Boosting:")
grid_search_gradient_boosting(output_dir_cubic, output_csv_cubic)
