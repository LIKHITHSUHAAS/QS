import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from abqo import ABQO

# =====================================================================
# Real-World Problem: Hyperparameter Tuning for Breast Cancer Diagnosis
# =====================================================================

# 1. Load and prepare dataset
print("Loading Breast Cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features (crucial for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Dataset Size: {X.shape[0]} samples, {X.shape[1]} features.")

# 2. Define the objective function for ABQO
# We want to tune C and gamma for an SVM with RBF kernel.
# Search Space:
# C: [0.1, 100]
# gamma: [0.0001, 1]

def svm_objective(params):
    """
    Evaluates SVM with given parameters using 3-fold cross validation.
    Returns: 1 - accuracy (because ABQO minimizes the objective)
    """
    # Exponentiate parameters since they are often searched in log space
    # but we will just map [0,1] domain to the actual hyperparameters
    # X[0] will map to C [0.1, 100]
    # X[1] will map to gamma [0.0001, 1]
    
    C_val = np.clip(params[0], 0.1, 100.0)
    gamma_val = np.clip(params[1], 0.0001, 1.0)
    
    model = SVC(kernel='rbf', C=C_val, gamma=gamma_val, random_state=42)
    
    # Use cross-validation accuracy as the fitness score
    scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)
    acc = np.mean(scores)
    
    # We want to MINIMIZE the error
    error = 1.0 - acc
    return error

# 3. Setup and Run ABQO
# Bounds: [C_lower, gamma_lower], [C_upper, gamma_upper]
bounds = (np.array([0.1, 0.0001]), np.array([100.0, 1.0]))
dim = 2
pop_size = 30
max_iter = 50 

print("\nStarting Advanced Biofilm-Quorum Optimization (ABQO)...")
print("Tuning SVM Hyperparameters (C and gamma)...")
start_time = time.time()

optimizer = ABQO(svm_objective, bounds=bounds, dim=dim, pop_size=pop_size, max_iter=max_iter)
best_params, best_error, history = optimizer.optimize()

end_time = time.time()

# 4. Results
best_C = np.clip(best_params[0], bounds[0][0], bounds[1][0])
best_gamma = np.clip(best_params[1], bounds[0][1], bounds[1][1])
best_cv_accuracy = 1.0 - best_error

print("\n================================================")
print("ABQO Optimization Complete!")
print(f"Time Taken: {end_time - start_time:.2f} seconds")
print(f"Best Hyperparameters Found:")
print(f"  C:     {best_C:.4f}")
print(f"  gamma: {best_gamma:.6f}")
print(f"Best Cross-Validation Accuracy: {best_cv_accuracy * 100:.2f}%")

# 5. Final Test on Hold-out Set
final_model = SVC(kernel='rbf', C=best_C, gamma=best_gamma, random_state=42)
final_model.fit(X_train, y_train)
test_accuracy = final_model.score(X_test, y_test)

print(f"Final Test Set Accuracy:        {test_accuracy * 100:.2f}%")
print("================================================")

