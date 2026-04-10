import numpy as np
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from abqo import ABQO

# =====================================================================
# Real-World Problem #2: Optimal Feature Selection
# =====================================================================

print("Loading Wine Quality dataset...")
data = load_wine()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_features = X.shape[1]
print(f"Dataset Size: {X.shape[0]} samples, {num_features} total features.")

def feature_selection_objective(params):
    """
    ABQO provides parameters in continuous space [0, 1].
    We convert > 0.5 to 'select feature', < 0.5 to 'drop feature'.
    Fitness = Error + Penalty for using too many features.
    """
    # Map continuous positions to binary [0 or 1]
    binary_mask = (params > 0.5).astype(int)
    
    # If no features selected, heavily penalize
    if np.sum(binary_mask) == 0:
        return 1.0  # Max error
        
    X_subset = X_scaled[:, binary_mask == 1]
    
    # Fast lightweight KNN to test feature quality
    model = KNeighborsClassifier(n_neighbors=5)
    
    # Quick 3-fold cross-val
    scores = cross_val_score(model, X_subset, y, cv=3, n_jobs=-1)
    acc = np.mean(scores)
    
    # We want to maximize accuracy AND minimize the number of features used
    error = 1.0 - acc
    feature_penalty = 0.01 * (np.sum(binary_mask) / num_features)
    
    fitness = error + feature_penalty
    return fitness

# Setup ABQO
# Bounds: [0, 0, ..., 0] to [1, 1, ..., 1] (one for each feature)
lb = np.zeros(num_features)
ub = np.ones(num_features)
bounds = (lb, ub)

# Use a slightly smaller max_iter for quick demonstration
pop_size = 20
max_iter = 30 

print("\nStarting Advanced Biofilm-Quorum Optimization (ABQO)...")
print("Target: Find the minimalist subset of features that retains maximum predictive power.")
start_time = time.time()

optimizer = ABQO(feature_selection_objective, bounds=bounds, dim=num_features, pop_size=pop_size, max_iter=max_iter)
best_params, best_fit, history = optimizer.optimize()

end_time = time.time()

# Results Interpretation
binary_mask = (best_params > 0.5).astype(int)
selected_features = [data.feature_names[i] for i in range(num_features) if binary_mask[i] == 1]
num_selected = len(selected_features)

# Calculate final stand-alone accuracy with just those features
X_subset = X_scaled[:, binary_mask == 1]
X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print("\n================================================")
print("ABQO Feature Selection Complete!")
print(f"Time Taken: {end_time - start_time:.2f} seconds")
print(f"Features Reduced From: {num_features} -> {num_selected}")
print(f"Selected Features: {', '.join(selected_features)}")
print(f"Final Test Set Accuracy on reduced data: {test_accuracy * 100:.2f}%")
print("================================================")
