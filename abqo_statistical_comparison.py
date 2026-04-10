import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from abqo import ABQO, rastrigin, sphere

# =====================================================================
# Statistical Comparison: ABQO vs Standard PSO
# =====================================================================

def standard_pso(func, bounds, dim, pop_size=50, max_iter=1000):
    """Standard Particle Swarm Optimization for Baseline Comparison."""
    lb, ub = bounds
    # Initialize
    pos = np.random.uniform(lb, ub, (pop_size, dim))
    vel = np.random.uniform(-0.1 * (ub-lb), 0.1 * (ub-lb), (pop_size, dim))
    
    pbest_pos = np.copy(pos)
    pbest_fit = np.array([func(ind) for ind in pos])
    
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = np.copy(pos[gbest_idx])
    gbest_fit = pbest_fit[gbest_idx]
    
    w, c1, c2 = 0.7, 1.5, 1.5
    history = []
    
    for _ in range(max_iter):
        r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
        
        # Velocity update
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = pos + vel
        pos = np.clip(pos, lb, ub)
        
        # Fitness evaluation
        fitness = np.array([func(ind) for ind in pos])
        
        # Pbest update
        improved = fitness < pbest_fit
        pbest_pos[improved] = pos[improved]
        pbest_fit[improved] = fitness[improved]
        
        # Gbest update
        current_gbest_idx = np.argmin(pbest_fit)
        if pbest_fit[current_gbest_idx] < gbest_fit:
            gbest_fit = pbest_fit[current_gbest_idx]
            gbest_pos = np.copy(pbest_pos[current_gbest_idx])
            
        history.append(gbest_fit)
        
    return gbest_pos, gbest_fit, history


# Experiment Setup
dim = 20
max_iter = 300
pop_size = 50
n_runs = 5 # 5 independent runs for statistical average

functions = {
    'Sphere': (sphere, (-5.12, 5.12)),
    'Rastrigin': (rastrigin, (-5.12, 5.12))
}

results_db = []
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

print("Running Statistical Comparison Engine...")
for idx, (func_name, (func, bounds)) in enumerate(functions.items()):
    print(f"\nEvaluating {func_name} Function ({dim}D)...")
    
    abqo_bests = []
    pso_bests = []
    
    abqo_histories = np.zeros((n_runs, max_iter))
    pso_histories = np.zeros((n_runs, max_iter))
    
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}...")
        # ABQO Run
        abqo = ABQO(func, bounds=bounds, dim=dim, pop_size=pop_size, max_iter=max_iter)
        _, abqo_best, abqo_hist = abqo.optimize()
        abqo_bests.append(abqo_best)
        abqo_histories[i] = abqo_hist['gbest_fitness']
        
        # PSO Run
        _, pso_best, pso_hist = standard_pso(func, bounds, dim, pop_size, max_iter)
        pso_bests.append(pso_best)
        pso_histories[i] = pso_hist
        
    # Metrics
    mean_abqo = np.mean(abqo_bests)
    std_abqo = np.std(abqo_bests)
    mean_pso = np.mean(pso_bests)
    std_pso = np.std(pso_bests)
    
    results_db.append({
        "Function": func_name, 
        "Algo": "ABQO", 
        "Mean Best": mean_abqo, 
        "Std": std_abqo
    })
    results_db.append({
        "Function": func_name, 
        "Algo": "Standard PSO", 
        "Mean Best": mean_pso, 
        "Std": std_pso
    })
    
    # Plotting Average Convergence curves
    ax = axes[idx]
    ax.plot(np.mean(abqo_histories, axis=0), color='green', linewidth=2.5, label='ABQO (Proposed)')
    ax.plot(np.mean(pso_histories, axis=0), color='red', linestyle='--', linewidth=2, label='Standard PSO')
    
    ax.set_yscale('log')
    ax.set_title(f'Mean Convergence: {func_name} ({dim}D)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Global Best Fitness (Log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = r'C:\Users\likhi\.gemini\antigravity\brain\5664c87c-da51-4cb0-87f0-438bdc388620\statistical_comparison.png'
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to {output_path}")

df = pd.DataFrame(results_db)
print("\n=== FINAL STATISTICAL TABLE ===")
print(df.to_markdown(index=False))
