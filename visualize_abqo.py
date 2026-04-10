import numpy as np
import matplotlib.pyplot as plt
from abqo import ABQO, rastrigin

# Run algorithm on 20 dimensions for 500 iterations
dim = 20
num_iterations = 500

print("Running ABQO for visualization...")
optimizer = ABQO(rastrigin, bounds=(-5.12, 5.12), dim=dim, max_iter=num_iterations, pop_size=60)
best_pos, best_fit, hist = optimizer.optimize()

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Global Best Fitness (Log Scale)', color=color)
ax1.plot(hist['gbest_fitness'], color=color, linewidth=2, label='Best Fitness')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')

ax2 = ax1.twinx()  
color_planktonic = 'tab:blue'
color_biofilm = 'tab:green'
ax2.set_ylabel('Population Count', color='black')  
ax2.plot(hist['planktonic_count'], color=color_planktonic, linestyle='--', alpha=0.7, label='Planktonic (Explorers)')
ax2.plot(hist['biofilm_count'], color=color_biofilm, linestyle=':', alpha=0.9, label='Biofilm (Exploiters)')
ax2.tick_params(axis='y', labelcolor='black')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('ABQO Convergence & Population State Dynamics (Rastrigin 20D)')
plt.tight_layout()

output_path = r'C:\Users\likhi\.gemini\antigravity\brain\5664c87c-da51-4cb0-87f0-438bdc388620\abqo_output.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")

