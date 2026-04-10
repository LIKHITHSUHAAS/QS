import numpy as np

class ABQO:
    """
    Advanced Biofilm-Quorum Optimization (ABQO) Algorithm
    
    A novel, state-driven metaheuristic optimization algorithm inspired by
    the physical biofilm lifecycle of bacteria, mediated by Quorum Sensing (QS).
    """
    
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=1000):
        """
        Initialize the ABQO optimizer.
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # --- Biological / Algorithmic Parameters ---
        # Chemotaxis phase (Exploration)
        self.step_size_base = 0.1 * (bounds[1] - bounds[0]) # Initial tumble step size
        self.chemotaxis_decay = 0.99 # Step size decay
        
        # Quorum & Biofilm Parameters (Exploitation)
        self.quorum_threshold = 0.6  # Normalized AI concentration needed to form biofilm
        self.biofilm_attraction = 1.5 # Attraction strength to the biofilm center
        self.biofilm_mutation_rate = 0.01 * (bounds[1] - bounds[0]) # Fine tuning step size
        
        # Dispersion Parameters (Escape Local Optima)
        self.max_stagnation = 20 # Iterations without improvement before dispersion
        
        # Optimization Tracking
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_fitness = np.inf
        
        self.history = {
            "gbest_fitness": [],
            "planktonic_count": [],
            "biofilm_count": [],
            "positions": [],
            "states": []
        }
    
    def _evaluate_fitness(self, population):
        """Evaluate the objective function for the whole population."""
        return np.array([self.objective_func(ind) for ind in population])
        
    def _calculate_ai_concentration(self, population, fitness, fitness_max, fitness_min):
        """
        Calculate Local Autoinducer (AI) concentration for each bacterium.
        AI is emitted based on high relative fitness.
        """
        # Normalize fitness to [0, 1], where 1 is the BEST (lowest fitness value)
        if fitness_max == fitness_min:
            normalized_fitness = np.zeros(self.pop_size)
        else:
            # Maximization of AI emission for minimization problem
            normalized_fitness = (fitness_max - fitness) / (fitness_max - fitness_min)
            
        # AI Concentration is a mix of personal emission and local diffusion.
        # For computational efficiency, we approximate local diffusion by 
        # finding the concentration in the neighborhood.
        ai_concentration = np.zeros(self.pop_size)
        
        for i in range(self.pop_size):
            # Calculate distance from bacterium i to all others
            distances = np.linalg.norm(population - population[i], axis=1)
            
            # Simulated diffusion: AI strength decays exponentially with distance
            # Diffusion factor alpha determines how fast the signal drops off over distance
            alpha = 1.0 / (np.mean(bounds_range) * 0.1 + 1e-8)
            diffusion_weights = np.exp(-alpha * distances)
            
            # Total local AI is the weighted sum of nearby emissions
            ai_concentration[i] = np.sum(normalized_fitness * diffusion_weights) / self.pop_size
            
        # Normalize concentration to [0, 1] for thresholding
        if np.max(ai_concentration) > 0:
            ai_concentration = ai_concentration / np.max(ai_concentration)
            
        return ai_concentration

    def optimize(self):
        """Run the ABQO algorithm."""
        # 1. Initialization
        lb, ub = self.bounds
        global bounds_range
        bounds_range = np.array(ub) - np.array(lb) if isinstance(ub, (list, tuple, np.ndarray)) else ub - lb
        
        # Initialize population matrices
        X = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = self._evaluate_fitness(X)
        
        # State Arrays
        # 0 = Planktonic (Exploration), 1 = Biofilm (Exploitation)
        states = np.zeros(self.pop_size, dtype=int) 
        
        # Personal best tracking for dispersion mechanics
        pbest_X = np.copy(X)
        pbest_fitness = np.copy(fitness)
        stagnation = np.zeros(self.pop_size, dtype=int)
        
        # Best global tracking
        best_idx = np.argmin(fitness)
        self.gbest_pos = np.copy(X[best_idx])
        self.gbest_fitness = fitness[best_idx]
        
        step_size = self.step_size_base
        
        # Main Loop
        for iteration in range(self.max_iter):
            fitness_max = np.max(fitness)
            fitness_min = np.min(fitness)
            
            # Step 1: Quorum Sensing (Determine AI concentrations)
            ai_concentration = self._calculate_ai_concentration(X, fitness, fitness_max, fitness_min)
            
            # Step 2: State Transitions
            for i in range(self.pop_size):
                if states[i] == 0:  # Planktonic Phase
                    if ai_concentration[i] >= self.quorum_threshold:
                        # Reached quorum, transition to Biofilm
                        states[i] = 1
                else:  # Biofilm Phase
                    if stagnation[i] >= self.max_stagnation:
                        # Biofilm stagnated (depleted nutrients/optimum reached) -> DISPERSE!
                        states[i] = 0
                        stagnation[i] = 0
                        # Explosion jump! Relocate arbitrarily far
                        X[i] = np.random.uniform(lb, ub, self.dim)
                        fitness[i] = self.objective_func(X[i])

            # Step 3: Movement based on State
            for i in range(self.pop_size):
                if states[i] == 0:
                    # Planktonic Phase: Run and Tumble (Random Chemotaxis)
                    # Random direction vector
                    direction = np.random.uniform(-1, 1, self.dim)
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    
                    # Tumble to new position
                    new_pos = X[i] + step_size * direction
                    
                else:
                    # Biofilm Phase: Exploit local optimum
                    # Find the physical center of mass of the current biofilm cluster
                    # To approximate fast, we attract towards the best known local position 
                    # bounded by a tiny Gaussian mutation
                    
                    # Standard PSO uses pure attraction. Biology uses extracellular matrix crawling.
                    # We model this as a tight random walk around the best position found by this agent.
                    r1 = np.random.uniform(0, 1, self.dim)
                    attraction = self.biofilm_attraction * r1 * (self.gbest_pos - X[i])
                    mutation = np.random.normal(0, self.biofilm_mutation_rate, self.dim)
                    
                    new_pos = X[i] + attraction + mutation

                # Boundary Constraints
                new_pos = np.clip(new_pos, lb, ub)
                
                # Evaluate and greedy selection (Bacteria only move if nutrients are better/same)
                # Actually, planktonic bacteria tumble blindly, we allow worse moves in planktonic
                # to escape flat zones, but biofilm strictly accepts better/same.
                new_fitness = self.objective_func(new_pos)
                
                if states[i] == 1:
                    # Biofilm: Strictly greedy
                    if new_fitness < fitness[i]:
                        X[i] = new_pos
                        fitness[i] = new_fitness
                else:
                    # Planktonic: accept move
                    X[i] = new_pos
                    fitness[i] = new_fitness

            # Update PBest, Stagnation, and GBest
            for i in range(self.pop_size):
                if fitness[i] < pbest_fitness[i]:
                    pbest_fitness[i] = fitness[i]
                    pbest_X[i] = X[i]
                    stagnation[i] = 0 
                else:
                    stagnation[i] += 1
                    
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.gbest_fitness:
                self.gbest_fitness = fitness[best_idx]
                self.gbest_pos = np.copy(X[best_idx])
                
            # Decay exploration step size like temperature cooling
            step_size *= self.chemotaxis_decay

            # Logging
            planktonic_cnt = np.sum(states == 0)
            biofilm_cnt = np.sum(states == 1)
            self.history["gbest_fitness"].append(self.gbest_fitness)
            self.history["planktonic_count"].append(planktonic_cnt)
            self.history["biofilm_count"].append(biofilm_cnt)
            
            # Save 2D slice of positions and state for Live Animation
            self.history["positions"].append(np.copy(X[:, :2]))
            self.history["states"].append(np.copy(states))
            
            if iteration % 100 == 0:
                print(f"Iter {iteration:4d} | Best Fitness: {self.gbest_fitness:.6e} | Planktonic: {planktonic_cnt} | Biofilm: {biofilm_cnt}")
                
        return self.gbest_pos, self.gbest_fitness, self.history


# =====================================================================
# Benchmark Functions
# =====================================================================

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

# Example usage/tester
if __name__ == "__main__":
    print("Testing Advanced Biofilm-Quorum Optimization (ABQO)")
    print("--------------------------------------------------")
    
    dim = 10
    num_iterations = 500
    
    print("\n1. Optimizing Sphere Function (Unimodal)")
    optimizer_sphere = ABQO(sphere, bounds=(-5.12, 5.12), dim=dim, max_iter=num_iterations)
    best_pos, best_fit, hist = optimizer_sphere.optimize()
    print(f"-> Sphere Final Best Fitness: {best_fit:.6e}")
    
    print("\n2. Optimizing Rastrigin Function (Multimodal)")
    optimizer_rast = ABQO(rastrigin, bounds=(-5.12, 5.12), dim=dim, max_iter=num_iterations)
    best_pos_r, best_fit_r, hist_r = optimizer_rast.optimize()
    print(f"-> Rastrigin Final Best Fitness: {best_fit_r:.6e}")
