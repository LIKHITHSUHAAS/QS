import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from abqo import ABQO, sphere, rastrigin, ackley

# --- Streamlit Page Config ---
st.set_page_config(page_title="ABQO Dashboard", page_icon="🦠", layout="wide")

st.title("🦠 ABQO: Advanced Biofilm-Quorum Optimization")

# --- UI Application Mode ---
app_mode = st.sidebar.radio("Dashboard Mode", ["Theoretical Benchmarks", "Real-World AI Applications"])
st.sidebar.divider()

if app_mode == "Theoretical Benchmarks":
    st.markdown("""
    Welcome to the interactive demonstration for the ABQO Algorithm! 
    This bio-inspired metaheuristic models the Planktonic-Biofilm lifecycle of bacteria mediated by Quorum Sensing.
    """)

    with st.expander("🔭 Benchmark Function Landscapes", expanded=False):
        st.markdown("Algorithms perform like blindfolded hikers trying to find the lowest valley. Here are 3D terrains they must navigate:")
        col_a, col_b, col_c = st.columns(3)

        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111, projection='3d')
        X1 = np.linspace(-5.12, 5.12, 30); Y1 = np.linspace(-5.12, 5.12, 30)
        X1, Y1 = np.meshgrid(X1, Y1)
        Z1 = X1**2 + Y1**2
        ax1.plot_surface(X1, Y1, Z1, cmap='Blues', edgecolor='none')
        ax1.set_title("Sphere (Unimodal)")
        ax1.axis('off')
        col_a.pyplot(fig1)

        fig2 = plt.figure(figsize=(3, 3))
        ax2 = fig2.add_subplot(111, projection='3d')
        Z2 = 20 + (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (Y1**2 - 10 * np.cos(2 * np.pi * Y1))
        ax2.plot_surface(X1, Y1, Z2, cmap='plasma', edgecolor='none')
        ax2.set_title("Rastrigin (Multimodal)")
        ax2.axis('off')
        col_b.pyplot(fig2)

        fig3 = plt.figure(figsize=(3, 3))
        ax3 = fig3.add_subplot(111, projection='3d')
        X3, Y3 = np.meshgrid(np.linspace(-32.0, 32.0, 30), np.linspace(-32.0, 32.0, 30))
        Z3 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X3**2 + Y3**2))) - np.exp(0.5 * (np.cos(2*np.pi*X3) + np.cos(2*np.pi*Y3))) + 20 + np.e
        ax3.plot_surface(X3, Y3, Z3, cmap='viridis', edgecolor='none')
        ax3.set_title("Ackley (Sinkhole)")
        ax3.axis('off')
        col_c.pyplot(fig3)

    # --- Sidebar Configuration ---
    st.sidebar.header("Algorithmic Settings")

    function_selection = st.sidebar.selectbox("Benchmark Function", ["Sphere", "Rastrigin", "Ackley"])
    func_map = {"Sphere": (sphere, (-5.12, 5.12)), "Rastrigin": (rastrigin, (-5.12, 5.12)), "Ackley": (ackley, (-32.0, 32.0))}
    selected_func, bounds = func_map[function_selection]

    dimension = st.sidebar.number_input("Dimensions (D)", min_value=2, max_value=50, value=2, step=1)
    pop_size = st.sidebar.number_input("Population Size (Swarm)", min_value=10, max_value=500, value=40, step=10)
    max_iter = st.sidebar.number_input("Max Iterations", min_value=10, max_value=1000, value=150, step=10)

    st.divider()

    if st.button("🚀 Launch ABQO Swarm", type="primary"):
        with st.spinner(f"Running Swarm of {pop_size} bacteria in {dimension}D space..."):
            start_time = time.time()
            optimizer = ABQO(selected_func, bounds=bounds, dim=dimension, pop_size=pop_size, max_iter=max_iter)
            best_pos, best_fit, history = optimizer.optimize()
            end_time = time.time()
            
        st.success(f"Algorithm Converged in {end_time - start_time:.2f} seconds!")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Final Best Fitness", f"{best_fit:.6e}")
        m2.metric("Total Iterations", max_iter)
        m3.metric("Final Population State", f"{(history['biofilm_count'][-1] / pop_size) * 100:.0f}% Biofilm")
        
        # Contour generation
        x_ls = np.linspace(bounds[0], bounds[1], 50)
        y_ls = np.linspace(bounds[0], bounds[1], 50)
        X_mesh, Y_mesh = np.meshgrid(x_ls, y_ls)
        Z = np.zeros_like(X_mesh)
        for i in range(50):
            for j in range(50):
                pt = np.zeros(dimension); pt[0] = X_mesh[i, j]; pt[1] = Y_mesh[i, j]
                Z[i, j] = selected_func(pt)
                
        # Animation Frame Data Construction (Optimized for speed)
        frames = []
        step = max(1, max_iter // 30) 
        iter_list = list(range(0, max_iter, step))
        if (max_iter - 1) not in iter_list: iter_list.append(max_iter - 1)

        for i in iter_list:
            pos_frame = history["positions"][i]
            state_frame = history["states"][i]
            for idx in range(pop_size):
                frames.append({"Iteration": i, "ID": idx, "X": pos_frame[idx, 0], "Y": pos_frame[idx, 1],
                    "State": 'Planktonic' if state_frame[idx] == 0 else 'Biofilm', "Size": 12 if state_frame[idx] == 0 else 22})
        
        fig_anim = px.scatter(pd.DataFrame(frames), x="X", y="Y", animation_frame="Iteration", animation_group="ID",
            color="State", size="Size", color_discrete_map={'Planktonic': '#3b82f6', 'Biofilm': '#22c55e'},
            range_x=[bounds[0], bounds[1]], range_y=[bounds[0], bounds[1]], height=600)
        
        cs = 'Blues' if function_selection == "Sphere" else ('plasma' if function_selection == "Rastrigin" else 'viridis')
        fig_anim.add_trace(go.Contour(z=Z, x=x_ls, y=y_ls, colorscale=cs, opacity=0.6, showscale=False, hoverinfo='skip'))
        fig_anim.data = tuple([fig_anim.data[-1]] + list(fig_anim.data[:-1]))
        fig_anim.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor='rgba(0,0,0,0.0)', margin=dict(l=0, r=0, t=10, b=0))
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 150
        
        st.subheader(f"🦠 Live Swarm Animation over {function_selection} Topology")
        st.plotly_chart(fig_anim, use_container_width=True)

elif app_mode == "Real-World AI Applications":
    st.markdown("Here we apply the mathematical power of Quorum Sensing to actively tackle **Machine Learning engineering problems**.")
    
    app_choice = st.sidebar.selectbox("Application Task", ["SVM Hyperparameter Tuning", "Optimal Feature Selection"])
    st.sidebar.divider()
    
    if app_choice == "SVM Hyperparameter Tuning":
        st.subheader("Neural Tuning: Search the Hidden Accuracy Hyper-Plane")
        st.markdown("We are finding the optimal **C** and **Gamma** weights for a Breast Cancer Prediction AI. The swarm generates an invisible map of 'High Accuracy' and explores it.")
        
        if st.button("Launch SVM Tune Simulation", type='primary'):
            # Load Data
            data = load_breast_cancer()
            X, y = StandardScaler().fit_transform(data.data), data.target
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            
            def svm_obj(params):
                C_v = np.clip(params[0], 0.1, 100.0)
                g_v = np.clip(params[1], 0.0001, 1.0)
                scores = cross_val_score(SVC(kernel='rbf', C=C_v, gamma=g_v, random_state=42), X_tr, y_tr, cv=3, n_jobs=-1)
                return 1.0 - np.mean(scores)
            
            # Fast Contour pre-mapping (approx)
            with st.spinner("Pre-calculating underlying Accuracy Map (This takes ~3 seconds)..."):
                bounds = (np.array([0.1, 0.0001]), np.array([100.0, 1.0]))
                c_vals = np.linspace(0.1, 100, 10); g_vals = np.linspace(0.0001, 1.0, 10)
                Zm = np.zeros((10, 10))
                for i, C in enumerate(c_vals):
                    for j, g in enumerate(g_vals):
                        Zm[j, i] = 1.0 - svm_obj([C, g]) # Z is accuracy now
            
            with st.spinner("Swarm is hunting for hyperparameters..."):
                start_time = time.time()
                opt = ABQO(svm_obj, bounds=bounds, dim=2, pop_size=15, max_iter=30)
                best, err, hist = opt.optimize()
                end_time = time.time()
                
            best_acc = (1.0 - err) * 100
            st.success(f"Tuning Finished in {end_time - start_time:.2f} seconds!")
            st.metric("Optimal Validation Accuracy Found", f"{best_acc:.2f}%")
            st.write(f"Best Params Found - C: {best[0]:.2f}, Gamma: {best[1]:.4f}")
            
            st.subheader("Visualizing the Swarm Over the Accuracy Heatmap")
            
            frames = []
            for it in range(30):
                for p in range(15):
                    frames.append({"Iter": it, "Bact": p, "C": hist["positions"][it][p, 0], 
                                   "Gamma": hist["positions"][it][p, 1], 
                                   "State": 'Planktonic' if hist["states"][it][p]==0 else 'Biofilm',
                                   "Size": 18})
            
            fig = px.scatter(pd.DataFrame(frames), x="C", y="Gamma", animation_frame="Iter", animation_group="Bact",
                             color="State", size="Size", color_discrete_map={'Planktonic': '#3b82f6', 'Biofilm': '#22c55e'})
            # Background is Accuracy Heatmap
            fig.add_trace(go.Contour(x=c_vals, y=g_vals, z=Zm, colorscale='RdYlGn', opacity=0.7, colorbar=dict(title='Local Accuracy')))
            fig.data = tuple([fig.data[-1]] + list(fig.data[:-1]))
            st.plotly_chart(fig, use_container_width=True)

    elif app_choice == "Optimal Feature Selection":
        st.subheader("Data Pruning: Eliminate Statistical Noise")
        st.markdown("We feed ABQO a high-dimensional Wine Quality Dataset. The bacteria will eliminate useless classification features, drastically shrinking the dataset size while protecting system accuracy.")
        
        if st.button("Launch Pruning Swarm", type='primary'):
            data = load_wine()
            X, y = StandardScaler().fit_transform(data.data), data.target
            num_feat = X.shape[1]
            
            def feat_obj(params):
                mask = (params > 0.5).astype(int)
                if np.sum(mask) == 0: return 1.0
                scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, mask==1], y, cv=3, n_jobs=-1)
                return (1.0 - np.mean(scores)) + 0.01 * (np.sum(mask)/num_feat)
            
            with st.spinner("Bacteria Swarm is analyzing dataset dimension values..."):
                t0 = time.time()
                opt = ABQO(feat_obj, bounds=(np.zeros(num_feat), np.ones(num_feat)), dim=num_feat, pop_size=20, max_iter=25)
                b, er, h = opt.optimize()
                t1 = time.time()
                
            mask = (b > 0.5).astype(int)
            sel_feats = [data.feature_names[i] for i in range(num_feat) if mask[i] == 1]
            
            st.success(f"Reduced Dimensions from 13 down to {len(sel_feats)} in {t1 - t0:.2f} seconds!")
            
            # --- Feature Matrix Scanner Visualization ---
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Feature Assessment Heatmap**")
                feat_df = pd.DataFrame({"Feature": data.feature_names, "Status": ["Selected" if mask[i]==1 else "Dropped" for i in range(num_feat)]})
                fig_f = px.bar(feat_df, y="Feature", x=[1]*num_feat, color="Status", color_discrete_map={"Selected":"#22c55e", "Dropped":"#64748b"}, orientation='h')
                st.plotly_chart(fig_f, use_container_width=True)
            
            # --- PCA Visualization ---
            with c2:
                st.write("**Data Integrity (PCA)**")
                # PCA on all features vs subset
                pca_all = PCA(n_components=2).fit_transform(X)
                pca_sub = PCA(n_components=2).fit_transform(X[:, mask==1])
                
                fig_pca = go.Figure()
                fig_pca.add_trace(go.Scatter(x=pca_all[:,0], y=pca_all[:,1], mode='markers', marker=dict(color=y, opacity=0.3, symbol='cross'), name='Original (13 Feat)'))
                fig_pca.add_trace(go.Scatter(x=pca_sub[:,0], y=pca_sub[:,1], mode='markers', marker=dict(color=y, opacity=1.0, size=8, line=dict(width=1, color='DarkSlateGrey')), name='Optimized subset'))
                fig_pca.update_layout(title="Dataset Splitting Separation Before/After", plot_bgcolor='rgba(0,0,0,0.05)')
                st.plotly_chart(fig_pca, use_container_width=True)
