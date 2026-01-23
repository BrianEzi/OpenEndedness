import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import os

def plot_results(metrics, eval_data=None, save_dir="."):
    """
    metrics: dict of arrays or DataFrame
    eval_data: dict containing 'grid_obs' (N, 2) and 'messages' (N, 5) from evaluation
    """
    df = pd.DataFrame(metrics)
    
    # 1. Learning Curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Updates')
    ax1.set_ylabel('Mean Reward', color=color)
    ax1.plot(df.index, df['reward_mean'], color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Success Rate', color=color)
    ax2.plot(df.index, df['success_rate'], color=color, label='Success Rate', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Learning Curve: Reward & Success Rate')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    plt.close()
    
    # 2. Communication Pulse
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['msg_magnitude'], label='Msg Norm', color='purple')
    plt.xlabel('Updates')
    plt.ylabel('Average Message L2 Norm')
    plt.title('Communication Pulse')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'communication_pulse.png'))
    plt.close()
    
    # 3. Grounding Heatmap
    if eval_data and 'grid_pos' in eval_data and 'messages' in eval_data:
        positions = eval_data['grid_pos'] # (N, 2)
        messages = eval_data['messages'] # (N, 5)
        
        # PCA to 1D
        pca = PCA(n_components=1)
        msgs_pca = pca.fit_transform(messages) # (N, 1)
        
        # Grid Heatmap
        grid_size = 5
        heatmap_grid = np.zeros((grid_size, grid_size))
        count_grid = np.zeros((grid_size, grid_size))
        
        for pos, val in zip(positions, msgs_pca):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < grid_size and 0 <= y < grid_size:
                heatmap_grid[y, x] += val[0] # y is row, x is col
                count_grid[y, x] += 1
                
        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_grid = heatmap_grid / count_grid
            heatmap_grid = np.nan_to_num(heatmap_grid)
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_grid, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Seer Message Grounding (PCA 1st comp)')
        plt.xlabel('Doer X')
        plt.ylabel('Doer Y')
        plt.savefig(os.path.join(save_dir, 'grounding_heatmap.png'))
        plt.close()

    # 3.1. Relative Grounding Heatmap (New)
    # Checks if we have target positions for all steps to calculate relative vectors
    if eval_data and 'grid_pos' in eval_data and 'messages' in eval_data and 'target_pos_all' in eval_data:
        positions = eval_data['grid_pos']      # (N, 2)
        targets = eval_data['target_pos_all']  # (N, 2)
        messages = eval_data['messages']       # (N, 5)

        # PCA to 1D (Re-fit to capture variance specifically for this comparison)
        pca = PCA(n_components=1)
        msgs_pca = pca.fit_transform(messages)

        # Relative Grid ranges from -4 to +4 (dimensions: 9x9)
        rel_grid_size = 9
        offset = 4  # The center index (0,0 relative) is at index 4
        
        heatmap_rel = np.zeros((rel_grid_size, rel_grid_size))
        count_rel = np.zeros((rel_grid_size, rel_grid_size))

        for pos, tgt, val in zip(positions, targets, msgs_pca):
            # Calculate relative vector: Target - Doer
            rx = int(tgt[0] - pos[0])
            ry = int(tgt[1] - pos[1])
            
            # Map relative coordinates to array indices
            ix = rx + offset
            iy = ry + offset
            
            # Safety check to keep within bounds
            if 0 <= ix < rel_grid_size and 0 <= iy < rel_grid_size:
                heatmap_rel[iy, ix] += val[0] # y is row, x is col
                count_rel[iy, ix] += 1
        
        # Compute Average
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_rel = heatmap_rel / count_rel
            heatmap_rel = np.nan_to_num(heatmap_rel)

        plt.figure(figsize=(8, 6))
        # Create tick labels from -4 to 4
        tick_labels = np.arange(-offset, offset + 1)
        sns.heatmap(heatmap_rel, annot=True, cmap='coolwarm', fmt='.2f',
                    xticklabels=tick_labels, yticklabels=tick_labels)
        plt.title('Seer Message Relative Grounding (Target - Doer)')
        plt.xlabel('Relative X (Target - Doer)')
        plt.ylabel('Relative Y (Target - Doer)')
        plt.savefig(os.path.join(save_dir, 'relative_grounding_heatmap_comp1.png'))
        plt.close()
    
    # 3.2. Relative Grounding Heatmap (PCA Component 2)
    if eval_data and 'grid_pos' in eval_data and 'messages' in eval_data and 'target_pos_all' in eval_data:
        positions = eval_data['grid_pos']      # (N, 2)
        targets = eval_data['target_pos_all']  # (N, 2)
        messages = eval_data['messages']       # (N, 5)

        # PCA with 2 components to capture the second dimension of variance
        pca = PCA(n_components=2)
        msgs_pca = pca.fit_transform(messages)
        
        # EXTRACT COMPONENT 2 (Index 1)
        # We expect this to correlate with the Y-axis (Up/Down)
        comp2_vals = msgs_pca[:, 1]

        # Relative Grid ranges from -4 to +4 (dimensions: 9x9)
        rel_grid_size = 9
        offset = 4 
        
        heatmap_rel_c2 = np.zeros((rel_grid_size, rel_grid_size))
        count_rel_c2 = np.zeros((rel_grid_size, rel_grid_size))

        for pos, tgt, val in zip(positions, targets, comp2_vals):
            rx = int(tgt[0] - pos[0])
            ry = int(tgt[1] - pos[1])
            
            ix = rx + offset
            iy = ry + offset
            
            if 0 <= ix < rel_grid_size and 0 <= iy < rel_grid_size:
                heatmap_rel_c2[iy, ix] += val
                count_rel_c2[iy, ix] += 1
        
        # Compute Average
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_rel_c2 = heatmap_rel_c2 / count_rel_c2
            heatmap_rel_c2 = np.nan_to_num(heatmap_rel_c2)

        plt.figure(figsize=(8, 6))
        tick_labels = np.arange(-offset, offset + 1)
        
        # We use a different colormap (e.g., 'PiYG') to distinguish it visually from Component 1
        sns.heatmap(heatmap_rel_c2, annot=True, cmap='PiYG', fmt='.2f',
                    xticklabels=tick_labels, yticklabels=tick_labels)
        
        plt.title('Seer Message Grounding (PCA Component 2)')
        plt.xlabel('Relative X (Target - Doer)')
        plt.ylabel('Relative Y (Target - Doer)')
        plt.savefig(os.path.join(save_dir, 'relative_grounding_heatmap_comp2.png'))
        plt.close()

    # 4. Trajectory Trace (Single Episode)
    if eval_data and 'trajectory' in eval_data:
        traj = eval_data['trajectory'] # List of (x, y) tuples
        target = eval_data.get('target_pos', [4, 4])
        
        grid_size = 5
        plt.figure(figsize=(6, 6))
        plt.xlim(-0.5, grid_size - 0.5)
        plt.ylim(grid_size - 0.5, -0.5) # Invert y to match matrix coords
        plt.grid(True)
        
        # Plot path
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        plt.plot(xs, ys, marker='o', linestyle='-', color='blue', alpha=0.6, label='Path')
        
        # Start
        plt.plot(xs[0], ys[0], marker='^', color='green', markersize=12, label='Start')
        
        # Target
        plt.plot(target[0], target[1], marker='*', color='gold', markersize=15, label='Target')
        
        plt.legend()
        plt.title('Doer Trajectory')
        plt.savefig(os.path.join(save_dir, 'trajectory_trace.png'))
        plt.close()

    print(f"Analysis plots saved to {os.path.abspath(save_dir)}")
