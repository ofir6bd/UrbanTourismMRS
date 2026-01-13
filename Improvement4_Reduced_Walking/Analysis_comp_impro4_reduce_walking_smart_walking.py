import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def load_and_compare_rome_walking_results():
    """
    Load and compare Rome experimental results between base and reduce walking versions
    Apply distance normalization to user satisfaction, filter for win-win points after correction
    """
    
    city = "Rome"
    
    # Define file pairs for comparison
    file_pairs = {
        'EASE_WMF': {
            'base': f"All_pk_conparison/runx_{city}_ease_wmf_4_reduce_walking_base.pk",
            'reduce_walking': f"All_pk_conparison/runx_{city}_ease_wmf_4_reduce_walking_with_routing.pk"
        },
        'EASE_VAE': {
            'base': f"All_pk_conparison/runx_{city}_ease_vae_4_reduce_walking_base.pk", 
            'reduce_walking': f"All_pk_conparison/runx_{city}_ease_vae_4_reduce_walking_with_routing.pk"
        }
    }
    
    # Colors for each version
    colors = {
        'base': 'red',
        'reduce_walking': 'blue'
    }
    
    all_stats = {}
    raw_data = {'base': {'tau': [], 'eta': [], 'distances': []}, 
                'reduce_walking': {'tau': [], 'eta': [], 'distances': []}}
    
    # First pass: collect all points (not filtering yet)
    for model_name, files in file_pairs.items():
        for version, file_path in files.items():
            try:
                with open(file_path, "rb") as file:
                    experiment = pickle.load(file)
                
                version_tau = []
                version_eta = []
                version_distances = []
                
                # Extract data for all t and k combinations
                for t in [2, 3, 4]:
                    for k in [4, 8, 12, 16, 20]:
                        if t in experiment and k in experiment[t]:
                            source = experiment[t][k]
                            tau = source["improvement_user"]
                            eta = source["improvement_harm"]
                            
                            # Process each lambda experiment - collect all points initially
                            for i, (tau_val, eta_val) in enumerate(zip(tau, eta)):
                                version_tau.append(tau_val)
                                version_eta.append(eta_val)
                                
                                # Extract distance for this experiment
                                if i < len(source['hist']) and 'void_total_distance' in source['hist'][i]:
                                    avg_distance = np.mean(source['hist'][i]['void_total_distance'])
                                    version_distances.append(avg_distance)
                                else:
                                    version_distances.append(0)  # fallback
                
                raw_data[version]['tau'].extend(version_tau)
                raw_data[version]['eta'].extend(version_eta)
                raw_data[version]['distances'].extend(version_distances)
                
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Find global min/max distances across both versions for normalization
    all_distances = []
    for version in ['base', 'reduce_walking']:
        all_distances.extend(raw_data[version]['distances'])
    
    if len(all_distances) > 0:
        min_distance = np.min(all_distances)
        max_distance = np.max(all_distances)
        distance_range = max_distance - min_distance
        print(f"Distance range (all points): {min_distance:.2f} - {max_distance:.2f} km")
    else:
        print("No distance data found!")
        min_distance = max_distance = distance_range = 0
    
    # Process each version: apply correction and filter for win-win points
    processed_data = {'base': {'tau_corrected': [], 'eta': [], 'distances': []}, 
                     'reduce_walking': {'tau_corrected': [], 'eta': [], 'distances': []}}
    
    for version in ['base', 'reduce_walking']:
        tau_data = np.array(raw_data[version]['tau'])
        eta_data = np.array(raw_data[version]['eta'])
        distances = np.array(raw_data[version]['distances'])
        
        if len(tau_data) > 0 and distance_range > 0:
            # Apply distance normalization: min distance = +0.1, max distance = -0.1
            normalized_distances = (distances - min_distance) / distance_range  # 0 to 1
            distance_bonus = (1 - normalized_distances) * 0.2 - 0.1  # flip and scale to -0.1 to +0.1
            
            # Add distance bonus to user satisfaction
            adjusted_tau = tau_data + distance_bonus
            
            # Filter for win-win points AFTER correction
            win_win_mask = (adjusted_tau > 0) & (eta_data > 0)
            
            processed_data[version]['tau_corrected'] = adjusted_tau[win_win_mask]
            processed_data[version]['eta'] = eta_data[win_win_mask]
            processed_data[version]['distances'] = distances[win_win_mask]
            
            # Calculate statistics with corrected and filtered data
            positive_count = np.sum(win_win_mask)
            tau_aggregate = np.sum(adjusted_tau[win_win_mask])
            eta_aggregate = np.sum(eta_data[win_win_mask])
            total_aggregate = tau_aggregate + eta_aggregate
            
            all_stats[version] = {
                'positive_count': positive_count,
                'tau_aggregate': tau_aggregate,
                'eta_aggregate': eta_aggregate,
                'total_aggregate': total_aggregate,
                'tau_avg': np.mean(adjusted_tau[win_win_mask]) if positive_count > 0 else 0,
                'eta_avg': np.mean(eta_data[win_win_mask]) if positive_count > 0 else 0,
                'distance_avg': np.mean(distances[win_win_mask]) if positive_count > 0 else 0,
                'distance_bonus_avg': np.mean(distance_bonus[win_win_mask]) if positive_count > 0 else 0,
                'distance_bonus_total': np.sum(distance_bonus[win_win_mask])
            }
            
        else:
            # No data case
            processed_data[version]['tau_corrected'] = np.array([])
            processed_data[version]['eta'] = np.array([])
            processed_data[version]['distances'] = np.array([])
            
            all_stats[version] = {
                'positive_count': 0,
                'tau_aggregate': 0,
                'eta_aggregate': 0,
                'total_aggregate': 0,
                'tau_avg': 0,
                'eta_avg': 0,
                'distance_avg': 0,
                'distance_bonus_avg': 0,
                'distance_bonus_total': 0
            }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Rome Results: Base vs Reduce Walking & Smart routing', 
                 fontsize=16, fontweight='bold')
    
    # Process each version and create plots
    axes = [ax1, ax2]
    version_names = ['base', 'reduce_walking']
    version_labels = ['Base', 'Reduce Walking']
    
    for idx, version in enumerate(version_names):
        ax = axes[idx]
        
        tau_corrected = processed_data[version]['tau_corrected']
        eta_data = processed_data[version]['eta']
        distances = processed_data[version]['distances']
        
        if len(tau_corrected) > 0:
            # Plot corrected win-win data
            scatter = ax.scatter(tau_corrected, eta_data, 
                               c=distances, cmap='viridis_r', 
                               s=30, alpha=0.8, 
                               edgecolors='black', linewidth=0.5)
            
            # Add colorbar for distances
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Avg Distance (km)', fontsize=10)
            
            # Set fixed axis limits
            ax.set_xlim(-0.2, 0.3)
            ax.set_ylim(-0.1, 0.2)
            
            # Shade the positive quadrant
            ax.axhspan(0, 0.2, xmin=0.4, xmax=1, alpha=0.12, color='green')
            
            # Draw quadrant lines
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Labels and title
            ax.set_xlabel('User uplift + Distance bonus', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dest uplift', fontsize=12, fontweight='bold')
            
            # Different titles for each version
            if version == 'reduce_walking':
                ax.set_title(f'{version_labels[idx]} & Smart Routing (Win-Win)', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'{version_labels[idx]} (Win-Win)', fontsize=14, fontweight='bold')
            
            # Add statistics text box
            stats = all_stats[version]
            stats_text = (f"Win-Win Points: {stats['positive_count']}\n"
                         f"Σ User+Dist: {stats['tau_aggregate']:.3f}\n"
                         f"Σ Dest: {stats['eta_aggregate']:.3f}\n"
                         f"Σ Total: {stats['total_aggregate']:.3f}\n"
                         f"Avg Distance: {stats['distance_avg']:.2f}km")
            
            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", 
                             facecolor=colors[version], alpha=0.8, edgecolor='black'),
                    fontsize=9, fontweight='bold', color='white',
                    verticalalignment='top')
            
    plt.tight_layout()
    plt.savefig(f'{city}_win_win_post_correction.png', dpi=300, bbox_inches='tight')
    plt.show()

    return all_stats

if __name__ == "__main__":
    results = load_and_compare_rome_walking_results()