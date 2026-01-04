import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def load_and_compare_rome_walking_results():
    """
    Load and compare Rome experimental results between base and reduce walking versions
    Apply distance normalization to both versions, filter win-win zone first
    """
    
    city = "Rome"
    
    # Define file pairs for comparison
    file_pairs = {
        'EASE_WMF': {
            'base': f"All_pk_conparison/runx_{city}_ease_wmf_4_reduce_walking_base.pk",
            # 'reduce_walking': f"All_pk_conparison/runx_{city}_ease_wmf_4_reduce_walking.pk"
            'reduce_walking': f"All_pk_conparison/runx_{city}_ease_wmf_4_reduce_walking_with_routing.pk"
            
        },
        'EASE_VAE': {
            'base': f"All_pk_conparison/runx_{city}_ease_vae_4_reduce_walking_base.pk", 
            # 'reduce_walking': f"All_pk_conparison/runx_{city}_ease_vae_4_reduce_walking.pk"
            'reduce_walking': f"All_pk_conparison/runx_{city}_ease_vae_4_reduce_walking_with_routing.pk"
        }
    }
    
    # Colors for each version
    colors = {
        'base': 'red',
        'reduce_walking': 'blue'
    }
    
    all_stats = {}
    processed_data = {'base': {'tau': [], 'eta': [], 'distances': []}, 
                     'reduce_walking': {'tau': [], 'eta': [], 'distances': []}}
    
    # First pass: collect all win-win points and their distances
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
                            
                            # Process each lambda experiment
                            for i, (tau_val, eta_val) in enumerate(zip(tau, eta)):
                                # Only keep win-win points
                                if tau_val > 0 and eta_val > 0:
                                    version_tau.append(tau_val)
                                    version_eta.append(eta_val)
                                    
                                    # Extract distance for this experiment
                                    if i < len(source['hist']) and 'void_total_distance' in source['hist'][i]:
                                        avg_distance = np.mean(source['hist'][i]['void_total_distance'])
                                        version_distances.append(avg_distance)
                                    else:
                                        version_distances.append(0)  # fallback
                
                processed_data[version]['tau'].extend(version_tau)
                processed_data[version]['eta'].extend(version_eta)
                processed_data[version]['distances'].extend(version_distances)
                
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Find global min/max distances across both versions for normalization
    all_distances = []
    for version in ['base', 'reduce_walking']:
        all_distances.extend(processed_data[version]['distances'])
    
    if len(all_distances) > 0:
        min_distance = np.min(all_distances)
        max_distance = np.max(all_distances)
        distance_range = max_distance - min_distance
        print(f"Distance range (win-win only): {min_distance:.2f} - {max_distance:.2f} km")
    else:
        print("No distance data found!")
        min_distance = max_distance = distance_range = 0
    
    # Create two separate figures
    
    # FIGURE 1: Base and Reduce Walking plots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig1.suptitle('Rome Results: Base vs Reduce Walking (Win-Win Zone + Distance Bonus)', 
                  fontsize=16, fontweight='bold')
    
    # Process each version and create plots
    axes = [ax1, ax2]
    version_names = ['base', 'reduce_walking']
    version_labels = ['Base', 'Reduce Walking']
    
    for idx, version in enumerate(version_names):
        ax = axes[idx]
        
        tau_data = np.array(processed_data[version]['tau'])
        eta_data = np.array(processed_data[version]['eta'])
        distances = np.array(processed_data[version]['distances'])
        
        if len(tau_data) > 0 and distance_range > 0:
            # Apply distance normalization: min distance = +0.1, max distance = -0.1
            normalized_distances = (distances - min_distance) / distance_range  # 0 to 1
            distance_bonus = (1 - normalized_distances) * 0.2 - 0.1  # flip and scale to -0.1 to +0.1
            
            # Add distance bonus to user satisfaction
            adjusted_tau = tau_data + distance_bonus
            
            # Plot adjusted data
            scatter = ax.scatter(adjusted_tau, eta_data, 
                               c=distances, cmap='viridis_r', 
                               s=30, alpha=0.8, 
                               edgecolors='black', linewidth=0.5)
            
            # Add colorbar for distances
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Avg Distance (km)', fontsize=10)
            
            # Calculate statistics
            positive_count = len(tau_data)  # All are win-win already
            tau_aggregate = np.sum(adjusted_tau)
            eta_aggregate = np.sum(eta_data)
            total_aggregate = tau_aggregate + eta_aggregate
            
            all_stats[version] = {
                'positive_count': positive_count,
                'tau_aggregate': tau_aggregate,
                'eta_aggregate': eta_aggregate,
                'total_aggregate': total_aggregate,
                'tau_avg': np.mean(adjusted_tau),
                'eta_avg': np.mean(eta_data),
                'distance_avg': np.mean(distances),
                'distance_bonus_avg': np.mean(distance_bonus)
            }
            
            # Set fixed axis limits
            ax.set_xlim(-0.2, 0.3)
            ax.set_ylim(-0.1, 0.2)
            
            # Shade the positive quadrant
            ax.axhspan(0, 0.2, xmin=0.4, xmax=1, 
                    alpha=0.12, color='green')
            
            # Draw quadrant lines
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Labels and title
            ax.set_xlabel('User uplift + Distance bonus', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dest uplift', fontsize=12, fontweight='bold')
            ax.set_title(f'{version_labels[idx]} (Win-Win + Distance + Routing)', fontsize=14, fontweight='bold')
            
            # Add statistics text box
            stats_text = (f"Win-Win Points: {positive_count}\n"
                         f"Σ User+Dist: {tau_aggregate:.3f}\n"
                         f"Σ Dest: {eta_aggregate:.3f}\n"
                         f"Σ Total: {total_aggregate:.3f}\n"
                         f"Avg Distance: {np.mean(distances):.2f}km\n"
                         f"Avg Bonus: {np.mean(distance_bonus):.3f}")
            
            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", 
                             facecolor=colors[version], alpha=0.8, edgecolor='black'),
                    fontsize=9, fontweight='bold', color='white',
                    verticalalignment='top')
            
        else:
            ax.text(0.5, 0.5, f'No win-win data\nfor {version_labels[idx]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{version_labels[idx]} (No Data)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{city}_base_vs_reduce_walking_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # FIGURE 2: Improvements Summary (separate window)
    fig2, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    fig2.suptitle('Improvements Analysis: Reduce Walking vs Base', fontsize=16, fontweight='bold')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')  # Remove axes for text-only display
    
    if 'base' in all_stats and 'reduce_walking' in all_stats:
        base_stats = all_stats['base']
        reduce_stats = all_stats['reduce_walking']
        
        # Calculate improvements
        points_change = reduce_stats['positive_count'] - base_stats['positive_count']
        points_pct = (points_change / base_stats['positive_count'] * 100) if base_stats['positive_count'] > 0 else 0
        
        tau_change = reduce_stats['tau_aggregate'] - base_stats['tau_aggregate']
        tau_pct = (tau_change / base_stats['tau_aggregate'] * 100) if base_stats['tau_aggregate'] > 0 else 0
        
        eta_change = reduce_stats['eta_aggregate'] - base_stats['eta_aggregate']
        eta_pct = (eta_change / base_stats['eta_aggregate'] * 100) if base_stats['eta_aggregate'] > 0 else 0
        
        total_change = reduce_stats['total_aggregate'] - base_stats['total_aggregate']
        total_pct = (total_change / base_stats['total_aggregate'] * 100) if base_stats['total_aggregate'] > 0 else 0
        
        distance_change = reduce_stats['distance_avg'] - base_stats['distance_avg']
        distance_pct = (distance_change / base_stats['distance_avg'] * 100) if base_stats['distance_avg'] > 0 else 0
        
        tau_avg_change = reduce_stats['tau_avg'] - base_stats['tau_avg']
        tau_avg_pct = (tau_avg_change / base_stats['tau_avg'] * 100) if base_stats['tau_avg'] > 0 else 0
        
        eta_avg_change = reduce_stats['eta_avg'] - base_stats['eta_avg']
        eta_avg_pct = (eta_avg_change / base_stats['eta_avg'] * 100) if base_stats['eta_avg'] > 0 else 0
        
        bonus_change = reduce_stats['distance_bonus_avg'] - base_stats['distance_bonus_avg']
        
        # Determine overall improvement color
        improvement_color = 'darkgreen' if total_pct >= 0 else 'darkred'
        walking_color = 'darkgreen' if distance_change < 0 else 'darkred'
        
        # Create comprehensive improvement summary text
        # Create comprehensive improvement summary text
        improvement_text = (f"REDUCE WALKING vs BASE:\n"
                        f"Win-Win: {base_stats['positive_count']} → {reduce_stats['positive_count']}\n"
                        
                        f"Rate: {points_pct:+.1f}%\n\n"
                        f"TOTAL AGGREGATES:\n"
                        f"Base: {base_stats['total_aggregate']:.3f} → {reduce_stats['total_aggregate']:.3f}\n"
                        
                        f"Rate: {total_pct:+.1f}%\n\n"
                        f"USER+DISTANCE AGGREGATE:\n"
                        f"Base: {base_stats['tau_aggregate']:.3f} → {reduce_stats['tau_aggregate']:.3f}\n"
                        
                        f"Rate: {tau_pct:+.1f}%\n\n"
                        f"DESTINATION AGGREGATE:\n"
                        f"Base: {base_stats['eta_aggregate']:.3f} → {reduce_stats['eta_aggregate']:.3f}\n"
                        # f"Change: {eta_change:+.3f}\n"
                        f"Rate: {eta_pct:+.1f}%\n\n"
                        f"WALKING DISTANCE:\n"
                        f"Base: {base_stats['distance_avg']:.2f}km → {reduce_stats['distance_avg']:.2f}km\n"
                        )
        
        # Add the improvement summary as a large text box
        ax3.text(0.5, 0.5, improvement_text,
                transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.8", 
                         facecolor=improvement_color, alpha=0.9, edgecolor='black', linewidth=2),
                fontsize=12, fontweight='bold', color='white',
                horizontalalignment='center', verticalalignment='center',
                family='monospace')  # monospace for better alignment
        
    else:
        ax3.text(0.5, 0.5, 'No comparison data available', 
               ha='center', va='center', transform=ax3.transAxes, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{city}_improvements_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    return all_stats

if __name__ == "__main__":
    results = load_and_compare_rome_walking_results()