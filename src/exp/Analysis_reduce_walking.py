import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from scipy import stats

def plot_distances_specific_models(city="Rome"):
    """
    Plot distance analysis for specific model combinations only
    """
    
    # Load the two specific model combinations
    files_to_load = [
        f"out/experiments/runx_{city}_ease_wmf.pk",
        f"out/experiments/runx_{city}_ease_vae.pk"
    ]
    
    # Collect distance data from specific experiments
    distance_data = {
        "void": [],
        "cond": [],
        "nors": [],
        "data": []
    }
    
    experiment_configs = []
    
    for file_path in files_to_load:
        try:
            with open(file_path, "rb") as file:
                experiment = pickle.load(file)
            
            # Extract model info from filename
            filename = file_path.split('/')[-1]  # Get just the filename
            parts = filename.replace('.pk', '').split('_')
            key = parts[2]  # ease
            recommender_system_key = parts[3]  # wmf or vae
            
            print(f"Loading: {key} -> {recommender_system_key}")
            
            for t in (2, 3, 4):
                for a_size in experiment[t].keys():
                    source = experiment[t][a_size]
                    
                    # Extract distance data from the last simulation in each experiment
                    if 'hist' in source and len(source['hist']) > 0:
                        last_hist = source['hist'][-1]  # Get the last lambda value's results
                        
                        if 'void_total_distance' in last_hist:
                            distance_data["void"].extend(last_hist['void_total_distance'])
                            distance_data["cond"].extend(last_hist['cond_total_distance'])
                            distance_data["nors"].extend(last_hist['nors_total_distance'])
                            distance_data["data"].extend(last_hist['data_total_distance'])
                            
                            experiment_configs.append({
                                'key': key,
                                'recommender': recommender_system_key,
                                't': t,
                                'a_size': a_size
                            })
                            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not any(distance_data.values()):
        print("No distance data found in experiment files. Make sure the updated simulation has been run.")
        return
    
    # Convert to numpy arrays
    for key in distance_data:
        distance_data[key] = np.array(distance_data[key])
    
    print(f"Loaded distance data:")
    for scenario, data in distance_data.items():
        print(f"  {scenario}: {len(data)} samples")
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Distance Analysis for {city} Tourism Recommendations\n(EASE→WMF & EASE→VAE models)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribution comparison (top-left)
    ax1 = axes[0, 0]
    scenarios = ['void', 'cond', 'nors', 'data']
    scenario_labels = ['With MRS (Void)', 'Conditional', 'No Recommendations', 'Ground Truth']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (scenario, label, color) in enumerate(zip(scenarios, scenario_labels, colors)):
        if len(distance_data[scenario]) > 0:
            ax1.hist(distance_data[scenario], bins=50, alpha=0.7, label=label, 
                    color=color, density=True)
    
    ax1.set_xlabel('Total Travel Distance (km)')
    ax1.set_ylabel('Density')
    ax1.set_title('(A) Distance Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison (top-right)
    ax2 = axes[0, 1]
    box_data = [distance_data[scenario] for scenario in scenarios if len(distance_data[scenario]) > 0]
    box_labels = [scenario_labels[i] for i, scenario in enumerate(scenarios) if len(distance_data[scenario]) > 0]
    
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Total Travel Distance (km)')
    ax2.set_title('(B) Distance Distribution Summary')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Statistical comparison (bottom-left)
    ax3 = axes[1, 0]
    
    # Calculate statistics
    stats_data = []
    for scenario, label in zip(scenarios, scenario_labels):
        if len(distance_data[scenario]) > 0:
            data = distance_data[scenario]
            stats_data.append({
                'Scenario': label,
                'Mean': np.mean(data),
                'Std': np.std(data),
                'Median': np.median(data),
                'Count': len(data)
            })
    
    if stats_data:
        # Create bar plot of means with error bars
        means = [s['Mean'] for s in stats_data]
        stds = [s['Std'] for s in stats_data]
        labels = [s['Scenario'] for s in stats_data]
        
        bars = ax3.bar(range(len(means)), means, yerr=stds, capsize=5, 
                       color=colors[:len(means)], alpha=0.7)
        ax3.set_xticks(range(len(means)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_ylabel('Mean Total Distance (km)')
        ax3.set_title('(C) Mean Distance with Standard Deviation')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Distance efficiency analysis (bottom-right)
    ax4 = axes[1, 1]
    
    if len(distance_data['nors']) > 0 and len(distance_data['void']) > 0:
        # Calculate percentage change from no-recommendation baseline
        baseline_mean = np.mean(distance_data['nors'])
        
        changes = []
        change_labels = []
        change_colors = []
        
        for scenario, label, color in zip(['void', 'data'], ['With MRS', 'Ground Truth'], ['#1f77b4', '#d62728']):
            if len(distance_data[scenario]) > 0:
                scenario_mean = np.mean(distance_data[scenario])
                pct_change = ((scenario_mean - baseline_mean) / baseline_mean) * 100
                changes.append(pct_change)
                change_labels.append(label)
                change_colors.append(color)
        
        if changes:
            bars = ax4.bar(range(len(changes)), changes, color=change_colors, alpha=0.7)
            ax4.set_xticks(range(len(changes)))
            ax4.set_xticklabels(change_labels)
            ax4.set_ylabel('Distance Change from Baseline (%)')
            ax4.set_title('(D) Distance Change vs No Recommendations')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, change in zip(bars, changes):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., 
                        height + (1 if height >= 0 else -3),
                        f'{change:.1f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for\nefficiency analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('(D) Distance Efficiency Analysis')
    
    plt.tight_layout()
    plt.savefig(f'{city}_distance_analysis_specific_models.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Distance Analysis Summary for {city} (EASE→WMF & EASE→VAE) ===")
    for stats in stats_data:
        print(f"{stats['Scenario']:20s}: Mean={stats['Mean']:6.2f}km, "
              f"Std={stats['Std']:6.2f}km, Median={stats['Median']:6.2f}km, "
              f"N={stats['Count']:,}")
    
    # Statistical significance tests
    print(f"\n=== Statistical Tests ===")
    if len(distance_data['void']) > 0 and len(distance_data['nors']) > 0:
        t_stat, p_val = stats.ttest_ind(distance_data['void'], distance_data['nors'])
        print(f"MRS vs No-Rec t-test: t={t_stat:.3f}, p={p_val:.6f}")
        
    if len(distance_data['void']) > 0 and len(distance_data['data']) > 0:
        t_stat, p_val = stats.ttest_ind(distance_data['void'], distance_data['data'])
        print(f"MRS vs Ground Truth t-test: t={t_stat:.3f}, p={p_val:.6f}")
    
    return distance_data, experiment_configs

def plot_distance_vs_utility_tradeoff_specific(city="Rome"):
    """
    Plot the relationship between distance and utility for specific model combinations
    """
    
    files_to_load = [
        f"out/experiments/runx_{city}_ease_wmf.pk",
        f"out/experiments/runx_{city}_ease_vae.pk"
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Distance-Utility Tradeoff Analysis for {city}', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for WMF, Orange for VAE
    
    for idx, file_path in enumerate(files_to_load):
        try:
            with open(file_path, "rb") as file:
                experiment = pickle.load(file)
            
            # Extract model info from filename
            filename = file_path.split('/')[-1]
            parts = filename.replace('.pk', '').split('_')
            key = parts[2]  # ease
            recommender_system_key = parts[3]  # wmf or vae
            
            # Collect data across lambda values
            lambda_values = []
            avg_distances = []
            avg_utilities = []
            
            # Use a representative configuration
            t, a_size = 3, 12  # middle values
            if t in experiment and a_size in experiment[t]:
                source = experiment[t][a_size]
                
                for i, hist in enumerate(source['hist']):
                    # Lambda values from 0 to 1 in 41 steps
                    lambda_val = i / 40.0
                    lambda_values.append(lambda_val)
                    
                    # Calculate average distance for this lambda
                    if 'void_total_distance' in hist:
                        avg_distances.append(np.mean(hist['void_total_distance']))
                
                # Get utility data
                avg_utilities = source['improvement_user']
                
                if len(lambda_values) > 0 and len(avg_distances) > 0:
                    # Plot on first subplot - Distance vs Lambda
                    axes[0].plot(lambda_values, avg_distances, 
                               color=colors[idx], marker='o', markersize=3, 
                               linewidth=2, label=f'EASE→{recommender_system_key.upper()}')
                    
                    # Plot on second subplot - Distance vs Utility
                    scatter = axes[1].scatter(avg_distances, avg_utilities, 
                                            c=lambda_values, cmap='viridis', 
                                            s=30, alpha=0.7, 
                                            label=f'EASE→{recommender_system_key.upper()}')
                    
                    print(f"Loaded {recommender_system_key.upper()}: {len(lambda_values)} lambda points")
                
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Configure first subplot
    axes[0].set_xlabel('Lambda (Sustainability Weight)')
    axes[0].set_ylabel('Average Travel Distance (km)')
    axes[0].set_title('(A) Distance vs Sustainability Weight')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Configure second subplot
    axes[1].set_xlabel('Average Travel Distance (km)')
    axes[1].set_ylabel('User Utility Uplift')
    axes[1].set_title('(B) Distance-Utility Tradeoff')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add colorbar for lambda values
    if 'scatter' in locals():
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('Lambda (Sustainability Weight)')
    
    plt.tight_layout()
    plt.savefig(f'{city}_distance_utility_tradeoff_specific_models.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage:
if __name__ == "__main__":
    # Plot distance analysis for specific Rome model combinations
    distance_data, configs = plot_distances_specific_models(city="Rome")
    
    # Plot distance-utility tradeoff for specific models
    plot_distance_vs_utility_tradeoff_specific(city="Rome")