import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def load_and_plot_rome_results():
    """
    Load Rome experimental results for EASE_WMF, EASE_VAE, EASE_HYBRID_KNN, and EASE_KNN_BASELINE
    and recreate Figure 4A with emphasized positive quadrant analysis
    """
    
    city = "Rome"
    
    # Load the model combinations including the correct baseline
    files_to_load = [
        f"out/experiments/runx_{city}_ease_wmf.pk",
        f"out/experiments/runx_{city}_ease_vae.pk", 
        f"out/experiments/runx_{city}_ease_hybrid_knn.pk",
        f"out/experiments/runx_{city}_ease_knn_baseline.pk"
    ]
    
    colors = ['royalblue', 'orange', 'green', 'red']
    labels = ['EASE_WMF', 'EASE_VAE', 'EASE_HYBRID_KNN', 'EASE_KNN_BASELINE']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    model_stats = {}
    
    for file_idx, file_path in enumerate(files_to_load):
        try:
            with open(file_path, "rb") as file:
                experiment = pickle.load(file)
            
            file_tau = []
            file_eta = []
            
            # Extract data for all t and k combinations
            for t in [2, 3, 4]:  # number of known preferences
                for k in [4, 8, 12, 16, 20]:  # awareness set sizes
                    if t in experiment and k in experiment[t]:
                        source = experiment[t][k]
                        tau = source["improvement_user"]  # user uplift array
                        eta = source["improvement_harm"]  # destination uplift array
                        
                        file_tau.extend(tau)
                        file_eta.extend(eta)
            
            # Convert to numpy arrays
            file_tau = np.array(file_tau)
            file_eta = np.array(file_eta)
            
            # Count positive quadrant points for this model
            positive_both = (file_tau > 0) & (file_eta > 0)
            positive_count = np.sum(positive_both)
            total_count = len(file_tau)
            
            # Calculate aggregate values in positive quadrant
            if positive_count > 0:
                pos_tau_values = file_tau[positive_both]
                pos_eta_values = file_eta[positive_both]
                
                # Sum of all positive values
                tau_aggregate = np.sum(pos_tau_values)
                eta_aggregate = np.sum(pos_eta_values)
                total_aggregate = tau_aggregate + eta_aggregate
                
                # Average values in positive quadrant
                tau_avg = np.mean(pos_tau_values)
                eta_avg = np.mean(pos_eta_values)
            else:
                tau_aggregate = eta_aggregate = total_aggregate = 0
                tau_avg = eta_avg = 0
            
            model_stats[labels[file_idx]] = {
                'positive_count': positive_count,
                'total_count': total_count,
                'percentage': positive_count/total_count*100 if total_count > 0 else 0,
                'tau': file_tau,
                'eta': file_eta,
                'positive_mask': positive_both,
                'tau_aggregate': tau_aggregate,
                'eta_aggregate': eta_aggregate,
                'total_aggregate': total_aggregate,
                'tau_avg': tau_avg,
                'eta_avg': eta_avg
            }
            
            # All models use circles with same size
            marker = 'o'  # All models use circles now
            marker_size = 15  # Same size for all
            edge_width = 0.8  # Same edge width for all
            
            # Plot all points with lower alpha
            ax.scatter(file_tau, file_eta, 
                      alpha=0.3, s=marker_size, marker=marker,
                      color=colors[file_idx], 
                      label=f'{labels[file_idx]} (all points)')
            
            # Emphasize positive quadrant points
            if positive_count > 0:
                ax.scatter(file_tau[positive_both], file_eta[positive_both], 
                          alpha=0.9, s=marker_size*1.5, marker=marker,
                          color=colors[file_idx], 
                          edgecolors='black', linewidth=edge_width,
                          label=f'{labels[file_idx]} (Win-Win: {positive_count})')
            
            print(f"Loaded {labels[file_idx]}: {total_count} points, {positive_count} Win-Win")
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Check if we have any data
    if not model_stats:
        print("No data loaded successfully!")
        return None
    
    # Get current axis limits to determine the range
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Set axis limits to start from 0
    x_max = max(xlim[1], 0.1)  # Ensure some positive range is visible
    y_max = max(ylim[1], 0.1)  # Ensure some positive range is visible
    x_min = min(xlim[0], -0.1)  # Ensure some negative range is visible
    y_min = min(ylim[0], -0.1)  # Ensure some negative range is visible
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Shade the positive quadrant (Win-Win zone) - now properly from (0,0)
    ax.axhspan(0, y_max, xmin=(0-x_min)/(x_max-x_min), xmax=1, 
               alpha=0.12, color='green', label='Win-Win Zone')
    
    # Draw quadrant lines through (0,0)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    ax.set_xlabel('User uplift', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dest uplift', fontsize=14, fontweight='bold')
    ax.set_title('Rome Results - Modles Comparison', 
                fontsize=16, fontweight='bold')
    
    # Add text annotations with statistics - ALL 4 MODELS IN SINGLE VERTICAL LINE ON LEFT
    text_x = x_min + (x_max - x_min) * 0.02  # Fixed x position on the left
    y_range = y_max - y_min
    
    # Calculate vertical positions for 4 models with equal spacing
    vertical_positions = [
        y_max - 0.05 * y_range,  # Top
        y_max - 0.28 * y_range,  # Upper middle
        y_max - 0.51 * y_range,  # Lower middle  
        y_max - 0.74 * y_range   # Bottom
    ]
    
    for i, (model_name, stats) in enumerate(model_stats.items()):
        text_y = vertical_positions[i]
        
        # Create comprehensive text box with model statistics
        text_content = (f"{model_name}:\n"
                       f"Win-Win: {stats['positive_count']}/{stats['total_count']} "
                       f"({stats['percentage']:.1f}%)\n"
                       f"Σ User: {stats['tau_aggregate']:.3f}\n"
                       f"Σ Dest: {stats['eta_aggregate']:.3f}\n"
                       f"Σ Total: {stats['total_aggregate']:.3f}\n"
                       f"Avg: {stats['tau_avg']:.3f}, {stats['eta_avg']:.3f}")
        
        # Same styling for all models
        bbox_style = dict(boxstyle="round,pad=0.4", 
                         facecolor=colors[i], alpha=0.85, 
                         edgecolor='black')
        font_size = 12
        
        ax.text(text_x, text_y, text_content,
                bbox=bbox_style,
                fontsize=font_size, fontweight='bold', color='white',
                verticalalignment='top')
    
    # Add overall statistics - moved to upper right to avoid overlap
    if model_stats:
        total_positive = sum(stats['positive_count'] for stats in model_stats.values())
        total_points = sum(stats['total_count'] for stats in model_stats.values())
        overall_percentage = total_positive/total_points*100 if total_points > 0 else 0
        
        # Calculate combined aggregates
        total_tau_aggregate = sum(stats['tau_aggregate'] for stats in model_stats.values())
        total_eta_aggregate = sum(stats['eta_aggregate'] for stats in model_stats.values())
        combined_total_aggregate = total_tau_aggregate + total_eta_aggregate
        
        # Add overall statistics box
        overall_text = (f"COMBINED WIN-WIN:\n"
                       f"Points: {total_positive}/{total_points} ({overall_percentage:.1f}%)\n"
                       f"Σ User: {total_tau_aggregate:.3f}\n"
                       f"Σ Dest: {total_eta_aggregate:.3f}\n"
                       f"Σ Total: {combined_total_aggregate:.3f}")
        
        ax.text(x_max * 0.98, y_max * 0.98, overall_text,
                bbox=dict(boxstyle="round,pad=0.6", 
                         facecolor='darkgreen', alpha=0.9, edgecolor='black'),
                fontsize=11, fontweight='bold', color='white',
                horizontalalignment='right', verticalalignment='top')
    
    # Add quadrant labels - positioned relative to (0,0) but avoiding left side overlap
    quad_offset_x = (x_max - 0) * 0.15
    quad_offset_y = (y_max - 0) * 0.15
    
    # Win-Win quadrant label - moved more to the right to avoid overlap
    # ax.text(0 + quad_offset_x * 1.8, 0 + quad_offset_y, 
    #         'WIN-WIN\n(Positive Sum)', 
    #         fontsize=12, color='darkgreen', alpha=0.8, fontweight='bold',
    #         horizontalalignment='center', verticalalignment='center')
    
    # Other quadrants - positioned to avoid left side
    ax.text(0 + quad_offset_x * 1.8, 0 - quad_offset_y, 
            'User Win\nDest Loss', 
            fontsize=10, color='gray', alpha=0.7,
            horizontalalignment='center', verticalalignment='center')
    
    ax.text(0 - quad_offset_x * 0.5, 0 + quad_offset_y, 
            'User Loss\nDest Win', 
            fontsize=10, color='gray', alpha=0.7,
            horizontalalignment='center', verticalalignment='center')
    
    ax.text(0 - quad_offset_x * 0.5, 0 - quad_offset_y, 
            'LOSE-LOSE\n(Negative Sum)', 
            fontsize=10, color='darkred', alpha=0.7,
            horizontalalignment='center', verticalalignment='center')
    
    # Legend - moved to bottom right to avoid overlap
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n{'='*70}")
    print(f"DETAILED POSITIVE QUADRANT ANALYSIS - EASE MODELS vs KNN BASELINE")
    print(f"{'='*70}")
    
    # Separate EASE models from baseline for clearer reporting
    ease_models = {k: v for k, v in model_stats.items() if 'baseline' not in k.lower()}
    baseline_models = {k: v for k, v in model_stats.items() if 'baseline' in k.lower()}
    
    print(f"\nEASE VARIANT MODELS:")
    print(f"-" * 40)
    for model_name, stats in ease_models.items():
        print(f"\n{model_name}:")
        print(f"  Total points: {stats['total_count']}")
        print(f"  Win-Win points: {stats['positive_count']}")
        print(f"  Success rate: {stats['percentage']:.2f}%")
        print(f"  Sum of User uplift (Win-Win): {stats['tau_aggregate']:.6f}")
        print(f"  Sum of Dest uplift (Win-Win): {stats['eta_aggregate']:.6f}")
        print(f"  Total aggregate value: {stats['total_aggregate']:.6f}")
        
        if stats['positive_count'] > 0:
            print(f"  Avg user uplift (Win-Win): {stats['tau_avg']:.4f}")
            print(f"  Avg dest uplift (Win-Win): {stats['eta_avg']:.4f}")
    
    if baseline_models:
        print(f"\nBASELINE MODEL:")
        print(f"-" * 40)
        for model_name, stats in baseline_models.items():
            print(f"\n{model_name}:")
            print(f"  Total points: {stats['total_count']}")
            print(f"  Win-Win points: {stats['positive_count']}")
            print(f"  Success rate: {stats['percentage']:.2f}%")
            print(f"  Sum of User uplift (Win-Win): {stats['tau_aggregate']:.6f}")
            print(f"  Sum of Dest uplift (Win-Win): {stats['eta_aggregate']:.6f}")
            print(f"  Total aggregate value: {stats['total_aggregate']:.6f}")
            
            if stats['positive_count'] > 0:
                print(f"  Avg user uplift (Win-Win): {stats['tau_avg']:.4f}")
                print(f"  Avg dest uplift (Win-Win): {stats['eta_avg']:.4f}")
    
    if model_stats:
        total_positive = sum(stats['positive_count'] for stats in model_stats.values())
        total_points = sum(stats['total_count'] for stats in model_stats.values())
        overall_percentage = total_positive/total_points*100 if total_points > 0 else 0
        total_tau_aggregate = sum(stats['tau_aggregate'] for stats in model_stats.values())
        total_eta_aggregate = sum(stats['eta_aggregate'] for stats in model_stats.values())
        combined_total_aggregate = total_tau_aggregate + total_eta_aggregate
        
        print(f"\n{'='*70}")
        print(f"COMBINED RESULTS (ALL MODELS):")
        print(f"Total Win-Win points: {total_positive}/{total_points}")
        print(f"Overall success rate: {overall_percentage:.2f}%")
        print(f"Combined User aggregate: {total_tau_aggregate:.6f}")
        print(f"Combined Dest aggregate: {total_eta_aggregate:.6f}")
        print(f"Total combined aggregate: {combined_total_aggregate:.6f}")
        
        # Performance ranking
        print(f"\nPERFORMANCE RANKING (by Win-Win success rate):")
        print(f"-" * 50)
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            print(f"{rank}. {model_name}: {stats['percentage']:.2f}% "
                  f"({stats['positive_count']}/{stats['total_count']} points)")
        
        # Compare EASE variants vs baseline
        if ease_models and baseline_models:
            ease_positive = sum(stats['positive_count'] for stats in ease_models.values())
            ease_total = sum(stats['total_count'] for stats in ease_models.values())
            ease_percentage = ease_positive/ease_total*100 if ease_total > 0 else 0
            
            baseline_stats = list(baseline_models.values())[0]  # Only one baseline
            baseline_percentage = baseline_stats['percentage']
            
            print(f"\nEASE VARIANTS vs KNN BASELINE COMPARISON:")
            print(f"EASE Variants Average Win-Win Rate: {ease_percentage:.2f}% ({ease_positive}/{ease_total})")
            print(f"KNN Baseline Win-Win Rate: {baseline_percentage:.2f}% "
                  f"({baseline_stats['positive_count']}/{baseline_stats['total_count']})")
            print(f"Improvement over baseline: {ease_percentage - baseline_percentage:.2f} percentage points")
            
            # Individual comparisons
            print(f"\nINDIVIDUAL vs BASELINE COMPARISONS:")
            for model_name, stats in ease_models.items():
                improvement = stats['percentage'] - baseline_percentage
                multiplier = stats['percentage'] / baseline_percentage if baseline_percentage > 0 else float('inf')
                print(f"• {model_name}: {improvement:+.2f} pp improvement ({multiplier:.1f}x better)")
        
        # Key insights
        print(f"\nKEY INSIGHTS:")
        print(f"-" * 20)
        best_model = sorted_models[0]
        worst_model = sorted_models[-1]
        print(f"• Best performer: {best_model[0]} with {best_model[1]['percentage']:.1f}% Win-Win rate")
        print(f"• Worst performer: {worst_model[0]} with {worst_model[1]['percentage']:.1f}% Win-Win rate")
        print(f"• Performance gap: {best_model[1]['percentage'] - worst_model[1]['percentage']:.1f} percentage points")
        
        if baseline_models:
            baseline_name = list(baseline_models.keys())[0]
            baseline_perf = baseline_models[baseline_name]['percentage']
            best_ease = max(ease_models.items(), key=lambda x: x[1]['percentage'])
            print(f"• Best EASE variant ({best_ease[0]}) is {best_ease[1]['percentage']/baseline_perf:.1f}x better than {baseline_name}")
        
        print(f"{'='*70}")
    
    return model_stats

if __name__ == "__main__":
    results = load_and_plot_rome_results()