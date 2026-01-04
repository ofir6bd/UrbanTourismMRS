import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def load_and_plot_rome_results():
    """
    Load Rome experimental results for EASE_WMF and EASE_VAE combinations
    and recreate Figure 4A with emphasized positive quadrant analysis including aggregate values
    """
    
    city = "Rome"
    
    # Load the two specific model combinations
    files_to_load = [
        f"out/experiments/runx_{city}_ease_wmf.pk",
        f"out/experiments/runx_{city}_ease_vae.pk"
    ]
    
    colors = ['royalblue', 'orange']
    labels = ['EASE_WMF', 'EASE_VAE']
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
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
                'percentage': positive_count/total_count*100,
                'tau': file_tau,
                'eta': file_eta,
                'positive_mask': positive_both,
                'tau_aggregate': tau_aggregate,
                'eta_aggregate': eta_aggregate,
                'total_aggregate': total_aggregate,
                'tau_avg': tau_avg,
                'eta_avg': eta_avg
            }
            
            # Plot all points with lower alpha
            ax.scatter(file_tau, file_eta, 
                      alpha=0.3, s=12, 
                      color=colors[file_idx], 
                      label=f'{labels[file_idx]} (all points)')
            
            # Emphasize positive quadrant points
            if positive_count > 0:
                ax.scatter(file_tau[positive_both], file_eta[positive_both], 
                          alpha=0.9, s=45, 
                          color=colors[file_idx], 
                          edgecolors='black', linewidth=0.8,
                          label=f'{labels[file_idx]} (Win-Win: {positive_count})')
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
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
    ax.set_xlabel('User uplift, (U_rs - U_nors) / U_nors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dest uplift, -(H_rs - H_nors) / H_nors', fontsize=14, fontweight='bold')
    ax.set_title('Rome Results - Positive-Sum Impact ', 
                fontsize=16, fontweight='bold')
    
    # Add text annotations with statistics in the plot
    text_y_start = y_max * 0.95
    text_x = x_min + (x_max - x_min) * 0.02
    
    for i, (model_name, stats) in enumerate(model_stats.items()):
        text_y = text_y_start - i * (y_max - y_min) * 0.20
        
        # Create comprehensive text box with model statistics
        text_content = (f"{model_name}:\n"
                       f"Win-Win: {stats['positive_count']}/{stats['total_count']} "
                       f"({stats['percentage']:.1f}%)\n"
                       f"Σ User: {stats['tau_aggregate']:.3f}\n"
                       f"Σ Dest: {stats['eta_aggregate']:.3f}\n"
                       f"Σ Total: {stats['total_aggregate']:.3f}\n"
                       f"Avg: {stats['tau_avg']:.3f}, {stats['eta_avg']:.3f}")
        
        ax.text(text_x, text_y, text_content,
                bbox=dict(boxstyle="round,pad=0.6", 
                         facecolor=colors[i], alpha=0.8, edgecolor='black'),
                fontsize=10, fontweight='bold', color='white',
                verticalalignment='top')
    
    # Add overall statistics
    total_positive = sum(stats['positive_count'] for stats in model_stats.values())
    total_points = sum(stats['total_count'] for stats in model_stats.values())
    overall_percentage = total_positive/total_points*100
    
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
    
    # Add quadrant labels - positioned relative to (0,0)
    quad_offset_x = (x_max - 0) * 0.15
    quad_offset_y = (y_max - 0) * 0.15
    
    # WIN-WIN quadrant (upper right from 0,0)
    # ax.text(0 + quad_offset_x, 0 + quad_offset_y, 
    #         'WIN-WIN\n(Positive Sum)', 
    #         fontsize=14, fontweight='bold', color='darkgreen',
    #         horizontalalignment='center', verticalalignment='center',
    #         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", 
    #                  alpha=0.8, edgecolor='darkgreen', linewidth=2))
    
    # Other quadrants
    ax.text(0 - quad_offset_x, 0 + quad_offset_y, 
            'User Loss\nDest Win', 
            fontsize=10, color='gray', alpha=0.7,
            horizontalalignment='center', verticalalignment='center')
    
    ax.text(0 + quad_offset_x, 0 - quad_offset_y, 
            'User Win\nDest Loss', 
            fontsize=10, color='gray', alpha=0.7,
            horizontalalignment='center', verticalalignment='center')
    
    ax.text(0 - quad_offset_x, 0 - quad_offset_y, 
            'LOSE-LOSE\n(Negative Sum)', 
            fontsize=10, color='darkred', alpha=0.7,
            horizontalalignment='center', verticalalignment='center')
    
    # Legend
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"DETAILED POSITIVE QUADRANT ANALYSIS WITH AGGREGATES")
    print(f"{'='*60}")
    
    for model_name, stats in model_stats.items():
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
    
    print(f"\n{'='*60}")
    print(f"COMBINED RESULTS:")
    print(f"Total Win-Win points: {total_positive}/{total_points}")
    print(f"Overall success rate: {overall_percentage:.2f}%")
    print(f"Combined User aggregate: {total_tau_aggregate:.6f}")
    print(f"Combined Dest aggregate: {total_eta_aggregate:.6f}")
    print(f"Total combined aggregate: {combined_total_aggregate:.6f}")
    print(f"{'='*60}")
    
    return model_stats

if __name__ == "__main__":
    results = load_and_plot_rome_results()
