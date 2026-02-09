import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import os

def load_and_plot_rome_comparison():
    """
    Compare WMF Random-Init vs WMF LLM-Init using original labels and colors
    to quantify the specific contribution of LLM embeddings.
    """
    
    city = "Rome"
    
    # Files to compare
    files_to_load = [
        f"All_pk_comparison/runx_{city}_ease_wmf_5_with_random_init.pk",
        f"All_pk_comparison/runx_{city}_ease_wmf_5_with_llm_init.pk"
    ]
    
    # Keeping your original color and label preferences
    colors = ['royalblue', 'orange']
    labels = ['EASE_WMF (Original)', 'EASE_WMF (LLM-Init)']
    
    fig, ax = plt.subplots(figsize=(12, 9))
    model_stats = {}
    
    for file_idx, file_path in enumerate(files_to_load):
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            with open(file_path, "rb") as file:
                experiment = pickle.load(file)
            
            file_tau = []
            file_eta = []
            
            # --- STRUCTURE FIX: Match runx.py dictionary keys ---
            for k_gathered in [2, 3, 4]:
                for aset_key in [4, 8, 12, 16, 20]:
                    if k_gathered in experiment and aset_key in experiment[k_gathered]:
                        source = experiment[k_gathered][aset_key]
                        tau = source["improvement_user"]
                        eta = source["improvement_harm"]
                        
                        file_tau.extend(tau)
                        file_eta.extend(eta)
            
            file_tau = np.array(file_tau)
            file_eta = np.array(file_eta)
            
            # Count positive quadrant points
            positive_both = (file_tau > 1e-9) & (file_eta > 1e-9)
            positive_count = np.sum(positive_both)
            total_count = len(file_tau)
            
            # Aggregate calculations for text boxes
            if positive_count > 0:
                tau_aggregate = np.sum(file_tau[positive_both])
                eta_aggregate = np.sum(file_eta[positive_both])
                tau_avg = np.mean(file_tau[positive_both])
                eta_avg = np.mean(file_eta[positive_both])
            else:
                tau_aggregate = eta_aggregate = tau_avg = eta_avg = 0
            
            model_stats[labels[file_idx]] = {
                'positive_count': positive_count,
                'total_count': total_count,
                'percentage': positive_count/total_count*100,
                'tau_aggregate': tau_aggregate,
                'eta_aggregate': eta_aggregate,
                'tau_avg': tau_avg,
                'eta_avg': eta_avg
            }
            
            # Plot all points (lower alpha)
            ax.scatter(file_tau, file_eta, alpha=0.3, s=12, color=colors[file_idx], 
                       label=f'{labels[file_idx]} (all points)')
            
            # Emphasize positive quadrant points (Win-Win)
            if positive_count > 0:
                ax.scatter(file_tau[positive_both], file_eta[positive_both], 
                           alpha=0.9, s=45, color=colors[file_idx], 
                           edgecolors='black', linewidth=0.8,
                           label=f'{labels[file_idx]} (Win-Win: {positive_count})')
                           
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # --- ADDED: IMPROVEMENT CALCULATION AND PRINT ---
    if len(model_stats) == 2:
        m1, m2 = labels[0], labels[1]
        user_imp = ((model_stats[m2]['tau_aggregate'] - model_stats[m1]['tau_aggregate']) / model_stats[m1]['tau_aggregate']) * 100
        dest_imp = ((model_stats[m2]['eta_aggregate'] - model_stats[m1]['eta_aggregate']) / model_stats[m1]['eta_aggregate']) * 100
        
        print("\n" + "="*60)
        print(f"PHASE 6: LLM-INIT IMPROVEMENT OVER ORIGINAL ({city})")
        print("="*60)
        print(f"User Benefit (Σ Tau) Improvement: {user_imp:+.2f}%")
        print(f"Dest Benefit (Σ Eta) Improvement: {dest_imp:+.2f}%")
        print("="*60 + "\n")

    # Maintain your original styling and shading
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_max, y_max = max(xlim[1], 0.1), max(ylim[1], 0.1)
    x_min, y_min = min(xlim[0], -0.1), min(ylim[0], -0.1)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Shade Win-Win Zone
    ax.axhspan(0, y_max, xmin=(0-x_min)/(x_max-x_min), xmax=1, alpha=0.12, color='green', label='Win-Win Zone')
    
    # Standard lines and grid
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Original Labels
    ax.set_xlabel('User uplift, (U_rs - U_nors) / U_nors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dest uplift, -(H_rs - H_nors) / H_nors', fontsize=14, fontweight='bold')
    ax.set_title(f'Rome: LLM-Initialization Impact Comparison', fontsize=16, fontweight='bold')
    
    # Statistics Boxes
    text_y_start = y_max * 0.95
    text_x = x_min + (x_max - x_min) * 0.02
    
    for i, (model_name, stats) in enumerate(model_stats.items()):
        text_y = text_y_start - i * (y_max - y_min) * 0.20
        text_content = (f"{model_name}:\n"
                        f"Win-Win: {stats['positive_count']}/{stats['total_count']} ({stats['percentage']:.1f}%)\n"
                        f"Σ User: {stats['tau_aggregate']:.3f}\n"
                        f"Σ Dest: {stats['eta_aggregate']:.3f}\n"
                        f"Avg: {stats['tau_avg']:.3f}, {stats['eta_avg']:.3f}")
        
        ax.text(text_x, text_y, text_content,
                bbox=dict(boxstyle="round,pad=0.6", facecolor=colors[i], alpha=0.8, edgecolor='black'),
                fontsize=10, fontweight='bold', color='white', verticalalignment='top')
    
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_plot_rome_comparison()