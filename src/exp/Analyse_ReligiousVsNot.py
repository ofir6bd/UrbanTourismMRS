import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def analyze_religious_site_impact_winwin():
    """
    Analyze the impact of religious domain promotion on visit patterns
    Focus only on win-win scenarios (positive user and destination uplift)
    Load only EASE_WMF and EASE_VAE base models
    """
    
    city = "Rome"
    
    # Load the two specific model combinations
    files_to_load = [
        f"out/experiments/runx_{city}_ease_wmf.pk",
        f"out/experiments/runx_{city}_ease_vae.pk"
    ]
    
    # Load LBSN data to get religious sites info
    try:
        from exp.datafactory import load_lbsn_artefacts
        y, f, _, envstruct = load_lbsn_artefacts(f"out/{city}_lbsn.hdf5")
        religious_sites = envstruct.get("religious_sites", np.array([]))
        total_sites = len(f)
        print(f"Religious sites: {len(religious_sites)} out of {total_sites} total sites")
    except:
        # Fallback if can't load - assume first 20% are religious (example)
        total_sites = 234  # Rome has 234 POIs
        religious_sites = np.arange(int(0.2 * total_sites))
        print(f"Using fallback: assuming first {len(religious_sites)} sites are religious")
    
    # Create religious site mask
    religious_mask = np.zeros(total_sites, dtype=bool)
    religious_mask[religious_sites] = True
    
    # Collect visit data - only from win-win scenarios
    winwin_visit_data = {
        'religious': [],
        'non_religious': [],
        'lambda_values': [],
        'user_uplift': [],
        'dest_uplift': []
    }
    
    for file_path in files_to_load:
        try:
            with open(file_path, "rb") as file:
                experiment = pickle.load(file)
            
            print(f"Loaded: {file_path}")
            
            # Extract choice data for all t and k combinations
            for t in [2, 3, 4]:
                for k in [4, 8, 12, 16, 20]:
                    if t in experiment and k in experiment[t]:
                        exp_data = experiment[t][k]
                        
                        # Get uplift arrays
                        user_uplifts = np.array(exp_data["improvement_user"])
                        dest_uplifts = np.array(exp_data["improvement_harm"])
                        hist_data = exp_data["hist"]
                        
                        # Find win-win scenarios (both uplifts > 0)
                        winwin_mask = (user_uplifts > 0) & (dest_uplifts > 0)
                        winwin_indices = np.where(winwin_mask)[0]
                        
                        print(f"  t={t}, k={k}: {len(winwin_indices)} win-win scenarios out of {len(user_uplifts)}")
                        
                        # Process only win-win scenarios
                        for idx in winwin_indices:
                            if idx < len(hist_data):
                                hist_entry = hist_data[idx]
                                choice_arr = hist_entry["void_choice_arr"]
                                
                                # Split visits by religious vs non-religious
                                religious_visits = np.sum(choice_arr[religious_mask])
                                non_religious_visits = np.sum(choice_arr[~religious_mask])
                                
                                winwin_visit_data['religious'].append(religious_visits)
                                winwin_visit_data['non_religious'].append(non_religious_visits)
                                winwin_visit_data['user_uplift'].append(user_uplifts[idx])
                                winwin_visit_data['dest_uplift'].append(dest_uplifts[idx])
                                
                                # Calculate lambda value (assuming linear spacing from 0 to 1)
                                lambda_val = idx / (len(user_uplifts) - 1) if len(user_uplifts) > 1 else 0
                                winwin_visit_data['lambda_values'].append(lambda_val)
                            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Convert to numpy arrays
    for key in winwin_visit_data:
        winwin_visit_data[key] = np.array(winwin_visit_data[key])
    
    # Calculate statistics for win-win scenarios only
    religious_total = np.sum(winwin_visit_data['religious'])
    non_religious_total = np.sum(winwin_visit_data['non_religious'])
    total_visits = religious_total + non_religious_total
    
    religious_percentage = (religious_total / total_visits) * 100 if total_visits > 0 else 0
    non_religious_percentage = (non_religious_total / total_visits) * 100 if total_visits > 0 else 0
    
    # Create visualization focusing on win-win scenarios
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define colors
    religious_color = '#8B4513'  # Brown for religious sites
    non_religious_color = '#4682B4'  # Steel blue for non-religious sites
    
    # 1. Bar chart - Total visits in win-win scenarios
    categories = ['Religious Sites\n(Win-Win)', 'Non-Religious Sites\n(Win-Win)']
    values = [religious_total, non_religious_total]
    colors = [religious_color, non_religious_color]
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Total Visits (Win-Win Scenarios Only)')
    ax1.set_title('Visit Distribution in Win-Win Scenarios')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.annotate(f'{int(value):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Pie chart - Visit percentage distribution in win-win
    sizes = [religious_percentage, non_religious_percentage]
    labels = [f'Religious\n({religious_percentage:.1f}%)', f'Non-Religious\n({non_religious_percentage:.1f}%)']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=[religious_color, non_religious_color], 
                                      autopct='', startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Win-Win Visit Share Distribution', fontsize=14, fontweight='bold')
    
    # 3. Scatter plot - Religious vs Non-religious visits in win-win scenarios
    ax3.scatter(winwin_visit_data['religious'], winwin_visit_data['non_religious'], 
               alpha=0.6, c=winwin_visit_data['lambda_values'], cmap='viridis', s=50)
    ax3.set_xlabel('Religious Site Visits')
    ax3.set_ylabel('Non-Religious Site Visits')
    ax3.set_title('Religious vs Non-Religious Visits (Win-Win Only)')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for lambda values
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Lambda Value')
    
    # 4. Box plot - Visit distribution comparison in win-win scenarios
    data_to_plot = [winwin_visit_data['religious'], winwin_visit_data['non_religious']]
    box_plot = ax4.boxplot(data_to_plot, labels=['Religious\n(Win-Win)', 'Non-Religious\n(Win-Win)'], 
                          patch_artist=True, notch=True)
    
    # Color the boxes
    box_plot['boxes'][0].set_facecolor(religious_color)
    box_plot['boxes'][0].set_alpha(0.7)
    box_plot['boxes'][1].set_facecolor(non_religious_color)
    box_plot['boxes'][1].set_alpha(0.7)
    
    ax4.set_ylabel('Visits per Win-Win Simulation')
    ax4.set_title('Win-Win Visit Distribution Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics for win-win scenarios
    print(f"\n{'='*80}")
    print(f"RELIGIOUS SITE VISIT ANALYSIS - ROME (WIN-WIN SCENARIOS ONLY)")
    print(f"{'='*80}")
    
    print(f"\nDataset Information:")
    print(f"  Total POIs: {total_sites}")
    print(f"  Religious sites: {len(religious_sites)} ({len(religious_sites)/total_sites*100:.1f}%)")
    print(f"  Non-religious sites: {total_sites - len(religious_sites)} ({(total_sites - len(religious_sites))/total_sites*100:.1f}%)")
    
    print(f"\nWin-Win Scenario Statistics:")
    print(f"  Total win-win simulations: {len(winwin_visit_data['religious'])}")
    print(f"  Total visits in win-win scenarios: {total_visits:,}")
    print(f"  Religious sites visits: {religious_total:,} ({religious_percentage:.1f}%)")
    print(f"  Non-religious sites visits: {non_religious_total:,} ({non_religious_percentage:.1f}%)")
    
    print(f"\nWin-Win Uplift Ranges:")
    print(f"  User uplift range: {np.min(winwin_visit_data['user_uplift']):.3f} to {np.max(winwin_visit_data['user_uplift']):.3f}")
    print(f"  Destination uplift range: {np.min(winwin_visit_data['dest_uplift']):.3f} to {np.max(winwin_visit_data['dest_uplift']):.3f}")
    print(f"  Lambda range in win-win: {np.min(winwin_visit_data['lambda_values']):.3f} to {np.max(winwin_visit_data['lambda_values']):.3f}")
    
    print(f"\nPer-Site Averages (Win-Win Only):")
    avg_religious = religious_total / len(religious_sites) if len(religious_sites) > 0 else 0
    avg_non_religious = non_religious_total / (total_sites - len(religious_sites)) if (total_sites - len(religious_sites)) > 0 else 0
    print(f"  Average visits per religious site: {avg_religious:.1f}")
    print(f"  Average visits per non-religious site: {avg_non_religious:.1f}")
    
    if avg_religious > avg_non_religious:
        ratio = avg_religious / avg_non_religious if avg_non_religious > 0 else float('inf')
        print(f"  Religious sites get {ratio:.1f}x more visits on average in win-win scenarios")
    else:
        ratio = avg_non_religious / avg_religious if avg_religious > 0 else float('inf')
        print(f"  Non-religious sites get {ratio:.1f}x more visits on average in win-win scenarios")
    
    print(f"\nWin-Win Distribution Statistics:")
    print(f"  Religious visits - Mean: {np.mean(winwin_visit_data['religious']):.1f}, Std: {np.std(winwin_visit_data['religious']):.1f}")
    print(f"  Non-religious visits - Mean: {np.mean(winwin_visit_data['non_religious']):.1f}, Std: {np.std(winwin_visit_data['non_religious']):.1f}")
    
    # Calculate correlation between lambda and religious site preference
    if len(winwin_visit_data['lambda_values']) > 1:
        religious_ratio = winwin_visit_data['religious'] / (winwin_visit_data['religious'] + winwin_visit_data['non_religious'])
        correlation = np.corrcoef(winwin_visit_data['lambda_values'], religious_ratio)[0, 1]
        print(f"\nCorrelation between lambda and religious site preference: {correlation:.3f}")
    
    print(f"{'='*80}")
    
    return {
        'winwin_religious_total': religious_total,
        'winwin_non_religious_total': non_religious_total,
        'winwin_religious_percentage': religious_percentage,
        'winwin_non_religious_percentage': non_religious_percentage,
        'winwin_total_visits': total_visits,
        'winwin_scenarios_count': len(winwin_visit_data['religious']),
        'religious_sites_count': len(religious_sites),
        'total_sites': total_sites,
        'winwin_visit_data': winwin_visit_data
    }

if __name__ == "__main__":
    results = analyze_religious_site_impact_winwin()