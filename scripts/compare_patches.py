"""
Comparison functions for Universal SNAL Patch vs Individualized SNAL Patch performance.
Generates visualizations comparing detection reduction and effectiveness.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_patch_data(individualized_path, universal_path):
    individualized = pd.read_csv(individualized_path)
    universal = pd.read_csv(universal_path)
    return individualized, universal


def plot_detection_reduction_comparison(individualized_df, universal_df, output_dir='outputs'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    data_to_plot = [individualized_df['detection_reduction'], 
                    universal_df['detection_reduction']]
    axes[0].boxplot(data_to_plot, labels=['Individualized', 'Universal'])
    axes[0].set_ylabel('Detection Reduction')
    axes[0].set_title('Detection Reduction Distribution Comparison')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(individualized_df['detection_reduction'], alpha=0.6, label='Individualized', bins=30)
    axes[1].hist(universal_df['detection_reduction'], alpha=0.6, label='Universal', bins=30)
    axes[1].set_xlabel('Detection Reduction')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Detection Reduction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'detection_reduction_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_reduction_rate_comparison(individualized_df, universal_df, output_dir='outputs'):
    individualized_df = individualized_df.copy()
    universal_df = universal_df.copy()
    
    individualized_df['reduction_rate'] = (individualized_df['detection_reduction'] / 
                                           individualized_df['clean_detections'].replace(0, 1)) * 100
    universal_df['reduction_rate'] = (universal_df['detection_reduction'] / 
                                      universal_df['clean_detections'].replace(0, 1)) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    data_to_plot = [individualized_df['reduction_rate'], 
                    universal_df['reduction_rate']]
    axes[0].boxplot(data_to_plot, labels=['Individualized', 'Universal'])
    axes[0].set_ylabel('Reduction Rate (%)')
    axes[0].set_title('Detection Reduction Rate Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(individualized_df['reduction_rate'], alpha=0.6, label='Individualized', bins=30)
    axes[1].hist(universal_df['reduction_rate'], alpha=0.6, label='Universal', bins=30)
    axes[1].set_xlabel('Reduction Rate (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Reduction Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'reduction_rate_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_detection_statistics(individualized_df, universal_df, output_dir='outputs'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    data = [individualized_df['clean_detections'], universal_df['clean_detections']]
    axes[0, 0].boxplot(data, labels=['Individualized', 'Universal'])
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Clean Detections Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    data = [individualized_df['patched_detections'], universal_df['patched_detections']]
    axes[0, 1].boxplot(data, labels=['Individualized', 'Universal'])
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Patched Detections Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(individualized_df['clean_detections'], alpha=0.6, label='Individualized', bins=30)
    axes[1, 0].hist(universal_df['clean_detections'], alpha=0.6, label='Universal', bins=30)
    axes[1, 0].set_xlabel('Clean Detections')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Clean Detections')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(individualized_df['patched_detections'], alpha=0.6, label='Individualized', bins=30)
    axes[1, 1].hist(universal_df['patched_detections'], alpha=0.6, label='Universal', bins=30)
    axes[1, 1].set_xlabel('Patched Detections')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Patched Detections')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'detection_statistics_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_statistics(individualized_df, universal_df, output_dir='outputs'):
    stats = {
        'Mean Detection Reduction': [
            individualized_df['detection_reduction'].mean(),
            universal_df['detection_reduction'].mean()
        ],
        'Median Detection Reduction': [
            individualized_df['detection_reduction'].median(),
            universal_df['detection_reduction'].median()
        ],
        'Std Dev Detection Reduction': [
            individualized_df['detection_reduction'].std(),
            universal_df['detection_reduction'].std()
        ],
        'Max Detection Reduction': [
            individualized_df['detection_reduction'].max(),
            universal_df['detection_reduction'].max()
        ],
        'Min Detection Reduction': [
            individualized_df['detection_reduction'].min(),
            universal_df['detection_reduction'].min()
        ]
    }
    
    x = np.arange(len(stats))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    individualized_values = [stats[key][0] for key in stats.keys()]
    universal_values = [stats[key][1] for key in stats.keys()]
    
    ax.bar(x - width/2, individualized_values, width, label='Individualized')
    ax.bar(x + width/2, universal_values, width, label='Universal')
    
    ax.set_xlabel('Statistic')
    ax.set_ylabel('Value')
    ax.set_title('Summary Statistics: Individualized vs Universal SNAL Patches')
    ax.set_xticks(x)
    ax.set_xticklabels(stats.keys(), rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'summary_statistics_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_statistics_report(individualized_df, universal_df):
    report = ""
    report += "SNAL PATCH PERFORMANCE COMPARISON REPORT\n"
    
    report += "INDIVIDUALIZED SNAL PATCH STATISTICS\n"
    report += f"Total Images: {len(individualized_df)}\n"
    report += f"Mean Detection Reduction: {individualized_df['detection_reduction'].mean():.2f}\n"
    report += f"Median Detection Reduction: {individualized_df['detection_reduction'].median():.2f}\n"
    report += f"Std Dev Detection Reduction: {individualized_df['detection_reduction'].std():.2f}\n"
    report += f"Max Detection Reduction: {individualized_df['detection_reduction'].max():.2f}\n"
    report += f"Min Detection Reduction: {individualized_df['detection_reduction'].min():.2f}\n"
    report += f"Total Detection Reduction: {individualized_df['detection_reduction'].sum():.2f}\n\n"
    
    report += "UNIVERSAL SNAL PATCH STATISTICS\n"
    report += f"Total Images: {len(universal_df)}\n"
    report += f"Mean Detection Reduction: {universal_df['detection_reduction'].mean():.2f}\n"
    report += f"Median Detection Reduction: {universal_df['detection_reduction'].median():.2f}\n"
    report += f"Std Dev Detection Reduction: {universal_df['detection_reduction'].std():.2f}\n"
    report += f"Max Detection Reduction: {universal_df['detection_reduction'].max():.2f}\n"
    report += f"Min Detection Reduction: {universal_df['detection_reduction'].min():.2f}\n"
    report += f"Total Detection Reduction: {universal_df['detection_reduction'].sum():.2f}\n\n"
    
    ind_mean = individualized_df['detection_reduction'].mean()
    univ_mean = universal_df['detection_reduction'].mean()    
    report += "COMPARATIVE ANALYSIS\n"
    report += f"Mean Reduction Difference (Universal - Individualized): {univ_mean - ind_mean:.2f}\n"
    report += f"Images with better reduction (Universal): {(universal_df['detection_reduction'] > individualized_df['detection_reduction']).sum()}\n"
    report += f"Images with worse reduction (Universal): {(universal_df['detection_reduction'] < individualized_df['detection_reduction']).sum()}\n"
    report += f"Images with equal reduction: {(universal_df['detection_reduction'] == individualized_df['detection_reduction']).sum()}\n"
    
    return report


def plot_universal_patch_only_statistics(universal_df, output_dir='outputs'):
    """
    Generate comprehensive statistics and graphs for the universal SNAL patch only.
    """
    universal_df = universal_df.copy()
    universal_df['reduction_rate'] = (universal_df['detection_reduction'] / 
                                      universal_df['clean_detections'].replace(0, 1)) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Detection Reduction
    axes[0, 0].hist(universal_df['detection_reduction'], bins=30, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(universal_df['detection_reduction'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {universal_df['detection_reduction'].mean():.2f}")
    axes[0, 0].axvline(universal_df['detection_reduction'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {universal_df['detection_reduction'].median():.2f}")
    axes[0, 0].set_xlabel('Detection Reduction')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Universal Patch: Detection Reduction Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reduction Rate
    axes[0, 1].hist(universal_df['reduction_rate'], bins=30, color='coral', edgecolor='black')
    axes[0, 1].axvline(universal_df['reduction_rate'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {universal_df['reduction_rate'].mean():.2f}%")
    axes[0, 1].axvline(universal_df['reduction_rate'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {universal_df['reduction_rate'].median():.2f}%")
    axes[0, 1].set_xlabel('Reduction Rate (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Universal Patch: Reduction Rate Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Clean vs Patched Detections
    axes[1, 0].scatter(universal_df['clean_detections'], universal_df['patched_detections'], alpha=0.6, s=50)
    max_val = max(universal_df['clean_detections'].max(), universal_df['patched_detections'].max())
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', label='Perfect Match Line')
    axes[1, 0].set_xlabel('Clean Detections')
    axes[1, 0].set_ylabel('Patched Detections')
    axes[1, 0].set_title('Universal Patch: Clean vs Patched Detections')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot for key metrics
    metrics_data = [
        universal_df['clean_detections'],
        universal_df['patched_detections'],
        universal_df['detection_reduction'],
        universal_df['reduction_rate']
    ]
    bp = axes[1, 1].boxplot(metrics_data, labels=['Clean\nDetections', 'Patched\nDetections', 'Detection\nReduction', 'Reduction\nRate (%)'], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Universal Patch: Key Metrics Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'universal_patch_statistics.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_universal_patch_report(universal_df):
    """
    Generate a detailed statistics report for the universal SNAL patch only.
    """
    universal_df = universal_df.copy()
    universal_df['reduction_rate'] = (universal_df['detection_reduction'] / 
                                      universal_df['clean_detections'].replace(0, 1)) * 100
    
    report = "UNIVERSAL SNAL PATCH - DETAILED STATISTICS REPORT\n"
    report += "=" * 60 + "\n\n"
    
    report += "DATASET OVERVIEW\n"
    report += f"Total Images: {len(universal_df)}\n"
    report += f"Images with Zero Clean Detections: {(universal_df['clean_detections'] == 0).sum()}\n"
    report += f"Images with Successful Reduction: {(universal_df['detection_reduction'] > 0).sum()}\n\n"
    
    report += "CLEAN DETECTIONS STATISTICS\n"
    report += f"Mean: {universal_df['clean_detections'].mean():.2f}\n"
    report += f"Median: {universal_df['clean_detections'].median():.2f}\n"
    report += f"Std Dev: {universal_df['clean_detections'].std():.2f}\n"
    report += f"Min: {universal_df['clean_detections'].min():.0f}\n"
    report += f"Max: {universal_df['clean_detections'].max():.0f}\n"
    report += f"Total: {universal_df['clean_detections'].sum():.0f}\n\n"
    
    report += "PATCHED DETECTIONS STATISTICS\n"
    report += f"Mean: {universal_df['patched_detections'].mean():.2f}\n"
    report += f"Median: {universal_df['patched_detections'].median():.2f}\n"
    report += f"Std Dev: {universal_df['patched_detections'].std():.2f}\n"
    report += f"Min: {universal_df['patched_detections'].min():.0f}\n"
    report += f"Max: {universal_df['patched_detections'].max():.0f}\n"
    report += f"Total: {universal_df['patched_detections'].sum():.0f}\n\n"
    
    report += "DETECTION REDUCTION STATISTICS\n"
    report += f"Mean: {universal_df['detection_reduction'].mean():.2f}\n"
    report += f"Median: {universal_df['detection_reduction'].median():.2f}\n"
    report += f"Std Dev: {universal_df['detection_reduction'].std():.2f}\n"
    report += f"Min: {universal_df['detection_reduction'].min():.2f}\n"
    report += f"Max: {universal_df['detection_reduction'].max():.2f}\n"
    report += f"Total: {universal_df['detection_reduction'].sum():.2f}\n\n"
    
    report += "REDUCTION RATE (%) STATISTICS\n"
    report += f"Mean: {universal_df['reduction_rate'].mean():.2f}%\n"
    report += f"Median: {universal_df['reduction_rate'].median():.2f}%\n"
    report += f"Std Dev: {universal_df['reduction_rate'].std():.2f}%\n"
    report += f"Min: {universal_df['reduction_rate'].min():.2f}%\n"
    report += f"Max: {universal_df['reduction_rate'].max():.2f}%\n\n"
    
    report += "PERFORMANCE SUMMARY\n"
    high_reduction = (universal_df['reduction_rate'] >= 50).sum()
    medium_reduction = ((universal_df['reduction_rate'] >= 25) & (universal_df['reduction_rate'] < 50)).sum()
    low_reduction = (universal_df['reduction_rate'] < 25).sum()
    
    report += f"Images with ≥50% reduction rate: {high_reduction} ({high_reduction/len(universal_df)*100:.1f}%)\n"
    report += f"Images with 25-50% reduction rate: {medium_reduction} ({medium_reduction/len(universal_df)*100:.1f}%)\n"
    report += f"Images with <25% reduction rate: {low_reduction} ({low_reduction/len(universal_df)*100:.1f}%)\n"
    
    return report

if __name__ == '__main__':
    individualized_path = '/home/fayzah/VisDrone/outputs/outputs_snal/universal_patch_results.csv'
    universal_path = '/home/fayzah/VisDrone/outputs/outputs_snal_universal/universal_patch_results.csv'
    output_dir = '/home/fayzah/VisDrone/outputs/comparison_analysis'
    
    print("Loading patch data...")
    individualized_df, universal_df = load_patch_data(individualized_path, universal_path)
    
    print("\nGenerating comparison plots...")
    plot_detection_reduction_comparison(individualized_df, universal_df, output_dir)
    plot_reduction_rate_comparison(individualized_df, universal_df, output_dir)
    plot_detection_statistics(individualized_df, universal_df, output_dir)
    plot_summary_statistics(individualized_df, universal_df, output_dir)
    
    print("\nGenerating universal patch only statistics...")
    plot_universal_patch_only_statistics(universal_df, output_dir)
    
    print("\nGenerating statistics reports...")
    report = generate_statistics_report(individualized_df, universal_df)
    print(report)
    
    report_path = Path(output_dir) / 'statistics_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")
    
    print("\nGenerating universal patch detailed report...")
    universal_report = generate_universal_patch_report(universal_df)
    print(universal_report)
    
    universal_report_path = Path(output_dir) / 'universal_patch_report.txt'
    with open(universal_report_path, 'w') as f:
        f.write(universal_report)
    print(f"Saved: {universal_report_path}")
