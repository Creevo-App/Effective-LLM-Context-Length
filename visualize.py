#!/usr/bin/env python3
"""
Visualization script for AIME 2025 token padding evaluation results.
Reads the evaluation JSON file and creates various plots to analyze the impact of token padding on model performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_results(file_path):
    """Load evaluation results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_accuracy_plot(results_data, save_path=None):
    """Create a line plot showing accuracy vs token padding size with error bars"""
    # Extract token sizes and statistics
    token_sizes = results_data['token_sizes_tested']
    mean_accuracies = []
    std_errors = []
    
    for token_size in token_sizes:
        summary = results_data['summary_by_token_size'][str(token_size)]
        mean_accuracy = summary['mean_accuracy'] * 100  # Convert to percentage
        std_error = summary['std_error'] * 100  # Convert to percentage
        mean_accuracies.append(mean_accuracy)
        std_errors.append(std_error)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy arrays for easier manipulation
    token_sizes_np = np.array(token_sizes)
    mean_accuracies_np = np.array(mean_accuracies)
    std_errors_np = np.array(std_errors)
    
    # Plot the main line
    plt.plot(token_sizes, mean_accuracies, 'o-', linewidth=3, markersize=8, 
             color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', 
             markeredgewidth=2, label='Mean AIME Accuracy', zorder=3)
    
    # Add error shadow (fill_between for 3× standard error)
    plt.fill_between(token_sizes, 
                     mean_accuracies_np - 3 * std_errors_np, 
                     mean_accuracies_np + 3 * std_errors_np,
                     alpha=0.3, color='#2E86AB', label='±3 Standard Error', zorder=1)
    
    # Customize the plot
    plt.xlabel('Token Padding Size', fontsize=12, fontweight='bold')
    plt.ylabel('AIME Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('AIME 2025 Model Performance vs Token Padding Size\n(Gemini 2.5 Flash - Mean ± 3 Standard Error)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', zorder=0)
    plt.legend(fontsize=10)
    
    # Use log scale for x-axis if we have large token sizes
    if max(token_sizes) > 10000:
        plt.xscale('log')
        plt.xlabel('Token Padding Size (log scale)', fontsize=12, fontweight='bold')
    
    # Add value labels on data points
    for i, (x, y, err) in enumerate(zip(token_sizes, mean_accuracies, std_errors)):
        plt.annotate(f'{y:.1f}%±{3*err:.1f}%', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontweight='bold', 
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                    zorder=4)
    
    # Set y-axis limits with some padding (using 3× std error)
    y_min = min(mean_accuracies_np - 3 * std_errors_np) - 2
    y_max = max(mean_accuracies_np + 3 * std_errors_np) + 2
    plt.ylim(max(0, y_min), min(100, y_max))
    
    # Add a subtle background
    plt.gca().set_facecolor('#f8f9fa')
    
    # Make the plot look professional
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Accuracy plot with error bars saved to {save_path}")
    
    return plt.gcf()

def create_performance_comparison_table(results_data, save_path=None):
    """Create a table showing performance comparison across token sizes"""
    token_sizes = results_data['token_sizes_tested']
    
    print("\n" + "="*90)
    print("PERFORMANCE COMPARISON ACROSS TOKEN PADDING SIZES")
    print("="*90)
    print(f"{'Token Padding':<15} {'Mean Accuracy':<15} {'Std Error':<12} {'Problems':<10} {'Runs':<8} {'Change':<10}")
    print("-" * 85)
    
    baseline_accuracy = None
    for i, token_size in enumerate(token_sizes):
        summary = results_data['summary_by_token_size'][str(token_size)]
        mean_accuracy = summary['mean_accuracy'] * 100
        std_error = summary['std_error'] * 100
        total_problems = summary['total_problems']
        total_runs = summary['total_runs']
        
        if i == 0:
            baseline_accuracy = mean_accuracy
            change = "baseline"
        else:
            change = f"{mean_accuracy - baseline_accuracy:+.2f}%"
        
        print(f"{token_size:<15} {mean_accuracy:<15.2f}% {std_error:<12.2f}% {total_problems:<10} {total_runs:<8} {change:<10}")
    
    if save_path:
        # Save table to text file
        with open(save_path, 'w') as f:
            f.write("PERFORMANCE COMPARISON ACROSS TOKEN PADDING SIZES\n")
            f.write("="*90 + "\n")
            f.write(f"{'Token Padding':<15} {'Mean Accuracy':<15} {'Std Error':<12} {'Problems':<10} {'Runs':<8} {'Change':<10}\n")
            f.write("-" * 85 + "\n")
            
            baseline_accuracy = None
            for i, token_size in enumerate(token_sizes):
                summary = results_data['summary_by_token_size'][str(token_size)]
                mean_accuracy = summary['mean_accuracy'] * 100
                std_error = summary['std_error'] * 100
                total_problems = summary['total_problems']
                total_runs = summary['total_runs']
                
                if i == 0:
                    baseline_accuracy = mean_accuracy
                    change = "baseline"
                else:
                    change = f"{mean_accuracy - baseline_accuracy:+.2f}%"
                
                f.write(f"{token_size:<15} {mean_accuracy:<15.2f}% {std_error:<12.2f}% {total_problems:<10} {total_runs:<8} {change:<10}\n")
        
        print(f"Performance table saved to {save_path}")

def create_bar_chart(results_data, save_path=None):
    """Create a bar chart showing accuracy for each token padding size with error bars"""
    token_sizes = results_data['token_sizes_tested']
    mean_accuracies = []
    std_errors = []
    
    for token_size in token_sizes:
        summary = results_data['summary_by_token_size'][str(token_size)]
        mean_accuracy = summary['mean_accuracy'] * 100  # Convert to percentage
        std_error = summary['std_error'] * 100  # Convert to percentage
        mean_accuracies.append(mean_accuracy)
        std_errors.append(std_error)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create bar chart with 3× standard error bars
    bars = plt.bar(range(len(token_sizes)), mean_accuracies, yerr=[3*err for err in std_errors],
                   color='skyblue', edgecolor='navy', linewidth=1.5, alpha=0.7,
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Customize the plot
    plt.xlabel('Token Padding Size', fontsize=12, fontweight='bold')
    plt.ylabel('AIME Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('AIME 2025 Model Performance by Token Padding Size\n(Gemini 2.5 Flash - Mean ± 3 Standard Error)', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(token_sizes)), [str(size) for size in token_sizes], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, (bar, acc, err) in enumerate(zip(bars, mean_accuracies, std_errors)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 3*err + 0.5,
                f'{acc:.1f}%±{3*err:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Set y-axis limits (using 3× std error)
    max_height = max(np.array(mean_accuracies) + 3 * np.array(std_errors))
    plt.ylim(0, max_height + 3)
    
    # Add a subtle background
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Bar chart with error bars saved to {save_path}")
    
    return plt.gcf()

def create_trend_analysis(results_data):
    """Analyze and print trends in the data"""
    token_sizes = results_data['token_sizes_tested']
    mean_accuracies = []
    std_errors = []
    
    for token_size in token_sizes:
        summary = results_data['summary_by_token_size'][str(token_size)]
        mean_accuracy = summary['mean_accuracy'] * 100  # Convert to percentage
        std_error = summary['std_error'] * 100  # Convert to percentage
        mean_accuracies.append(mean_accuracy)
        std_errors.append(std_error)
    
    print("\n" + "="*70)
    print("TREND ANALYSIS")
    print("="*70)
    
    # Find best and worst performing configurations
    best_idx = np.argmax(mean_accuracies)
    worst_idx = np.argmin(mean_accuracies)
    
    print(f"Best performance: {mean_accuracies[best_idx]:.2f}% ± {std_errors[best_idx]:.2f}% at {token_sizes[best_idx]} token padding")
    print(f"Worst performance: {mean_accuracies[worst_idx]:.2f}% ± {std_errors[worst_idx]:.2f}% at {token_sizes[worst_idx]} token padding")
    print(f"Performance range: {max(mean_accuracies) - min(mean_accuracies):.2f} percentage points")
    print(f"Average standard error: {np.mean(std_errors):.2f} percentage points")
    
    # Calculate correlation with token size
    correlation = np.corrcoef(token_sizes, mean_accuracies)[0, 1]
    print(f"Correlation between token padding and accuracy: {correlation:.3f}")
    
    if correlation > 0.1:
        print("→ Weak positive correlation: More padding slightly improves performance")
    elif correlation < -0.1:
        print("→ Weak negative correlation: More padding slightly hurts performance")
    else:
        print("→ No clear correlation: Token padding has minimal impact on performance")
    
    # Statistical significance analysis
    baseline_accuracy = mean_accuracies[0]
    baseline_std_error = std_errors[0]
    
    print(f"\nStatistical significance (compared to baseline {token_sizes[0]} tokens):")
    for i, (token_size, accuracy, std_err) in enumerate(zip(token_sizes[1:], mean_accuracies[1:], std_errors[1:]), 1):
        # Simple z-test approximation for difference
        diff = accuracy - baseline_accuracy
        combined_se = np.sqrt(std_err**2 + baseline_std_error**2)
        z_score = abs(diff) / combined_se if combined_se > 0 else 0
        
        significance = "***" if z_score > 2.58 else "**" if z_score > 1.96 else "*" if z_score > 1.64 else ""
        print(f"  {token_size} tokens: {diff:+.2f}% (z={z_score:.2f}) {significance}")
    
    print("\nSignificance levels: * p<0.10, ** p<0.05, *** p<0.01")
    
    # Look for optimal range
    if len(token_sizes) > 2:
        # Find the token size with best performance
        optimal_size = token_sizes[best_idx]
        print(f"\n→ Optimal token padding appears to be around {optimal_size} tokens")
        print(f"→ Multiple runs per configuration provide {np.mean([summary['total_runs'] for summary in results_data['summary_by_token_size'].values()])} samples per token size")

def main():
    parser = argparse.ArgumentParser(description='Visualize AIME evaluation results')
    parser.add_argument('json_file', nargs='?', default='gemini-2.5-flash/aime_2025_token_padding_evaluation_results.json',
                        help='Path to the evaluation results JSON file')
    parser.add_argument('--output-dir', '-o', default='gemini-2.5-flash',
                        help='Directory to save plots (default: gemini-2.5-flash)')
    parser.add_argument('--show-plots', '-s', action='store_true',
                        help='Show plots interactively')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots to files')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.json_file):
        print(f"❌ Results file not found: {args.json_file}")
        print("Make sure to run the evaluation script first to generate results.")
        return
    
    # Create output directory
    if not args.no_save:
        Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load results
    print(f"📊 Loading results from {args.json_file}...")
    results_data = load_results(args.json_file)
    
    # Print basic info
    print(f"📈 Experiment: {results_data.get('experiment_description', 'AIME evaluation')}")
    print(f"🔢 Token sizes tested: {results_data['token_sizes_tested']}")
    print(f"🔄 Number of runs per configuration: {results_data.get('num_runs', 'Unknown')}")
    print(f"📊 Total evaluations: {results_data.get('total_evaluations', 'Unknown')}")
    print(f"⚙️  Max workers: {results_data.get('max_workers', 'Unknown')}")
    
    # Create performance table (console output only)
    create_performance_comparison_table(results_data, None)
    
    # Create trend analysis
    create_trend_analysis(results_data)
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # Line plot
    line_plot_path = os.path.join(args.output_dir, 'line_chart.png') if not args.no_save else None
    fig1 = create_accuracy_plot(results_data, line_plot_path)
    
    # Bar chart
    bar_chart_path = os.path.join(args.output_dir, 'bar_chart.png') if not args.no_save else None
    fig2 = create_bar_chart(results_data, bar_chart_path)
    
    # Show plots if requested
    if args.show_plots:
        print("📱 Displaying plots...")
        plt.show()
    else:
        plt.close('all')
    
    if not args.no_save:
        print(f"\n✅ All visualizations saved to '{args.output_dir}' directory")
    
    print("🎯 Visualization complete!")

if __name__ == "__main__":
    main()
