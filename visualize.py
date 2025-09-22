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
    """Create a line plot showing accuracy vs token padding size"""
    # Extract token sizes and accuracies
    token_sizes = results_data['token_sizes_tested']
    accuracies = []
    
    for token_size in token_sizes:
        accuracy = results_data['summary_by_token_size'][str(token_size)]['accuracy']
        accuracies.append(accuracy * 100)  # Convert to percentage
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Line plot with markers
    plt.plot(token_sizes, accuracies, 'o-', linewidth=3, markersize=8, 
             color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', 
             markeredgewidth=2, label='AIME Accuracy')
    
    # Customize the plot
    plt.xlabel('Token Padding Size', fontsize=12, fontweight='bold')
    plt.ylabel('AIME Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('AIME 2025 Model Performance vs Token Padding Size\n(Gemini 2.5 Flash with Thinking)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Use log scale for x-axis if we have large token sizes
    if max(token_sizes) > 10000:
        plt.xscale('log')
        plt.xlabel('Token Padding Size (log scale)', fontsize=12, fontweight='bold')
    
    # Add value labels on data points
    for i, (x, y) in enumerate(zip(token_sizes, accuracies)):
        plt.annotate(f'{y:.1f}%', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontweight='bold', 
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Set y-axis limits with some padding
    y_min = min(accuracies) - 5
    y_max = max(accuracies) + 5
    plt.ylim(max(0, y_min), min(100, y_max))
    
    # Add a subtle background
    plt.gca().set_facecolor('#f8f9fa')
    
    # Make the plot look professional
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Accuracy plot saved to {save_path}")
    
    return plt.gcf()

def create_performance_comparison_table(results_data, save_path=None):
    """Create a table showing performance comparison across token sizes"""
    token_sizes = results_data['token_sizes_tested']
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON ACROSS TOKEN PADDING SIZES")
    print("="*80)
    print(f"{'Token Padding':<15} {'Correct':<10} {'Total':<8} {'Accuracy':<12} {'Change':<10}")
    print("-" * 65)
    
    baseline_accuracy = None
    for i, token_size in enumerate(token_sizes):
        summary = results_data['summary_by_token_size'][str(token_size)]
        correct = summary['correct_count']
        total = summary['total_problems']
        accuracy = summary['accuracy'] * 100
        
        if i == 0:
            baseline_accuracy = accuracy
            change = "baseline"
        else:
            change = f"{accuracy - baseline_accuracy:+.2f}%"
        
        print(f"{token_size:<15} {correct:<10} {total:<8} {accuracy:<12.2f}% {change:<10}")
    
    if save_path:
        # Save table to text file
        with open(save_path, 'w') as f:
            f.write("PERFORMANCE COMPARISON ACROSS TOKEN PADDING SIZES\n")
            f.write("="*80 + "\n")
            f.write(f"{'Token Padding':<15} {'Correct':<10} {'Total':<8} {'Accuracy':<12} {'Change':<10}\n")
            f.write("-" * 65 + "\n")
            
            for i, token_size in enumerate(token_sizes):
                summary = results_data['summary_by_token_size'][str(token_size)]
                correct = summary['correct_count']
                total = summary['total_problems']
                accuracy = summary['accuracy'] * 100
                
                if i == 0:
                    baseline_accuracy = accuracy
                    change = "baseline"
                else:
                    change = f"{accuracy - baseline_accuracy:+.2f}%"
                
                f.write(f"{token_size:<15} {correct:<10} {total:<8} {accuracy:<12.2f}% {change:<10}\n")
        
        print(f"Performance table saved to {save_path}")

def create_bar_chart(results_data, save_path=None):
    """Create a bar chart showing accuracy for each token padding size"""
    token_sizes = results_data['token_sizes_tested']
    accuracies = []
    
    for token_size in token_sizes:
        accuracy = results_data['summary_by_token_size'][str(token_size)]['accuracy']
        accuracies.append(accuracy * 100)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create bar chart
    bars = plt.bar(range(len(token_sizes)), accuracies, 
                   color='skyblue', edgecolor='navy', linewidth=1.5, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Token Padding Size', fontsize=12, fontweight='bold')
    plt.ylabel('AIME Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('AIME 2025 Model Performance by Token Padding Size\n(Gemini 2.5 Flash with Thinking)', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(token_sizes)), [str(size) for size in token_sizes], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Set y-axis limits
    plt.ylim(0, max(accuracies) + 5)
    
    # Add a subtle background
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Bar chart saved to {save_path}")
    
    return plt.gcf()

def create_trend_analysis(results_data):
    """Analyze and print trends in the data"""
    token_sizes = results_data['token_sizes_tested']
    accuracies = []
    
    for token_size in token_sizes:
        accuracy = results_data['summary_by_token_size'][str(token_size)]['accuracy']
        accuracies.append(accuracy * 100)
    
    print("\n" + "="*60)
    print("TREND ANALYSIS")
    print("="*60)
    
    # Find best and worst performing configurations
    best_idx = np.argmax(accuracies)
    worst_idx = np.argmin(accuracies)
    
    print(f"Best performance: {accuracies[best_idx]:.2f}% at {token_sizes[best_idx]} token padding")
    print(f"Worst performance: {accuracies[worst_idx]:.2f}% at {token_sizes[worst_idx]} token padding")
    print(f"Performance range: {max(accuracies) - min(accuracies):.2f} percentage points")
    
    # Calculate correlation with token size
    correlation = np.corrcoef(token_sizes, accuracies)[0, 1]
    print(f"Correlation between token padding and accuracy: {correlation:.3f}")
    
    if correlation > 0.1:
        print("→ Weak positive correlation: More padding slightly improves performance")
    elif correlation < -0.1:
        print("→ Weak negative correlation: More padding slightly hurts performance")
    else:
        print("→ No clear correlation: Token padding has minimal impact on performance")
    
    # Look for optimal range
    if len(token_sizes) > 2:
        # Find the token size with best performance
        optimal_size = token_sizes[best_idx]
        print(f"→ Optimal token padding appears to be around {optimal_size} tokens")

def main():
    parser = argparse.ArgumentParser(description='Visualize AIME evaluation results')
    parser.add_argument('json_file', nargs='?', default='aime_2025_token_padding_evaluation_results.json',
                        help='Path to the evaluation results JSON file')
    parser.add_argument('--output-dir', '-o', default='plots',
                        help='Directory to save plots (default: plots)')
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
    print(f"⚙️  Max workers: {results_data.get('max_workers', 'Unknown')}")
    
    # Create performance table
    table_path = os.path.join(args.output_dir, 'performance_table.txt') if not args.no_save else None
    create_performance_comparison_table(results_data, table_path)
    
    # Create trend analysis
    create_trend_analysis(results_data)
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # Line plot
    line_plot_path = os.path.join(args.output_dir, 'accuracy_vs_token_padding.png') if not args.no_save else None
    fig1 = create_accuracy_plot(results_data, line_plot_path)
    
    # Bar chart
    bar_chart_path = os.path.join(args.output_dir, 'accuracy_bar_chart.png') if not args.no_save else None
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
