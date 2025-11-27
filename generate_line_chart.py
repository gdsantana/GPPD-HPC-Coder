import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
CHART_TITLE = "pass@k - Média por Modelo"
INPUT_FILE = "metrics/pass@k.csv"
OUTPUT_FILE = "graphs/pass@k-gppd-average.png"
# ---------------------

def generate_chart():
    print(f"Reading data from: {INPUT_FILE}")
    
    try:
        # Read CSV
        df = pd.read_csv(INPUT_FILE)
        
        # Clean up column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Drop any completely empty rows or rows where 'model' is NaN
        df = df.dropna(subset=['model'])
        
        print(f"Columns found: {df.columns.tolist()}")
        print(f"Models found: {df['model'].unique()}")
        
        # Detect metric columns (pass@k, build@k, efficiency@k, etc.)
        # Exclude 'model' and 'problem type' columns
        metric_cols = [col for col in df.columns if '@' in col and col not in ['model', 'problem type']]
        
        if not metric_cols:
            print("ERROR: No metric columns found (expected format: metric@k)")
            return
        
        # Detect the metric name (e.g., 'pass', 'build', 'efficiency')
        metric_name = metric_cols[0].split('@')[0]
        print(f"Detected metric: {metric_name}@k")
        
        # Calculate mean for each model
        grouped = df.groupby('model')[metric_cols].mean()
        
        print(f"\nMean values per model:")
        print(grouped)
        
        # Extract k values from column names (e.g., 'pass@1' -> 1)
        k_values = [int(col.split('@')[1]) for col in metric_cols]
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        # Plot a line for each model
        for model in grouped.index:
            values = grouped.loc[model].values
            plt.plot(k_values, values, marker='o', linewidth=2, markersize=8, label=model)
        
        # Styling
        plt.title(CHART_TITLE, fontsize=24, pad=20)
        plt.xlabel('k', fontsize=18)
        plt.ylabel(f'{metric_name}@k (média)', fontsize=18)
        plt.xticks(k_values, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12, framealpha=0.9)
        
        # Set y-axis limits for better visualization
        plt.ylim([0, 0.4])
        
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        print(f"\nSaving chart to: {OUTPUT_FILE}")
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print("Done!")
        
    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_chart()
