import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
CHART_TITLE = "speedup@1-without-stencil por problem_type"
INPUT_FILE = "metrics/speedup@1-without-stencil.csv"
OUTPUT_FILE = "graphs/speedup@1-without-stencil.png"
# ---------------------

def generate_chart():
    print(f"Reading data from: {INPUT_FILE}")
    
    try:
        # Read CSV, handling potential trailing commas or extra whitespace
        df = pd.read_csv(INPUT_FILE)
        
        # Clean up column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Drop any completely empty rows or rows where 'model' is NaN (like the separator lines in the example)
        df = df.dropna(subset=['model'])
        
        # Identify columns
        # Assuming structure: model, problem_type, value_column
        # We'll take the third column as the value column if specific name isn't guaranteed, 
        # but here we can try to find the numeric one.
        model_col = 'model'
        problem_col = 'problem type'
        
        # Find the value column (the one that is not model or problem type)
        value_col = [c for c in df.columns if c not in [model_col, problem_col] and 'Unnamed' not in c][0]
        
        print(f"Value column identified as: {value_col}")
        
        # Pivot the data for plotting
        # Index: problem_type (x-axis)
        # Columns: model (bars)
        # Values: value_col
        pivot_df = df.pivot(index=problem_col, columns=model_col, values=value_col)
        
        # Plotting
        ax = pivot_df.plot(kind='bar', figsize=(15, 8), width=0.8)
        
        # Styling
        plt.title(CHART_TITLE, fontsize=16, pad=20)
        plt.ylabel(value_col, fontsize=12)
        plt.xlabel(problem_col, fontsize=12)
        plt.xticks(rotation=0)  # Keep x-axis labels horizontal if possible
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)
            
        plt.tight_layout()
        
        # Save
        print(f"Saving chart to: {OUTPUT_FILE}")
        plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
        print("Done!")
        
    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_chart()
