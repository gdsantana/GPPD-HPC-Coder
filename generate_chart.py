import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# CHART_TITLE = "pass@1"
INPUT_FILE = "metrics/pass@1-all-models.csv"
OUTPUT_FILE = "graphs/pass@1-all-models.png"
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
        ax = pivot_df.plot(kind='bar', figsize=(20, 12), width=0.9)
        
        # Styling
        # plt.title(CHART_TITLE, fontsize=32, pad=20)
        plt.ylabel(value_col, fontsize=24)
        plt.xlabel(problem_col, fontsize=24)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(rotation=45, fontsize=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.legend(title='Model', loc='lower right', fontsize=20, framealpha=1)
        plt.legend(title='Model', loc='best', framealpha=1, fontsize=18)
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=14, padding=1)
            
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
