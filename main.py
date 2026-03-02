import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Executing: {script_name}")
    print(f"{'='*50}")
    try:
        # Using sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            print(f"Successfully completed: {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Banking Loan Risk Analysis Pipeline...")
    
    # Define the sequence of scripts
    scripts = [
        'data_cleaning.py',
        'eda.py',
        'modeling.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Critical Error: {script} not found in the current directory.")
            sys.exit(1)
            
    print("\n" + "#"*50)
    print("PIPELINE EXECUTION COMPLETE")
    print("#"*50)
    print("Final results saved to: final_loan_risk_results.csv")
    print("Visualizations saved to: eda_plots/ folder")
