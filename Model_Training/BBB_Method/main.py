# This file is used to run other scripts
import subprocess
import sys
import random

def run_python_script(script_name, description, seed=None, train_col=None):
    print(f"\n===== Starting {description} ({script_name}) =====")
    command = [sys.executable, script_name]
    command.append(str(seed))
    command.append(str(train_col))
    print(f"Executing command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True
        )
        print(f"===== {description} ({script_name}) FINISHED SUCCESSFULLY =====")
        return True
    except subprocess.CalledProcessError as e:
        print(f"===== {description} ({script_name}) FAILED =====")
        print(f"Error: {e}")
        return False
    
if __name__ == "__main__":
    print("Main controller script started.")
    random.seed(3047)
    SEED_GRID = random.sample(range(1000), 20)
    print(f"Using SEED_GRID: {SEED_GRID}")
    train_cols = ['CX(MPa)', 'UTX(Mpa)', 'G(KJm3)', 'MiniSF(cm)']
    for train_col in train_cols:
        for seed in SEED_GRID:
            program1_success = run_python_script('BBB model training.py', f'{seed} Model Training', seed=seed, train_col=train_col)
            program2_success = run_python_script('BBB model validate.py', f'{seed} Model Validation', seed=seed, train_col=train_col)

            print("\n---------------------------------------------------")
            print("\nMain controller script finished: ALL PROGRAMS EXECUTED SUCCESSFULLY.")