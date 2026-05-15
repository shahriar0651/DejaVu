import subprocess
import itertools
import sys
from pathlib import Path
import hydra
from hydra.utils import get_original_cwd


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: dict) -> None:
    """
    Main function to run the KITTI tracking dataset poisoning and evaluation pipeline.
    
    This script runs multiple Python scripts in sequence:
    1. 1_poison_kitti_tracking.py - Apply temporal delays
    2. 2_finalize_kitti_tracking.py - Finalize dataset
    3. 3_evaluate_kitti_tracking.py - Evaluate models (optional, controlled by evaluate)
    4. 4_visualize_attacks.py - Visualize results (optional, controlled by visualize)
    
    Args:
        args: Hydra config object containing dataset configuration
    """
    print(f"Arguments: {sys.argv}")
    
    # Get config values
    evaluate = args.dataset.get('evaluate', True)
    visualize = args.dataset.get('visualize', False)
    original_cwd = Path(get_original_cwd())
    # Scripts are in the same directory as this file
    script_dir = Path(__file__).parent
    
    # Base list of Python files to run (always run steps 1 and 2)
    python_files = [
        '1_poison_kitti_tracking.py',
        '2_finalize_kitti_tracking.py',
    ]
    
    # Conditionally add evaluation script based on config flag
    if evaluate:
        python_files.append('3_evaluate_kitti_tracking.py')
        print("Evaluation enabled: will run 3_evaluate_kitti_tracking.py")
    else:
        print("Evaluation disabled: skipping 3_evaluate_kitti_tracking.py")
    
    # Conditionally add visualization script based on config flag
    if visualize:
        python_files.append('4_visualize_attacks.py')
        print("Visualization enabled: will run 4_visualize_attacks.py")
    else:
        print("Visualization disabled: skipping 4_visualize_attacks.py")
    
    # Parse command line arguments for delay combinations
    # Look for arguments after the script name (skip Hydra's -m flag and config overrides)
    arg_list_total = []
    args_dict = {}
    
    # Parse arguments that look like "key=val1,val2,val3"
    # Skip Hydra-specific flags like -m, --config-path, etc.
    for argument in sys.argv[1:]:
        if '=' in argument and not argument.startswith('-'):
            arg = argument.split("=")[0]
            vals = argument.split("=")[1].split(",")
            args_dict[arg] = vals
    
    print("args_dict:", args_dict)
    
    # Generate all combinations of delay values
    if args_dict:
        combinations = list(itertools.product(*args_dict.values()))
    else:
        # If no delay arguments provided, run once with default config values
        combinations = [()]
        args_dict = {}
    
    # Run scripts for each combination
    for combo in combinations:
        print("\n" + "="*50)
        print("Running with combination:", combo)
        print("="*50 + "\n")
        
        arg_list = []
        for arg, val in zip(args_dict.keys(), combo):
            arg_list.append(f'{arg}={val}')
        
        # Run each Python script with the current argument combination
        for python_file in python_files:
            script_path = script_dir / python_file
            if not script_path.exists():
                print(f"Warning: Script not found: {script_path}")
                continue
            
            print(f"Running: {python_file} with args: {arg_list}")
            cmd = ['python', str(script_path)] + arg_list
            print(f"Command: {' '.join(cmd)}")
            
            subprocess.run(cmd, cwd=str(script_dir), check=False)
    
    print("\n" + "="*50)
    print("Pipeline completed!")
    print("="*50)


if __name__ == '__main__':
    main()

"""
Usage examples:

# Run with default config (no delay combinations) - runs steps 1, 2, 3
python mm_poison_and_evaluate_detect_kitti_dataset.py

# Run steps 1, 2, 4 (skip evaluation, only visualize)
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset.evaluate=False dataset.visualize=True

# Run steps 1, 2, 3 (skip visualization)
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset.evaluate=True dataset.visualize=False

# Run steps 1, 2, 3, 4 (run everything)
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset.evaluate=True dataset.visualize=True

# Run steps 1, 2 only (skip both evaluation and visualization)
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset.evaluate=False dataset.visualize=False

# Run with delay combinations (using Hydra's -m flag for config overrides)
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset=kitti delays.image=0,1,2 delays.lidar=0,1,2

# Run with visualization enabled and evaluation disabled
python mm_poison_and_evaluate_detect_kitti_dataset.py -m dataset.evaluate=False dataset.visualize=True delays.image=5,10

# Run with multiple delay values
python mm_poison_and_evaluate_detect_kitti_dataset.py -m delays.image=0,1,2,3,4,5 delays.lidar=0,1,2,3,4,5
"""
