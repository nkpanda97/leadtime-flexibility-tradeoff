#!/bin/bash
# Set job requirements
#SBATCH --partition=himem_8tb
#SBATCH --job-name=flex_analysis_different_windows
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --output=slurm/slurm_output_%j.out
#SBATCH --error=slurm/slurm_error_%j.err

# Loading modules
module load 2023
module load Anaconda3/2023.07-2 
source activate applied_energies

# Define the ROOT directory of the project
export ROOT_DIR='/home/npanda/DataBaseCodes/powertech2025/'

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <script_to_run> [<cs_category>]"
    exit 1
fi

# Get the script to run from the first argument
script_to_run=$1

# Check if the correct script argument is provided and execute the corresponding Python script
if [ "$script_to_run" -eq 1 ]; then
    echo "1: Running BAU profile generation"
    # Ensure the script exists before running it
    ~/.conda/envs/applied_energies/bin/python /home/npanda/github_repos/ev_products_tradeoff/scripts/run_bau_generation.py --root_path $ROOT_DIR


elif [ "$script_to_run" -eq 2 ]; then
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 2 <cs_category>"
        exit 1
    fi
    cs_category=$2
    echo "2: Running Flexi product quantification"
    # Ensure the script exists before running it
    ~/.conda/envs/applied_energies/bin/python /home/npanda/github_repos/ev_products_tradeoff/scripts/run_flex_product.py --root_path $ROOT_DIR --category_cs $cs_category
    

    
else
    echo "Invalid input. Please enter 1 or 2."
    exit 1
fi
