import os
import subprocess

def run_command(command, env=None):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True, env=env)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {command}")
        print(e.stderr)

# Step 1: Create and activate a Conda environment
def setup_conda_environment():
    print("Setting up Conda environment...")
    run_command("conda create -n newCondaEnvironment -c cctbx202208 -y")
    run_command("source /opt/conda/bin/activate newCondaEnvironment && conda install -c cctbx202208 python=3.6.13 -y")
    run_command("source activate newCondaEnvironment && python --version")


# Step 3: Copy ANLI data into the InfoBERT project directory
def copy_anli_data():
    print("Copying ANLI data...")
    src_path = "/kaggle/input/anlp-anli-data/anli_data"
    dest_path = "/kaggle/working/InfoBERT/ANLI"
    run_command(f"cp -r {src_path} {dest_path}")

# Step 4: Change to InfoBERT directory
def change_to_infobert_directory():
    print("Navigating to InfoBERT directory...")
    os.chdir("/kaggle/working/InfoBERT")

# Step 5: Install required packages
def install_requirements():
    print("Installing requirements...")
    run_command("source activate newCondaEnvironment && pip install -r requirements.txt")

# Step 6: Change to ANLI directory
def change_to_anli_directory():
    print("Navigating to ANLI directory...")
    os.chdir("/kaggle/working/InfoBERT/ANLI")

# Step 7: Run the experiment
def run_experiment():
    print("Running InfoBERT experiment...")
    command = (
        "source activate newCondaEnvironment && source setup.sh && "
        "runexp anli-full infobert roberta-base 2e-5 32 128 -1 1000 42 "
        "1e-5 5e-3 6 0.1 0 4e-2 8e-2 0 3 5e-3 0.5 0.9"
    )
    run_command(command)

# Main function
if __name__ == "__main__":
    setup_conda_environment()
    copy_anli_data()
    change_to_infobert_directory()
    install_requirements()
    change_to_anli_directory()
    run_experiment()
