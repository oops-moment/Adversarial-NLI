import os
import sys
import subprocess

# Create required directories
os.makedirs('/kaggle/working/ANLI_Project_Data', exist_ok=True)
os.makedirs('/kaggle/working/ANLI_Project_Data/scripts', exist_ok=True)
os.makedirs('/kaggle/working/ANLI_Project_Data/checkpoints', exist_ok=True)

# Check GPU status
subprocess.run(["nvidia-smi", "-L"])
subprocess.run(["nvidia-smi", "-q"])

# Clone the ANLI repository
if not os.path.exists('anli'):
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/anli.git"])

# Run setup script
subprocess.run(["bash", "anli/setup.sh"])

# Change directory to 'anli/'
try:
    os.chdir('anli')
except FileNotFoundError as e:
    print(f"Could not change directory: {str(e)}")

# Install required packages
subprocess.run(["pip", "install", "transformers", "sentencepiece"])

# Set environment variables
os.environ['PYTHONPATH'] = '/env/python:/kaggle/working/anli/src:/kaggle/working/anli/utest:/kaggle/working/anli/src/dataset_tools'
os.environ['MASTER_ADDR'] = 'localhost'

# Download data using the setup script
subprocess.run(["bash", "./script/download_data.sh"])

# Modify build_data.py to import config correctly
subprocess.run(["sed", "-i", "s/import config/from src import config/", "/kaggle/working/anli/src/dataset_tools/build_data.py"])

# Add src path to sys.path and import config for building data
sys.path.append('/kaggle/working/anli/src')
try:
    import config
except ImportError as e:
    print(f"Import failed: {e}")

# Run build_data script
subprocess.run(["python", "/kaggle/working/anli/src/dataset_tools/build_data.py"])

# Ensure the path is correctly appended at the start of build_data.py and training.py
subprocess.run(["sed", "-i", "1i\\import sys\\nimport os\\nsys.path.append('/kaggle/working/anli/src')", "/kaggle/working/anli/src/dataset_tools/build_data.py"])
subprocess.run(["sed", "-i", "1i\\import sys\\nimport os\\nsys.path.append('/kaggle/working/anli/src')", "/kaggle/working/anli/src/nli/training.py"])

# Run training script with specified parameters
subprocess.run([
    "python", "./src/nli/training.py",
    "--model_class_name", "bert-base",
    "-n", "1",
    "-g", "2",
    "--single_gpu",
    "-nr", "0",
    "--max_length", "156",
    "--gradient_accumulation_steps", "1",
    "--per_gpu_train_batch_size", "16",
    "--per_gpu_eval_batch_size", "16",
    "--save_prediction",
    "--train_data", "anli_r1_train:none,anli_r2_train:none,anli_r3_train:none",
    "--train_weights", "5,10,5",
    "--eval_data", "anli_r1_dev:none,anli_r2_dev:none,anli_r3_dev:none",
    "--eval_frequency", "2000",
    "--experiment_name", "ANLI-R1|ANLI-R2|ANLI-R3"
])