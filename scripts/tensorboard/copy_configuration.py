import shutil
import sys
import os

configuration_folder = sys.argv[1]
experiments_folder = sys.argv[2]

for configuration_file in os.listdir(configuration_folder):
    try:
        print(f"Exporting {configuration_file}...")
        shutil.copy(f"{configuration_folder}/{configuration_file}", f"{experiments_folder}/{configuration_file}/configuration.toml")
    except Exception as e:
        print(f"Failed: {e}")
