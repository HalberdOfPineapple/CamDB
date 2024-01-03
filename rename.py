import os

# Directory path where the files are located
# Replace '/path/to/the/directory' with the actual path of your directory
directory_path = "/home/viktor/Experiments/CamDB/camtune/benchmarks/tpch_pgbench"

# Script to rename files
for filename in os.listdir(directory_path):
    if filename.endswith(".sql"):
        # Extracting the number part from the file name
        number_part = filename[:-4]
        # New file name
        new_filename = f"{int(number_part)}.sql"
        # Renaming the file
        os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))

# Confirmation message
print("Files have been renamed successfully.")
