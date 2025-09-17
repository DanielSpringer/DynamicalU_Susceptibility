import os

def delete_slurm_files(directory):
    """
    Recursively deletes all files with the prefix 'slurm' in the given directory and its subdirectories.

    Parameters:
    directory (str): The root directory to search for slurm files.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            # print(file)
            if file.startswith("oo_3"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    directory = input("Enter the directory path: ").strip()

    if os.path.isdir(directory):
        delete_slurm_files(directory)
    else:
        print("Invalid directory. Please enter a valid path.")
