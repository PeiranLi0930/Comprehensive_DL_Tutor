import os


def rename_files_in_folder(folder_path):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort files to maintain any existing order
    files.sort()

    # Rename files
    for i, filename in enumerate(files, start = 1):
        # Define the new filename
        new_filename = f"NOV_14_2_{i}.mp4"
        # Define the source and destination paths
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(src, dst)
        print(f"Renamed {filename} to {new_filename}")


folder_path = 'nov_14_2'  # Replace with your folder path
rename_files_in_folder(folder_path)
