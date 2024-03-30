import os

def copy_dataclass_contents(source_directory_path, target_directory_path):
    # Ensure target directory exists
    os.makedirs(target_directory_path, exist_ok=True)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_directory_path):
        source_file_path = os.path.join(source_directory_path, filename)

        # Skip directories, process only files
        if os.path.isfile(source_file_path):
            with open(source_file_path, 'r', encoding='utf-8') as source_file:
                lines = source_file.readlines()
            
            # Find the line with "@dataclass" and capture everything after it
            try:
                start_index = lines.index(next(filter(lambda line: "@dataclass" in line, lines)))
            except StopIteration:
                continue  # Skip file if "@dataclass" is not found
            
            contents_to_copy = lines[start_index:]
            
            with open(source_file_path, 'w')             as source_file:
                source_file.writelines(lines[:start_index])
            # Write the captured contents to a new file in the target directory
            target_file_path = os.path.join(target_directory_path, filename)
            with open(target_file_path, 'w', encoding='utf-8') as target_file:
                target_file.writelines(contents_to_copy)

# Example usage
source_directory_path = '/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/algorithms'
target_directory_path = '/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/conf/algorithm'
copy_dataclass_contents(source_directory_path, target_directory_path)
