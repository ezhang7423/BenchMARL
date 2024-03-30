import os
from pathlib import Path

import yaml


def copy_dataclass_contents(source_directory_path, target_directory_path):
    # Ensure target directory exists
    os.makedirs(target_directory_path, exist_ok=True)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_directory_path):
        if "__" in filename or "common" in filename:
            continue
        print(filename)
        source_file_path = os.path.join(source_directory_path, filename)
        # Skip directories, process only files
        if os.path.isfile(source_file_path):
            with open(source_file_path, "r", encoding="utf-8") as source_file:
                lines = source_file.readlines()

            yaml_info = yaml.safe_load(
                open(
                    f"{Path(target_directory_path) / Path(source_file_path).stem}.yaml",
                    "r",
                )
            )
            new_lines = []

            for ln in lines:
                if "MISSING" not in ln or "from" in ln:
                    new_lines.append(ln)
                else:
                    new_lines.append(
                        ln.replace("MISSING", str(yaml_info[ln.split(":")[0].strip()]))
                    )

            # # Find the line with "@dataclass" and capture everything after it
            # try:
            #     start_index = lines.index(next(filter(lambda line: "@dataclass" in line, lines)))
            # except StopIteration:
            #     continue  # Skip file if "@dataclass" is not found

            # contents_to_copy = lines[start_index:]

            # with open(source_file_path, 'w')             as source_file:
            #     source_file.writelines(lines[:start_index])
            # # Write the captured contents to a new file in the target directory
            target_file_path = os.path.join(target_directory_path, filename)
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.writelines(new_lines)


# Example usage
# source_directory_path = '/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/algorithms'
# target_directory_path = '/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/conf/algorithm'


source_directory_path = (
    "/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/environments/vmas"
)
target_directory_path = (
    "/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/conf/task/vmas"
)

# source_directory_path = '/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/environments/pettingzoo'
# target_directory_path = '/home/ezipe/git/sed-mothership/BenchMARL/benchmarl/conf/task/pettingzoo'

copy_dataclass_contents(source_directory_path, target_directory_path)
