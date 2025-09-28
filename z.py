import os
import json

def build_hierarchy(root_folder, exclude_folders=None, exclude_files=None, read_files=True):
    """
    Build folder hierarchy with file contents (optional).
    
    Args:
        root_folder (str): Path to the root folder.
        exclude_folders (list): List of folder names to exclude.
        exclude_files (list): List of file names to exclude.
        read_files (bool): Whether to include file content.
    
    Returns:
        dict: Nested hierarchy of folders and files.
    """
    if exclude_folders is None:
        exclude_folders = []
    if exclude_files is None:
        exclude_files = []

    hierarchy = {"name": os.path.basename(root_folder), "type": "folder", "children": []}

    for entry in sorted(os.listdir(root_folder)):
        path = os.path.join(root_folder, entry)

        # Skip excluded folders
        if os.path.isdir(path) and entry in exclude_folders:
            continue
        # Skip excluded files
        if os.path.isfile(path) and entry in exclude_files:
            continue

        if os.path.isdir(path):
            hierarchy["children"].append(build_hierarchy(path, exclude_folders, exclude_files, read_files))
        else:
            file_info = {"name": entry, "type": "file"}
            if read_files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        file_info["content"] = f.read()
                except Exception as e:
                    file_info["content"] = f"[Error reading file: {e}]"
            hierarchy["children"].append(file_info)

    return hierarchy


if __name__ == "__main__":
    root = "."  # current folder, change as needed
    exclude_dirs = ["__pycache__", ".git", "node_modules", ".idea", ".github", ".venv", "build"]
    exclude_files = [".gitignore", "coca.png", "LICENSE", "README.md", "z.py", "setup.py", "hierarchy.txt"]

    tree = build_hierarchy(root, exclude_dirs, exclude_files)

    # Convert to pretty JSON string
    output_str = json.dumps(tree, indent=2, ensure_ascii=False)

    # Print to console
    # print(output_str)

    # Save to txt file
    output_file = "hierarchy.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_str)

    print(f"\n✅ Hierarchy saved to {output_file}")
