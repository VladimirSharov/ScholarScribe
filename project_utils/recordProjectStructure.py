import os

def record_project_structure(start_path, output_file='project_structure.txt'):
    """Walk through the project directory and record its structure to a file."""
    with open(output_file, 'w', encoding='utf-8') as file:  # Ensure the file is opened with utf-8 encoding
        for root, dirs, files in os.walk(start_path):
            level = root.replace(start_path, '').count(os.sep)
            indent = '│   ' * level + '├── '
            file.write(f'{indent}{os.path.basename(root)}/\n')
            subindent = '│   ' * (level + 1)
            for f in files:
                file.write(f'{subindent}├── {f}\n')

if __name__ == '__main__':
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Going one level up
    output_file = os.path.join(os.path.dirname(__file__), 'project_structure.txt')  # Saves in the current folder
    record_project_structure(project_path, output_file)
