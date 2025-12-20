"""
Setup script to create PaCoRe project structure
"""

import os
from pathlib import Path

# Define project structure
project_root = Path(__file__).parent
directories = [
    'src',
    'src/models',
    'config',
    'tests',
    'examples',
    'logs',
    'data'
]

# Create directories
for dir_path in directories:
    full_path = project_root / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {dir_path}")

# Create __init__ files for Python packages
init_files = [
    'src/__init__.py',
    'src/models/__init__.py',
    'tests/__init__.py'
]

for init_file in init_files:
    full_path = project_root / init_file
    if not full_path.exists():
        full_path.write_text('"""Package initialization"""\n')
        print(f"Created: {init_file}")

print("\nProject structure created successfully!")
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Configure the system: Edit config/config.yaml")
print("3. Run tests: python -m pytest tests/")
print("4. Start master node: python src/master.py")
