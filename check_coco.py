import json
from pathlib import Path

base_dir = Path('/orcd/data/edboyden/002/ezh/deeplearning/MIDOGpp')
json_path = base_dir / 'databases' / 'MIDOG++.json'

with open(json_path, 'r') as f:
    data = json.load(f)

# Inspect structure
print("Keys:", data.keys())
print("\nFirst image:", data['images'][0])
print("\nFirst annotation:", data['annotations'][0])
print("\nCategories:", data['categories'])