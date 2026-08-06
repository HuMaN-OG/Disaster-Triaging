import os
import json
from pathlib import Path

def count_cdd_images():
    base_dir = Path("Comprehensive Disaster Dataset(CDD)")
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return
    
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    
    for root, _, files in os.walk(base_dir):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                rel_path = Path(root, f).relative_to(base_dir).as_posix()
                image_files.append(rel_path)
                
    print(f"Total CDD images found: {len(image_files)}")
    # Print sample relative paths
    for p in image_files[:10]:
        print(" Sample relative path:", p)

if __name__ == "__main__":
    count_cdd_images()
