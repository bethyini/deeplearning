"""
Extract patches from MIDOG++ dataset
MIDOG uses [x1, y1, x2, y2] bbox format
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def load_midog_annotations(json_path='databases/MIDOG.json'):
    """Load COCO format annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['images'])} images")
    print(f"Loaded {len(data['annotations'])} annotations")
    print(f"\nCategories:")
    for cat in data['categories']:
        print(f"  {cat['id']}: {cat['name']}")
    
    return data

def extract_patch_with_padding(img_array, center_x, center_y, patch_size=224):
    """Extract patch with zero padding at edges"""
    h, w = img_array.shape[:2]
    half = patch_size // 2
    
    src_y1 = center_y - half
    src_y2 = center_y + half
    src_x1 = center_x - half
    src_x2 = center_x + half
    
    at_edge = (src_y1 < 0 or src_y2 > h or src_x1 < 0 or src_x2 > w)
    
    valid_y1 = max(0, src_y1)
    valid_y2 = min(h, src_y2)
    valid_x1 = max(0, src_x1)
    valid_x2 = min(w, src_x2)
    
    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    
    dst_y1 = valid_y1 - src_y1
    dst_y2 = dst_y1 + (valid_y2 - valid_y1)
    dst_x1 = valid_x1 - src_x1
    dst_x2 = dst_x1 + (valid_x2 - valid_x1)
    
    patch[dst_y1:dst_y2, dst_x1:dst_x2] = img_array[valid_y1:valid_y2, valid_x1:valid_x2]
    
    return patch, at_edge

def extract_all_patches(
    data, 
    image_dir='images',
    output_dir='patches',
    patch_size=224,
    include_hard_negatives=True
):
    """Extract all patches from MIDOG++"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    id_to_file = {img['id']: img['file_name'] for img in data['images']}
    
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    total_patches = 0
    edge_patches = 0
    metadata = []
    
    for img_info in tqdm(data['images'], desc="Extracting patches"):
        img_id = img_info['id']
        filename = img_info['file_name']
        
        img_path = Path(image_dir) / filename
        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue
            
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        anns = img_to_anns.get(img_id, [])
        
        for ann in anns:
            category_id = ann['category_id']
            
            if category_id == 2 and not include_hard_negatives:
                continue
            
            # MIDOG format: bbox = [x1, y1, x2, y2]
            x1, y1, x2, y2 = ann['bbox']
            
            # Calculate center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Extract patch
            patch, at_edge = extract_patch_with_padding(
                img_array, center_x, center_y, patch_size
            )
            
            # Create filename
            label_name = 'mitosis' if category_id == 1 else 'hard_negative'
            patch_filename = f"{filename.split('.')[0]}_{ann['id']:06d}_{label_name}.png"
            
            # Save patch
            Image.fromarray(patch).save(Path(output_dir) / patch_filename)
            
            # Record metadata
            metadata.append({
                'filename': patch_filename,
                'image_id': filename.split('.')[0],
                'annotation_id': ann['id'],
                'label': 1 if category_id == 1 else 0,
                'category': label_name,
                'center_x': center_x,
                'center_y': center_y,
                'at_edge': at_edge,
                'original_image': filename,
                'bbox_width': x2 - x1,
                'bbox_height': y2 - y1
            })
            
            total_patches += 1
            if at_edge:
                edge_patches += 1
    
    df = pd.DataFrame(metadata)
    df.to_csv(Path(output_dir) / 'patch_metadata.csv', index=False)
    
    print(f"Total patches: {total_patches}")
    print(f"Edge patches: {edge_patches} ({100*edge_patches/total_patches:.1f}%)")
    print(f"Mitosis: {df['label'].sum()}")
    print(f"Hard negatives: {len(df) - df['label'].sum()}")
    print(f"Avg bbox size: {df['bbox_width'].mean():.1f} x {df['bbox_height'].mean():.1f} px")
    print(f"\nSaved to: {output_dir}/")
    
    return df

if __name__ == '__main__':
    print("Loading annotations")
    data = load_midog_annotations('databases/MIDOG++.json')
    
    print("\nExtracting patches")
    df = extract_all_patches(
        data,
        image_dir='images',
        output_dir='patches',
        patch_size=224,
        include_hard_negatives=True
    )