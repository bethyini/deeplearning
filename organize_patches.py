"""
Organize extracted patches into train/val splits
"""

import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_patches(
    metadata_path='patches/patch_metadata.csv',
    output_dir='data',
    test_size=0.2,
    random_state=42
):
    """
    Organize patches into train/val folders
    """
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"Total patches: {len(df)}")
    print(f"Mitosis: {df['label'].sum()}")
    print(f"Hard negatives: {len(df) - df['label'].sum()}")
    
    # Create directories
    for split in ['train', 'val']:
        Path(f'{output_dir}/{split}/mitosis').mkdir(parents=True, exist_ok=True)
        Path(f'{output_dir}/{split}/hard_negative').mkdir(parents=True, exist_ok=True)
    
    # Split by image_id to avoid data leakage
    # (patches from same image should be in same split)
    unique_images = df['image_id'].unique()
    train_imgs, val_imgs = train_test_split(
        unique_images, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_df = df[df['image_id'].isin(train_imgs)]
    val_df = df[df['image_id'].isin(val_imgs)]
    
    print(f"\nTrain: {len(train_df)} patches ({train_df['label'].sum()} mitosis)")
    print(f"Val: {len(val_df)} patches ({val_df['label'].sum()} mitosis)")
    
    # Copy files
    for idx, row in train_df.iterrows():
        src = f"patches/{row['filename']}"
        dst = f"{output_dir}/train/{row['category']}/{row['filename']}"
        shutil.copy(src, dst)
    
    for idx, row in val_df.iterrows():
        src = f"patches/{row['filename']}"
        dst = f"{output_dir}/val/{row['category']}/{row['filename']}"
        shutil.copy(src, dst)
    
    # Save split info
    train_df.to_csv(f'{output_dir}/train_metadata.csv', index=False)
    val_df.to_csv(f'{output_dir}/val_metadata.csv', index=False)
    
    print(f"\nOrganized into {output_dir}/train and {output_dir}/val")

if __name__ == '__main__':
    organize_patches()