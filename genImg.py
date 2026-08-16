# generate_alpaca_dataset.py - FIXED VERSION
import os
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
import zipfile

def generate_alpaca_dataset(
    alpaca_folder='data/alpaca',
    not_alpaca_folder='data/not_alpaca',
    output_folder='alpaca_classification',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42
):
    """
    Generate Excel file and organize images for alpaca classification
    """
    
    print("="*60)
    print("🦙 GENERATING ALPACA CLASSIFICATION DATASET")
    print("="*60)
    
    # Create output folders
    images_folder = os.path.join(output_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    
    print(f"\n📁 Output folder: {output_folder}")
    print(f"📁 Images folder: {images_folder}")
    
    # Collect all images
    print("\n📸 Scanning for images...")
    
    all_images = []
    
    # Process alpaca images
    if os.path.exists(alpaca_folder):
        alpaca_files = [f for f in os.listdir(alpaca_folder) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        print(f"  Found {len(alpaca_files)} alpaca images")
        for f in alpaca_files:
            all_images.append({
                'original_path': os.path.join(alpaca_folder, f),
                'label': 'alpaca',
                'original_name': f
            })
    else:
        print(f"  ⚠️ Warning: Alpaca folder not found: {alpaca_folder}")
    
    # Process not_alpaca images
    if os.path.exists(not_alpaca_folder):
        not_alpaca_files = [f for f in os.listdir(not_alpaca_folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        print(f"  Found {len(not_alpaca_files)} not_alpaca images")
        for f in not_alpaca_files:
            all_images.append({
                'original_path': os.path.join(not_alpaca_folder, f),
                'label': 'not_alpaca',
                'original_name': f
            })
    else:
        print(f"  ⚠️ Warning: Not Alpaca folder not found: {not_alpaca_folder}")
    
    if not all_images:
        print("\n❌ Error: No images found!")
        print("Please make sure your folders contain image files (.jpg, .jpeg, .png)")
        return None
    
    print(f"\n✅ Total images found: {len(all_images)}")
    
    # Shuffle and split
    random.seed(random_seed)
    random.shuffle(all_images)
    
    # Calculate split indices
    n = len(all_images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_images)} images ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_images)} images ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_images)} images ({test_ratio*100:.0f}%)")
    
    # Copy images to single folder with unique names
    print("\n📁 Copying images to single folder...")
    
    def copy_images(image_list, prefix):
        copied = []
        for i, img_info in enumerate(image_list):
            # Get file extension
            original_name = img_info['original_name']
            ext = os.path.splitext(original_name)[1]
            if not ext:
                # Try to detect extension from file
                if os.path.exists(img_info['original_path']):
                    # Check if file exists and get its extension
                    ext = os.path.splitext(img_info['original_path'])[1]
                if not ext:
                    ext = '.jpg'  # Default extension
            
            # Create unique name
            new_name = f"{prefix}_{i:04d}{ext}"
            new_path = os.path.join(images_folder, new_name)
            
            # Copy file
            try:
                shutil.copy2(img_info['original_path'], new_path)
                copied.append({
                    'sample_id': f"{prefix}_{i:04d}",
                    'image_name': new_name,
                    'label': img_info['label'],
                    'original_path': img_info['original_path']
                })
                if (i + 1) % 50 == 0:
                    print(f"  Copied {i + 1} images...")
            except Exception as e:
                print(f"  ⚠️ Error copying {img_info['original_name']}: {e}")
        
        return copied
    
    # Copy all images
    train_copied = copy_images(train_images, 'TRAIN')
    val_copied = copy_images(val_images, 'VAL')
    test_copied = copy_images(test_images, 'TEST')
    
    print(f"\n  ✅ Copied {len(train_copied)} training images")
    print(f"  ✅ Copied {len(val_copied)} validation images")
    print(f"  ✅ Copied {len(test_copied)} test images")
    
    # Create DataFrames for Excel
    print("\n📊 Creating Excel sheets...")
    
    # Train sheet
    train_df = pd.DataFrame([
        {
            'Sample ID': item['sample_id'],
            'Image Name': item['image_name'],
            'Target': item['label']
        }
        for item in train_copied
    ])
    
    # Val sheet
    val_df = pd.DataFrame([
        {
            'Sample ID': item['sample_id'],
            'Image Name': item['image_name'],
            'Target': item['label']
        }
        for item in val_copied
    ])
    
    # Test sheet (without target for prediction)
    test_df = pd.DataFrame([
        {
            'Sample ID': item['sample_id'],
            'Image Name': item['image_name']
        }
        for item in test_copied
    ])
    
    # Save to Excel
    excel_path = os.path.join(output_folder, 'alpaca_dataset.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        train_df.to_excel(writer, sheet_name='Train', index=False)
        val_df.to_excel(writer, sheet_name='Val', index=False)
        test_df.to_excel(writer, sheet_name='Test', index=False)
    
    print(f"  ✅ Excel file saved: {excel_path}")
    
    # Create a summary file
    summary_path = os.path.join(output_folder, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("🦙 ALPACA CLASSIFICATION DATASET SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"  Alpaca: {sum(1 for img in all_images if img['label'] == 'alpaca')}\n")
        f.write(f"  Not Alpaca: {sum(1 for img in all_images if img['label'] == 'not_alpaca')}\n\n")
        f.write(f"Train: {len(train_copied)} images\n")
        f.write(f"  Alpaca: {sum(1 for img in train_copied if img['label'] == 'alpaca')}\n")
        f.write(f"  Not Alpaca: {sum(1 for img in train_copied if img['label'] == 'not_alpaca')}\n\n")
        f.write(f"Validation: {len(val_copied)} images\n")
        f.write(f"  Alpaca: {sum(1 for img in val_copied if img['label'] == 'alpaca')}\n")
        f.write(f"  Not Alpaca: {sum(1 for img in val_copied if img['label'] == 'not_alpaca')}\n\n")
        f.write(f"Test: {len(test_copied)} images\n")
        f.write(f"  Alpaca: {sum(1 for img in test_copied if img['label'] == 'alpaca')}\n")
        f.write(f"  Not Alpaca: {sum(1 for img in test_copied if img['label'] == 'not_alpaca')}\n\n")
        f.write("="*60 + "\n")
        f.write("📁 Files:\n")
        f.write(f"  Excel: {excel_path}\n")
        f.write(f"  Images: {images_folder}\n")
        f.write("="*60 + "\n")
    
    print(f"  ✅ Summary saved: {summary_path}")
    
    # Print sample of the data
    print("\n📋 Sample of Training Data:")
    print(train_df.head(10).to_string())
    
    print("\n" + "="*60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*60)
    
    return {
        'excel_path': excel_path,
        'images_folder': images_folder,
        'train_count': len(train_copied),
        'val_count': len(val_copied),
        'test_count': len(test_copied)
    }

def create_zip_file(folder_path, output_path=None):
    """Create a ZIP file of the images folder - FIXED to preserve structure"""
    if output_path is None:
        output_path = folder_path + '.zip'
    
    print(f"\n📦 Creating ZIP file: {output_path}")
    
    # Get the folder name
    folder_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Store with relative path
                arcname = os.path.relpath(file_path, parent_dir)
                zipf.write(file_path, arcname)
    
    print(f"  ✅ ZIP file created: {output_path}")
    
    # List contents of ZIP
    print("\n📋 ZIP file contents:")
    with zipfile.ZipFile(output_path, 'r') as zipf:
        for i, name in enumerate(zipf.namelist()[:10]):  # Show first 10
            print(f"  {name}")
        if len(zipf.namelist()) > 10:
            print(f"  ... and {len(zipf.namelist()) - 10} more files")
    
    return output_path

if __name__ == "__main__":
    # Run the dataset generation
    result = generate_alpaca_dataset(
        alpaca_folder='data/alpaca',
        not_alpaca_folder='data/not_alpaca',
        output_folder='alpaca_classification',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )
    
    if result:
        # Create ZIP file of images for easy upload
        zip_path = create_zip_file(result['images_folder'])
        
        print("\n🎯 Next Steps:")
        print("  1. Open ML-Forge in your browser")
        print(f"  2. Upload the Excel file: {result['excel_path']}")
        print(f"  3. Upload the ZIP file: {zip_path}")
        print("  4. Select the Feature Extraction Model (ResNet50 recommended)")
        print("  5. Train and evaluate!")