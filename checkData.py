# verify_dataset.py
import os
import pandas as pd
import zipfile

def verify_dataset(folder='alpaca_classification'):
    """Verify the dataset structure"""
    
    print("="*60)
    print("🔍 VERIFYING DATASET")
    print("="*60)
    
    # Check Excel file
    excel_path = os.path.join(folder, 'alpaca_dataset.xlsx')
    if os.path.exists(excel_path):
        print(f"✅ Excel file found: {excel_path}")
        df_train = pd.read_excel(excel_path, sheet_name='Train')
        df_val = pd.read_excel(excel_path, sheet_name='Val')
        df_test = pd.read_excel(excel_path, sheet_name='Test')
        print(f"  Train: {len(df_train)} samples")
        print(f"  Val: {len(df_val)} samples")
        print(f"  Test: {len(df_test)} samples")
        print(f"  Image column: {df_train.columns[1]}")
        print(f"  Sample image names: {df_train['Image Name'].head(3).tolist()}")
    else:
        print(f"❌ Excel file not found: {excel_path}")
        return
    
    # Check images folder
    images_folder = os.path.join(folder, 'images')
    if os.path.exists(images_folder):
        images = os.listdir(images_folder)
        print(f"\n✅ Images folder found: {images_folder}")
        print(f"  Total images: {len(images)}")
        print(f"  Sample images: {images[:5]}")
    else:
        print(f"❌ Images folder not found: {images_folder}")
        return
    
    # Check ZIP file
    zip_path = os.path.join(folder, 'images.zip')
    if os.path.exists(zip_path):
        print(f"\n✅ ZIP file found: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            files = zipf.namelist()
            print(f"  Total files in ZIP: {len(files)}")
            print(f"  Sample files in ZIP: {files[:5]}")
            # Check if images are in a subfolder
            has_subfolder = any('/' in f for f in files)
            if has_subfolder:
                print(f"  ⚠️ Files are in subfolders: {files[0]}")
                print("  This might cause issues. Please recreate ZIP without subfolders.")
            else:
                print("  ✅ Files are at root level (good)")
    else:
        print(f"❌ ZIP file not found: {zip_path}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    verify_dataset()