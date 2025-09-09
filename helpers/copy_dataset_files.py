import os
import shutil
import random
from pathlib import Path

def count_files(directory, extensions=None):
    """Count files in directory with specific extensions"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.txt']
    
    if not directory.exists():
        return 0
    
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f"*{ext}")))
    return count

def get_files_to_copy(source_dir, extensions, copy_ratio=1.0):
    """Get list of files to copy based on ratio"""
    if not source_dir.exists():
        return []
    
    files = []
    for ext in extensions:
        files.extend(source_dir.glob(f"*{ext}"))
    
    if copy_ratio < 1.0:
        random.shuffle(files)
        num_to_copy = int(len(files) * copy_ratio)
        files = files[:num_to_copy]
    
    return files

def preview_copy_operation(from_dataset, to_dataset, copy_ratios):
    """Preview what will be copied"""
    from_path = Path(from_dataset)
    to_path = Path(to_dataset)
    
    # Define splits and file types
    splits = ['train', 'test', 'val']
    subdirs = ['images', 'labels']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    label_extensions = ['.txt']
    
    print("📋 COPY PREVIEW")
    print("=" * 60)
    
    total_to_copy = {'images': 0, 'labels': 0}
    copy_plan = {}
    
    for split in splits:
        copy_plan[split] = {}
        
        print(f"\n📁 {split.upper()} SPLIT:")
        print("-" * 40)
        
        for subdir in subdirs:
            from_dir = from_path / split / subdir
            to_dir = to_path / split / subdir
            
            # Get current counts
            current_in_source = count_files(from_dir, image_extensions if subdir == 'images' else label_extensions)
            current_in_dest = count_files(to_dir, image_extensions if subdir == 'images' else label_extensions)
            
            # Calculate files to copy
            copy_ratio = copy_ratios.get(split, 1.0)
            extensions = image_extensions if subdir == 'images' else label_extensions
            files_to_copy = get_files_to_copy(from_dir, extensions, copy_ratio)
            num_to_copy = len(files_to_copy)
            
            # Store in plan
            copy_plan[split][subdir] = {
                'files': files_to_copy,
                'count': num_to_copy,
                'source_dir': from_dir,
                'dest_dir': to_dir
            }
            total_to_copy[subdir] += num_to_copy
            
            # Display info
            ratio_text = f" ({copy_ratio*100:.0f}%)" if copy_ratio < 1.0 else ""
            print(f"  {subdir:8} | Source: {current_in_source:4d} | Destination: {current_in_dest:4d} | Will copy: {num_to_copy:4d}{ratio_text}")
    
    print("\n" + "=" * 60)
    print("📊 TOTAL SUMMARY:")
    print(f"  Images to copy: {total_to_copy['images']}")
    print(f"  Labels to copy: {total_to_copy['labels']}")
    print("=" * 60)
    
    return copy_plan

def copy_files(copy_plan):
    """Execute the copy operation"""
    print("\n🚀 Starting copy operation...")
    
    total_copied = {'images': 0, 'labels': 0}
    errors = []
    
    for split, split_data in copy_plan.items():
        print(f"\n📁 Processing {split}...")
        
        for subdir, data in split_data.items():
            if data['count'] == 0:
                continue
                
            # Create destination directory if it doesn't exist
            data['dest_dir'].mkdir(parents=True, exist_ok=True)
            
            print(f"  Copying {data['count']} {subdir}...")
            
            copied_count = 0
            for file_path in data['files']:
                try:
                    dest_path = data['dest_dir'] / file_path.name
                    
                    # Handle file name conflicts
                    counter = 1
                    original_dest = dest_path
                    while dest_path.exists():
                        stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_path = data['dest_dir'] / f"{stem}_{counter:03d}{suffix}"
                        counter += 1
                    
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
                    
                except Exception as e:
                    error_msg = f"Error copying {file_path.name}: {e}"
                    errors.append(error_msg)
                    print(f"    ❌ {error_msg}")
            
            total_copied[subdir] += copied_count
            print(f"    ✅ Successfully copied {copied_count} {subdir}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 COPY OPERATION COMPLETED!")
    print(f"✅ Total images copied: {total_copied['images']}")
    print(f"✅ Total labels copied: {total_copied['labels']}")
    
    if errors:
        print(f"⚠️  Errors encountered: {len(errors)}")
        print("First few errors:")
        for error in errors[:3]:
            print(f"  - {error}")
    
    print("=" * 60)

def main():
    print("🔄 DATASET COPY TOOL")
    print("=" * 60)
    
    # Get input paths
    from_dataset = 'C:/Sami/ScaleXI Projects/suspicious_detection/dataset'
    to_dataset = './annotated_banknote_dataset'
    
    if not os.path.exists(from_dataset):
        print(f"❌ Error: Source dataset '{from_dataset}' does not exist!")
        return
    
    if not os.path.exists(to_dataset):
        print(f"⚠️  Destination dataset '{to_dataset}' does not exist. It will be created.")
    
    # Get copy ratios for each split
    print("\n📝 Copy ratios (1.0 = 100%, 0.5 = 50%, etc.):")
    copy_ratios = {}
    
    for split in ['train', 'test', 'val']:
        while True:
            try:
                ratio_input = input(f"  {split} (default 1.0): ").strip()
                if not ratio_input:
                    copy_ratios[split] = 1.0
                    break
                ratio = float(ratio_input)
                if 0 < ratio <= 1.0:
                    copy_ratios[split] = ratio
                    break
                else:
                    print("    Please enter a value between 0 and 1.0")
            except ValueError:
                print("    Please enter a valid number")
    
    # Preview the operation
    copy_plan = preview_copy_operation(from_dataset, to_dataset, copy_ratios)
    
    # Confirm operation
    print(f"\n❓ Proceed with copying files from '{from_dataset}' to '{to_dataset}'?")
    response = input("Enter 'yes' to continue: ").strip().lower()
    
    if response == 'yes':
        copy_files(copy_plan)
    else:
        print("❌ Operation cancelled.")

if __name__ == "__main__":
    main()