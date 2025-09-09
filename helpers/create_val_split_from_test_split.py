import os
import shutil
import random
from pathlib import Path

def create_val_split(dataset_root, split_ratio=0.5):
    """
    Create validation set by moving files from test directory
    
    Args:
        dataset_root: Path to the root directory containing train/test folders
        split_ratio: Fraction of test data to move to validation (default: 0.5 for 50%)
    """
    
    dataset_root = Path(dataset_root)
    
    # Define paths
    test_dir = dataset_root / "test"
    val_dir = dataset_root / "val"
    
    test_images_dir = test_dir / "images"
    test_labels_dir = test_dir / "labels"
    
    # Check if test directories exist
    if not test_images_dir.exists():
        print(f"Error: {test_images_dir} does not exist!")
        return
    
    if not test_labels_dir.exists():
        print(f"Error: {test_labels_dir} does not exist!")
        return
    
    # Create val directories
    val_images_dir = val_dir / "images"
    val_labels_dir = val_dir / "labels"
    
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from test directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(test_images_dir.glob(f"*{ext}"))
    
    if not image_files:
        print("No image files found in test/images directory!")
        return
    
    print(f"Found {len(image_files)} images in test directory")
    
    # Randomly shuffle the files
    random.shuffle(image_files)
    
    # Calculate number of files to move
    num_to_move = int(len(image_files) * split_ratio)
    files_to_move = image_files[:num_to_move]
    
    print(f"Moving {num_to_move} images (and corresponding labels) to validation set...")
    
    moved_images = 0
    moved_labels = 0
    missing_labels = []
    
    for image_file in files_to_move:
        try:
            # Move image file
            dest_image = val_images_dir / image_file.name
            shutil.move(str(image_file), str(dest_image))
            moved_images += 1
            
            # Find corresponding label file
            label_name = image_file.stem + ".txt"
            label_file = test_labels_dir / label_name
            
            if label_file.exists():
                # Move label file
                dest_label = val_labels_dir / label_name
                shutil.move(str(label_file), str(dest_label))
                moved_labels += 1
            else:
                missing_labels.append(label_name)
                
        except Exception as e:
            print(f"Error moving {image_file.name}: {e}")
    
    # Print summary
    print(f"\n✅ Successfully created validation split!")
    print(f"📁 Moved {moved_images} images to val/images/")
    print(f"🏷️  Moved {moved_labels} labels to val/labels/")
    
    if missing_labels:
        print(f"⚠️  Warning: {len(missing_labels)} images had no corresponding label files:")
        for label in missing_labels[:5]:  # Show first 5
            print(f"   - {label}")
        if len(missing_labels) > 5:
            print(f"   ... and {len(missing_labels) - 5} more")
    
    # Show final directory structure
    remaining_test_images = len(list(test_images_dir.glob("*")))
    val_images_count = len(list(val_images_dir.glob("*")))
    
    print(f"\n📊 Final split:")
    print(f"   Test: {remaining_test_images} images")
    print(f"   Val:  {val_images_count} images")

def main():
    # Set your dataset root directory here
    dataset_root = "./annotated_banknote_dataset"
    
    if not os.path.exists(dataset_root):
        print(f"Error: Directory '{dataset_root}' does not exist!")
        return
    
    # Show current structure
    print(f"\nCurrent directory structure:")
    for item in Path(dataset_root).iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
            for subitem in item.iterdir():
                if subitem.is_dir():
                    count = len(list(subitem.glob("*")))
                    print(f"    📁 {subitem.name}/ ({count} files)")
    
    # Confirm action
    response = input(f"\nThis will move 50% of test images/labels to a new val directory. Continue? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        create_val_split(dataset_root)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()