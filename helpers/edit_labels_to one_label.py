import os
import glob

def update_label_files(folder_paths):
    """
    Update all txt files in the specified folders to change class ID to 0
    Handles polygon/segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
    Also handles empty files (no annotations) by leaving them unchanged
    
    Args:
        folder_paths: List of folder paths containing label txt files
    """
    total_files_processed = 0
    empty_files_count = 0
    updated_files_count = 0
    warning_files_count = 0
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist. Skipping...")
            continue
            
        # Find all txt files in the folder
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        
        if not txt_files:
            print(f"No txt files found in '{folder_path}'")
            continue
            
        print(f"Processing {len(txt_files)} files in '{folder_path}'...")
        
        for file_path in txt_files:
            try:
                # Read the file
                with open(file_path, 'r') as file:
                    content = file.read().strip()
                
                # Check if file is empty (no annotations)
                if not content:
                    print(f"Info: File '{os.path.basename(file_path)}' is empty (no annotations) - keeping as is")
                    empty_files_count += 1
                    total_files_processed += 1
                    continue
                
                # Process the single line
                parts = content.split()
                
                if len(parts) < 1:
                    print(f"Warning: File '{os.path.basename(file_path)}' has no content")
                    warning_files_count += 1
                    total_files_processed += 1
                    continue
                
                # Check if we have at least class_id + coordinates
                if len(parts) < 3:  # At least class_id + x1 + y1
                    print(f"Warning: File '{os.path.basename(file_path)}' has insufficient data (only {len(parts)} values)")
                    warning_files_count += 1
                    total_files_processed += 1
                    continue
                
                # Update the class ID (first element) to 0
                parts[0] = '0'
                updated_line = ' '.join(parts)
                
                # Write back to file
                with open(file_path, 'w') as file:
                    file.write(updated_line)
                
                # Show sample of what was processed
                print(f"  ✅ {os.path.basename(file_path)}: Updated class ID")
                
                updated_files_count += 1
                total_files_processed += 1
                
            except Exception as e:
                print(f"Error processing file '{file_path}': {e}")
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY:")
    print(f"{'='*70}")
    print(f"📁 Total files processed: {total_files_processed}")
    print(f"✅ Files with updated class IDs: {updated_files_count}")
    print(f"📄 Empty files (no annotations): {empty_files_count}")
    print(f"⚠️  Files with warnings: {warning_files_count}")
    print(f"{'='*70}")
    
    if updated_files_count > 0:
        print("🎯 All class IDs have been updated to 0")
        print("🔢 Polygon coordinates remain unchanged")

def main():
    # Define your label folder paths here
    label_folders = [
        "./annotated_banknote_dataset/test/labels",
        "./annotated_banknote_dataset/train/labels",
        "./annotated_banknote_dataset/val/labels"  
    ]
    
    print("🏷️  POLYGON LABEL CLASS ID UPDATER")
    print("="*70)
    print("This script will:")
    print("- Update all class IDs to 0 in polygon annotation files")
    print("- Keep empty files unchanged (no annotations)")
    print("- Handle polygon format: class_id x1 y1 x2 y2 x3 y3 ...")
    print("- Preserve all coordinate data")
    print("="*70)
    
    print(f"Target folders:")
    total_files = 0
    for folder in label_folders:
        if os.path.exists(folder):
            txt_count = len(glob.glob(os.path.join(folder, "*.txt")))
            total_files += txt_count
            print(f"  ✅ {folder} ({txt_count} txt files)")
        else:
            print(f"  ❌ {folder} (does not exist)")
    
    print(f"\n📊 Total files to process: {total_files}")
    print("="*70)
    
    # Show example of what will happen
    print("📝 Example transformation:")
    print("  Before: 2 0.298828125 0.7119140625 0.448843... (polygon coordinates)")
    print("  After:  0 0.298828125 0.7119140625 0.448843... (polygon coordinates)")
    print("="*70)
    
    # Ask for confirmation
    response = input("Do you want to proceed? This will modify all txt files in the specified folders. (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        update_label_files(label_folders)
    else:
        print("❌ Operation cancelled.")

if __name__ == "__main__":
    main()