#!/usr/bin/env python3
"""
Script to delete all .npy files within a specified folder
Supports both single folder and recursive deletion
"""

from pathlib import Path

def delete_npy_files(folder_path, recursive=False, dry_run=False):
    """
    Delete all .npy files in the specified folder
    
    Parameters:
        folder_path (str): Path to the folder
        recursive (bool): Whether to search recursively in subfolders
        dry_run (bool): If True, only show what would be deleted without actually deleting
    
    Returns:
        int: Number of files deleted (or would be deleted in dry run)
    """
    
    # Convert to Path object for easier handling
    folder = Path(folder_path)
    
    # Check if folder exists
    if not folder.exists():
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
        return 0
    
    if not folder.is_dir():
        print(f"❌ Error: '{folder_path}' is not a directory!")
        return 0
    
    # Find .npy files
    if recursive:
        pattern = "**/*.npy"
        npy_files = list(folder.glob(pattern))
    else:
        pattern = "*.npy"
        npy_files = list(folder.glob(pattern))
    
    if not npy_files:
        print(f"✅ No .npy files found in '{folder_path}'" + (" (recursive)" if recursive else ""))
        return 0
    
    print(f"📁 Found {len(npy_files)} .npy files in '{folder_path}'" + (" (recursive)" if recursive else ""))
    
    deleted_count = 0
    
    for file_path in npy_files:
        try:
            if dry_run:
                print(f"🔍 [DRY RUN] Would delete: {file_path}")
            else:
                file_path.unlink()  # Delete the file
                print(f"🗑️  Deleted: {file_path}")
            deleted_count += 1
            
        except PermissionError:
            print(f"❌ Permission denied: {file_path}")
        except FileNotFoundError:
            print(f"❌ File not found (may have been deleted): {file_path}")
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")
    
    if dry_run:
        print(f"\n🔍 [DRY RUN] Would delete {deleted_count} files")
    else:
        print(f"\n✅ Successfully deleted {deleted_count} .npy files")
    
    return deleted_count

def main(folder_path, recursive=False, dry_run=False):
    """
    Main function with direct parameters
    
    Parameters:
        folder_path (str): Path to the folder
        recursive (bool): Delete .npy files recursively in all subfolders
        dry_run (bool): Show what would be deleted without actually deleting
    """
    
    print("🧹 NPY File Cleaner")
    print("=" * 50)
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will actually be deleted\n")
    
    deleted_count = delete_npy_files(folder_path, recursive, dry_run)
    
    print("=" * 50)
    if deleted_count > 0:
        if dry_run:
            print(f"📊 Summary: {deleted_count} .npy files would be deleted")
        else:
            print(f"📊 Summary: {deleted_count} .npy files deleted successfully")
    else:
        print("📊 Summary: No .npy files found or deleted")
    
    return deleted_count

# Simple function for direct use
def quick_delete_npy(folder_path):
    """
    Quick function to delete .npy files - no options, just delete
    
    Parameters:
        folder_path (str): Path to folder containing .npy files
    
    Example:
        quick_delete_npy("./my_folder")
    """
    import os
    import glob
    
    # Get all .npy files in the folder
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in '{folder_path}'")
        return
    
    print(f"Deleting {len(npy_files)} .npy files...")
    
    for file_path in npy_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"✅ Done! Deleted {len(npy_files)} .npy files")

# Using pathlib (more modern approach)
def delete_npy_pathlib(folder_path, recursive=False):
    """
    Delete .npy files using pathlib (modern Python approach)
    
    Parameters:
        folder_path (str): Path to folder
        recursive (bool): Whether to search recursively
    
    Example:
        delete_npy_pathlib("./cache", recursive=True)
    """
    from pathlib import Path
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder '{folder_path}' does not exist!")
        return
    
    # Find .npy files
    if recursive:
        npy_files = list(folder.rglob("*.npy"))  # rglob for recursive
    else:
        npy_files = list(folder.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in '{folder_path}'")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    for file_path in npy_files:
        try:
            file_path.unlink()
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"✅ Deleted {len(npy_files)} .npy files")

# Batch processing function
def clean_multiple_folders(folder_list, recursive=False, dry_run=False):
    """
    Clean .npy files from multiple folders
    
    Parameters:
        folder_list (list): List of folder paths to clean
        recursive (bool): Whether to search recursively
        dry_run (bool): Show what would be deleted without deleting
    
    Example:
        clean_multiple_folders(["./cache", "./temp", "./data"], recursive=True)
    """
    total_deleted = 0
    
    print("🧹 Batch NPY File Cleaner")
    print("=" * 50)
    
    for folder in folder_list:
        print(f"\n📂 Processing folder: {folder}")
        deleted = delete_npy_files(folder, recursive, dry_run)
        total_deleted += deleted
    
    print("=" * 50)
    print(f"📊 Total: {total_deleted} .npy files " + ("would be deleted" if dry_run else "deleted"))

if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    
    # Single folder cleanup
    main(
        folder_path="./annotated_banknote_dataset/val/images",      # Change to your target folder
        recursive=False,             # Set to False for current folder only
        dry_run=False               # Set to False to actually delete files
    )
    
    # Alternative: Quick cleanup (no options)
    # quick_delete_npy("./my_folder")
    
    # Alternative: Modern pathlib approach
    # delete_npy_pathlib("./data", recursive=True)
    
    # Alternative: Multiple folders
    # clean_multiple_folders(["./cache", "./temp", "./data"], recursive=True, dry_run=True)

# ================================
# USAGE EXAMPLES:
# ================================

"""
1. Direct Function Usage:
   main("./cache", recursive=True, dry_run=True)
   quick_delete_npy("./cache")
   delete_npy_pathlib("./data", recursive=True)

2. Batch Processing:
   clean_multiple_folders(["./cache", "./temp"], recursive=True)

3. For YOLO Cache Cleanup:
   main("./runs/detect/train/cache", recursive=False, dry_run=True)
   clean_multiple_folders(["./model_outputs", "./cache"], recursive=True)
   
4. Quick Examples:
   # Test first with dry run
   main("./my_folder", recursive=True, dry_run=True)
   
   # Then actually delete
   main("./my_folder", recursive=True, dry_run=False)
"""