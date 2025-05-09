#!/bin/bash

# delete_intermediate_files.sh
#
# Purpose: Cleans up intermediate files (e.g., *_taut.smi) from terminal directories
#          within a specified root directory structure (like ZINC downloads).
# WARNING: This script uses 'rm'. Be careful where you run it.
# Usage:   ./delete_intermediate_files.sh /path/to/root <filename_pattern>
# Example: ./delete_intermediate_files.sh /path/to/zinc/3D "*_taut.smi"

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <root_directory> <filename_pattern_to_delete>"
  echo "Example: $0 /path/to/data \"*_taut.smi\""
  exit 1
fi

root_directory="$1"
filename_pattern="$2"

if [ ! -d "$root_directory" ]; then
  echo "Error: Root directory not found: ${root_directory}"
  exit 1
fi

echo "Searching for files matching '${filename_pattern}' in terminal directories under: ${root_directory}"
echo "WARNING: Files found will be deleted!"
read -p "Do you want to proceed? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Operation cancelled."
  exit 0
fi

# Find all directories under the root directory
find "${root_directory}" -type d | while read -r dir; do
  # Find out if there are any subdirectories under this directory
  subdirs=$(find "${dir}" -mindepth 1 -maxdepth 1 -type d -print -quit)

  # If no subdirectories exist, this is a "terminal" directory
  if [ -z "${subdirs}" ]; then
    # Look for files matching the pattern in this directory
    # Use find with -delete for efficiency and safety (handles spaces etc.)
    # Use -print first to show what would be deleted (optional)
    # find "${dir}" -maxdepth 1 -name "${filename_pattern}" -print
    num_deleted=$(find "${dir}" -maxdepth 1 -name "${filename_pattern}" -delete -print | wc -l)

    if [ "$num_deleted" -gt 0 ]; then
      echo "Deleted ${num_deleted} file(s) matching '${filename_pattern}' in: ${dir}"
    fi
  fi
done

echo "Cleanup complete."
