#!/bin/bash

# find_missing_downloads.sh
#
# Purpose: Checks a ZINC download directory structure for completeness.
#          Identifies terminal directories (those without subdirectories)
#          that are missing the expected SMILES file (*.smi).
# Usage:   ./find_missing_downloads.sh /path/to/zinc/download/root

if [ -z "$1" ]; then
  echo "Usage: $0 <root_directory>"
  exit 1
fi

root_directory="$1"

if [ ! -d "$root_directory" ]; then
  echo "Error: Root directory not found: ${root_directory}"
  exit 1
fi

echo "Searching for terminal directories missing .smi files under: ${root_directory}"

# Find all directories under the root directory
find "${root_directory}" -type d | while read -r dir; do
  # Find out if there are any subdirectories under this directory
  # Using -print -quit ensures find stops after finding the first subdir
  subdirs=$(find "${dir}" -mindepth 1 -maxdepth 1 -type d -print -quit)

  # If no subdirectories exist, this is a "terminal" directory
  if [ -z "${subdirs}" ]; then
    # Look for .smi files specifically in this directory (not recursive)
    # Using -print -quit to quickly check existence
    smi_files=$(find "${dir}" -maxdepth 1 -name "*.smi" -print -quit)

    # If no .smi files found, report the directory
    if [ -z "${smi_files}" ]; then
      echo "Missing .smi file in terminal directory: ${dir}"
    fi
  fi
done

echo "Search complete."
