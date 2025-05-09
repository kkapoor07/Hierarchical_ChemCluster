# scripts/utils/check_h5_contents.py
import argparse
import h5py
import sys
import numpy as np

def check_h5(filepath, list_keys=True, show_details=True, view_dataset=None, view_rows=10):
    """
    Inspects an HDF5 file, listing keys, showing dataset details,
    and optionally viewing the first few rows of a dataset.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return False

    print(f"Inspecting HDF5 file: {filepath}")
    print("-" * 60)

    try:
        with h5py.File(filepath, 'r') as h5f:
            root_keys = list(h5f.keys())
            if list_keys:
                print("Root Level Keys:", root_keys if root_keys else "None")
                print("-" * 60)

            if show_details:
                print("Dataset Details:")
                for key in root_keys:
                    try:
                        item = h5f[key]
                        if isinstance(item, h5py.Dataset):
                            print(f"  - Dataset: '{key}'")
                            print(f"      Shape: {item.shape}")
                            print(f"      Dtype: {item.dtype}")
                            print(f"      Chunks: {item.chunks}")
                            print(f"      Compression: {item.compression} (Opts: {item.compression_opts})")
                        elif isinstance(item, h5py.Group):
                             print(f"  - Group: '{key}' (Contains: {list(item.keys())})") # Basic group info
                        else:
                             print(f"  - Item: '{key}' (Type: {type(item)})")

                    except Exception as e:
                       print(f"  - Error accessing key '{key}': {e}")
                print("-" * 60)

            if view_dataset:
                if view_dataset not in h5f:
                    print(f"Error: Dataset '{view_dataset}' not found in the file.", file=sys.stderr)
                elif not isinstance(h5f[view_dataset], h5py.Dataset):
                    print(f"Error: '{view_dataset}' is not a dataset.", file=sys.stderr)
                else:
                    ds = h5f[view_dataset]
                    print(f"Viewing first {view_rows} rows of dataset '{view_dataset}':")
                    if ds.shape[0] == 0:
                        print("  (Dataset is empty)")
                    else:
                        num_to_show = min(view_rows, ds.shape[0])
                        data_slice = ds[:num_to_show]
                        # Pretty print, handling bytes decoding and potential nested arrays
                        for i, row in enumerate(data_slice):
                            try:
                                if ds.dtype == h5py.special_dtype(vlen=bytes) or isinstance(row, bytes):
                                    row_str = row.decode('utf-8', errors='replace')
                                elif ds.dtype == h5py.string_dtype(encoding='utf-8') or isinstance(row, str):
                                    row_str = row
                                elif isinstance(row, np.ndarray) and row.size == 1: # Handle [[b'BYTES']] or [['STR']]
                                    item = row.item(0)
                                    row_str = item.decode('utf-8', errors='replace') if isinstance(item, bytes) else str(item)
                                else: # Numeric or other
                                    row_str = str(row)
                                print(f"  Row {i}: {row_str}")
                            except Exception as view_err:
                                print(f"  Row {i}: Error viewing data - {view_err}, Raw: {row}")

                    print("-" * 60)

        return True

    except Exception as e:
        print(f"An error occurred opening or reading the HDF5 file: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 file contents.")
    parser.add_argument("filepath", help="Path to the HDF5 file.")
    parser.add_argument("--list", action='store_true', default=True, help="List root keys (default: True). Use --no-list to disable.")
    parser.add_argument("--no-list", action='store_false', dest='list', help="Do not list root keys.")
    parser.add_argument("--details", action='store_true', default=True, help="Show details (shape, dtype, etc.) for datasets (default: True). Use --no-details to disable.")
    parser.add_argument("--no-details", action='store_false', dest='details', help="Do not show dataset details.")
    parser.add_argument("--view", metavar="DATASET_NAME", help="Name of the dataset to view the first few rows.")
    parser.add_argument("-n", "--rows", type=int, default=10, help="Number of rows to view (default: 10).")

    # Need 'import os' for checking file existence
    import os

    args = parser.parse_args()

    if not check_h5(args.filepath, args.list, args.details, args.view, args.rows):
        sys.exit(1)
