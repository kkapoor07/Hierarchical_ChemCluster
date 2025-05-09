# scripts/utils/compress_h5_file.py
import argparse
import h5py
import os
import sys
import time

def compress_h5(input_file, output_file, compression="gzip", compression_opts=9):
    """
    Copies datasets from an input HDF5 file to an output file with specified compression.

    Args:
        input_file (str): Path to the source HDF5 file.
        output_file (str): Path to the destination compressed HDF5 file.
        compression (str): Compression algorithm ('gzip', 'lzf').
        compression_opts (int): Compression level (1-9 for gzip).

    Returns:
        bool: True on success, False otherwise.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return False
    if compression not in ['gzip', 'lzf']:
        print(f"Error: Unsupported compression type: {compression}", file=sys.stderr)
        return False

    print(f"Compressing {input_file} -> {output_file} using {compression}({compression_opts})")
    start_time = time.perf_counter()

    try:
        with h5py.File(input_file, 'r') as source, h5py.File(output_file, 'w') as dest:
            total_datasets = len(source.keys())
            print(f"Found {total_datasets} datasets to copy.")
            count = 0
            for name, dataset in source.items():
                count += 1
                print(f"Processing dataset {count}/{total_datasets}: {name} (Shape: {dataset.shape}, Dtype: {dataset.dtype})")
                # Copy dataset with specified compression
                # Consider adding chunking options if needed for very large datasets
                dest.create_dataset(
                    name,
                    data=dataset, # Reads source dataset into memory - potential issue for huge datasets
                    # To handle huge datasets, might need chunked read/write:
                    # chunks = dataset.chunks # Get chunking from source if available
                    # dest_ds = dest.create_dataset(name, shape=dataset.shape, dtype=dataset.dtype, chunks=chunks, compression=compression, compression_opts=compression_opts)
                    # for chunk_slice in dataset.iter_chunks():
                    #    dest_ds[chunk_slice] = dataset[chunk_slice]
                    # Simpler copy for now:
                    compression=compression,
                    compression_opts=compression_opts
                )
        end_time = time.perf_counter()
        print(f"Compression completed successfully in {end_time - start_time:.2f} seconds.")
        return True
    except Exception as e:
        print(f"An error occurred during compression: {e}", file=sys.stderr)
        # Clean up partial output file
        if os.path.exists(output_file):
            try: os.remove(output_file)
            except OSError: pass
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress an HDF5 file.")
    parser.add_argument("input_file", help="Path to the source HDF5 file.")
    parser.add_argument("output_file", help="Path to the destination compressed HDF5 file.")
    parser.add_argument("-c", "--compression", default="gzip", choices=['gzip', 'lzf'], help="Compression type.")
    parser.add_argument("-l", "--level", type=int, default=9, help="Compression level (1-9 for gzip).")

    args = parser.parse_args()

    if not compress_h5(args.input_file, args.output_file, args.compression, args.level):
        sys.exit(1)
