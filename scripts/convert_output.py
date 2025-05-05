# scripts/convert_output.py
import argparse
import logging
import os
import sys
from glob import glob

# --- Setup sys.path ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.common.logging_utils import setup_logging
    from src.io.writers import hdf5_to_smi, hdf5_details_to_csv
except ImportError as e:
    print(f"Error importing project modules: {e}. Ensure script is run from the correct directory or PYTHONPATH is set.")
    sys.exit(1)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 output files (aggregated centers or cluster details) to other formats (SMI, CSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", help="Input HDF5 file path OR directory containing HDF5 files (for converting details).")
    parser.add_argument("output_path", help="Output file path (for SMI conversion) OR output directory (for CSV conversion).")
    parser.add_argument("--mode", required=True, choices=['centers_to_smi', 'details_to_csv'], help="Conversion mode.")

    # Options for centers_to_smi
    parser.add_argument("--smiles_dset", default='smiles_list', help="Dataset name for SMILES (for centers_to_smi).")
    parser.add_argument("--name_dset", default='name_list', help="Dataset name for IDs/Names (for centers_to_smi).")
    parser.add_argument("--smi_delimiter", default=' ', help="Delimiter for output SMI file.")

    # Options for details_to_csv
    parser.add_argument("--details_pattern", default="details_cluster_*.h5", help="Glob pattern for input HDF5 detail files if input_path is a directory.")
    parser.add_argument("--csv_cols", nargs='+', default=['SMILES', 'NAME', 'CLUSTER_k2', 'DISTANCE_k2'], help="Columns (HDF5 datasets) to include in output CSV.")

    # General options
    parser.add_argument("--log_file", default="logs/convert_output.log", help="Path to log file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(args.log_file, level=log_level)

    start_time_overall = time.perf_counter() # Standard library time

    if args.mode == 'centers_to_smi':
        if not os.path.isfile(args.input_path):
            logger.error(f"Input path must be a file for mode 'centers_to_smi': {args.input_path}")
            sys.exit(1)
        # Ensure output directory exists if output_path includes directory part
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Starting conversion: {args.input_path} -> {args.output_path} (SMI format)")
        success = hdf5_to_smi(
            input_h5_path=args.input_path,
            output_smi_path=args.output_path,
            smiles_dset=args.smiles_dset,
            name_dset=args.name_dset,
            delimiter=args.smi_delimiter
        )
        if not success:
            logger.error("Conversion failed.")
            sys.exit(1)

    elif args.mode == 'details_to_csv':
        if not os.path.isdir(args.input_path):
            logger.error(f"Input path must be a directory for mode 'details_to_csv': {args.input_path}")
            sys.exit(1)
        os.makedirs(args.output_path, exist_ok=True) # Output path is directory for CSVs

        # Find detail files
        search_pattern = os.path.join(args.input_path, args.details_pattern)
        detail_files = glob(search_pattern)
        if not detail_files:
            logger.warning(f"No files found matching pattern '{search_pattern}'. Exiting.")
            sys.exit(0)

        logger.info(f"Found {len(detail_files)} detail files to convert to CSV in {args.output_path}.")
        success_count = 0
        fail_count = 0
        for input_file in detail_files:
            base_name = os.path.basename(input_file)
            output_filename = os.path.splitext(base_name)[0] + ".csv"
            output_filepath = os.path.join(args.output_path, output_filename)
            logger.info(f"Converting: {input_file} -> {output_filepath}")
            success = hdf5_details_to_csv(
                input_h5_path=input_file,
                output_csv_path=output_filepath,
                datasets_to_include=args.csv_cols
            )
            if success:
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"Finished CSV conversion. Success: {success_count}, Failed: {fail_count}")
        if fail_count > 0:
             sys.exit(1) # Exit with error if any conversions failed

    else:
        # Should not happen due to choices in argparse
        logger.error(f"Invalid mode selected: {args.mode}")
        sys.exit(1)

    end_time_overall = time.perf_counter()
    logger.info(f"Conversion process completed in {end_time_overall - start_time_overall:.2f} seconds.")

if __name__ == "__main__":
    main()
