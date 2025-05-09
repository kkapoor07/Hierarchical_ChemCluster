# src/io/writers.py
import h5py
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def save_fingerprints_h5(
    output_filename: str,
    fp_array: np.ndarray,
    smiles_list: list[bytes], # Expect bytes based on original code
    name_list: list[bytes],   # Expect bytes based on original code
    fp_dtype: np.dtype = np.int8, # Make dtype explicit
    compression: str | None = "gzip", # Add optional compression
    compression_opts: int = 4 # Moderate compression level
    ):
    """
    Writes fingerprint data (fingerprints, SMILES, names) to an HDF5 file.

    Uses variable-length bytes for SMILES and names, matching the original script's behavior.

    Args:
        output_filename (str): Path to the output HDF5 file.
        fp_array (np.ndarray): NumPy array of fingerprints (e.g., shape N x bits).
        smiles_list (list[bytes]): List of SMILES strings encoded as bytes.
        name_list (list[bytes]): List of molecule names encoded as bytes.
        fp_dtype (np.dtype): NumPy dtype for storing fingerprints. Defaults to np.int8.
        compression (str | None): Compression algorithm ('gzip', 'lzf', None). Defaults to 'gzip'.
        compression_opts (int): Compression level (e.g., 1-9 for gzip). Defaults to 4.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    logger.debug(f"Attempting to save {len(smiles_list)} fingerprints to {output_filename}")

    # Ensure directory exists
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create directory for {output_filename}: {e}")
        return False

    # Define HDF5 variable-length bytes type
    # Use newer h5py string dtype if preferred and compatible downstream
    # dt_smiles = h5py.string_dtype(encoding='utf-8') # Modern approach
    # dt_names = h5py.string_dtype(encoding='utf-8')
    dt_bytes = h5py.special_dtype(vlen=bytes) # Original approach

    try:
        with h5py.File(output_filename, 'w') as h5f:
            # --- Fingerprints ---
            if fp_array.size > 0: # Handle empty arrays
                ds_fp = h5f.create_dataset(
                    'fp_list',
                    data=fp_array.astype(fp_dtype), # Ensure correct dtype
                    compression=compression,
                    compression_opts=compression_opts
                )
                logger.debug(f"Saved 'fp_list' dataset with shape {ds_fp.shape} and dtype {ds_fp.dtype}")
            else:
                # Create an empty dataset with appropriate shape/dtype if input is empty
                shape = (0,) + fp_array.shape[1:] if fp_array.ndim > 1 else (0,)
                h5f.create_dataset('fp_list', shape=shape, dtype=fp_dtype)
                logger.warning(f"Input fp_array is empty. Created empty 'fp_list' dataset.")

            # --- SMILES List ---
            # Reshape lists to N x 1 for consistency with original save_data
            smiles_array = np.array(smiles_list, dtype=object).reshape(-1, 1)
            ds_smiles = h5f.create_dataset(
                'smiles_list',
                shape=smiles_array.shape,
                dtype=dt_bytes, # Use original bytes dtype
                data=smiles_array,
                compression=compression, # Compress metadata too
                compression_opts=compression_opts
            )
            logger.debug(f"Saved 'smiles_list' dataset with shape {ds_smiles.shape}")

            # --- Name List ---
            name_array = np.array(name_list, dtype=object).reshape(-1, 1)
            ds_names = h5f.create_dataset(
                'name_list',
                shape=name_array.shape,
                dtype=dt_bytes, # Use original bytes dtype
                data=name_array,
                compression=compression,
                compression_opts=compression_opts
            )
            logger.debug(f"Saved 'name_list' dataset with shape {ds_names.shape}")

        logger.info(f"Successfully saved data to {output_filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to save data to HDF5 file {output_filename}: {e}", exc_info=True)
        # Clean up potentially partially written file
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                logger.warning(f"Removed partially written file: {output_filename}")
            except OSError as rm_err:
                logger.error(f"Could not remove partially written file {output_filename}: {rm_err}")
        return False

def hdf5_to_smi(
    input_h5_path: str,
    output_smi_path: str,
    smiles_dset: str = 'smiles_list',
    name_dset: str = 'name_list',
    delimiter: str = ' ',
    write_header: bool = False
    ) -> bool:
    """
    Extracts SMILES and Names from an HDF5 file and writes them to a SMILES file.

    Args:
        input_h5_path (str): Path to the input HDF5 file.
        output_smi_path (str): Path to the output SMILES (.smi) file.
        smiles_dset (str): Name of the dataset containing SMILES strings.
        name_dset (str): Name of the dataset containing molecule names/IDs.
        delimiter (str): Delimiter to use in the output SMI file.
        write_header (bool): Whether to write a header line (e.g., "smiles name").

    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"Converting HDF5 {input_h5_path} to SMI file {output_smi_path}")
    try:
        with h5py.File(input_h5_path, 'r') as h5f, open(output_smi_path, 'w', encoding='utf-8') as outfile:
            if smiles_dset not in h5f or name_dset not in h5f:
                logger.error(f"Required datasets '{smiles_dset}' or '{name_dset}' not found in {input_h5_path}")
                return False

            smiles_list_ds = h5f[smiles_dset]
            name_list_ds = h5f[name_dset]
            num_entries = smiles_list_ds.shape[0]
            logger.info(f"Found {num_entries} entries to write.")

            if write_header:
                outfile.write(f"{smiles_dset}{delimiter}{name_dset}\n") # Use dataset names as header? Or fixed 'smiles name'?

            # Iterate and write, handling potential nested arrays and decoding
            # Process in chunks if datasets are huge to avoid loading all names/smiles at once
            chunk_size = 100000 # Process 100k entries at a time
            for i in range(0, num_entries, chunk_size):
                smiles_chunk = smiles_list_ds[i:min(i + chunk_size, num_entries)]
                names_chunk = name_list_ds[i:min(i + chunk_size, num_entries)]

                for smiles_val, name_val in zip(smiles_chunk, names_chunk):
                    # Decode bytes if necessary (adjust based on how they are stored)
                    # Check if stored as vlen bytes (like original) or vlen string
                    try:
                        if smiles_list_ds.dtype == h5py.special_dtype(vlen=bytes) or isinstance(smiles_val, bytes):
                            smiles = smiles_val.decode('utf-8')
                        elif smiles_list_ds.dtype == h5py.string_dtype(encoding='utf-8') or isinstance(smiles_val, str):
                             smiles = smiles_val
                        elif isinstance(smiles_val, np.ndarray) and smiles_val.size > 0: # Handle [[b'CCO']] case
                             smiles = smiles_val.item(0).decode('utf-8') if isinstance(smiles_val.item(0), bytes) else str(smiles_val.item(0))
                        else:
                             smiles = str(smiles_val) # Fallback

                        if name_list_ds.dtype == h5py.special_dtype(vlen=bytes) or isinstance(name_val, bytes):
                            name = name_val.decode('utf-8')
                        elif name_list_ds.dtype == h5py.string_dtype(encoding='utf-8') or isinstance(name_val, str):
                            name = name_val
                        elif isinstance(name_val, np.ndarray) and name_val.size > 0: # Handle [[b'ID']] case
                             name = name_val.item(0).decode('utf-8') if isinstance(name_val.item(0), bytes) else str(name_val.item(0))
                        else:
                            name = str(name_val) # Fallback

                        outfile.write(f"{smiles}{delimiter}{name}\n")
                    except Exception as decode_err:
                        logger.warning(f"Skipping entry due to decode/processing error: {decode_err} - SMILES: {smiles_val}, Name: {name_val}")

        logger.info(f"Successfully wrote SMILES file: {output_smi_path}")
        return True
    except FileNotFoundError:
        logger.error(f"Input HDF5 file not found: {input_h5_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to convert HDF5 to SMI: {e}", exc_info=True)
        # Clean up potentially partially written file
        if os.path.exists(output_smi_path):
            try:
                os.remove(output_smi_path)
                logger.warning(f"Removed partially written file: {output_smi_path}")
            except OSError as rm_err:
                logger.error(f"Could not remove partially written file {output_smi_path}: {rm_err}")
        return False


def hdf5_details_to_csv(
    input_h5_path: str,
    output_csv_path: str,
    datasets_to_include: list[str] | None = None, # Explicitly list expected datasets
    chunk_size: int = 100000 # Process in chunks for memory efficiency
    ) -> bool:
    """
    Extracts datasets (like SMILES, Name, Cluster ID, Distance) from an HDF5 file
    (typically a 'details' file from clustering) and writes them to a CSV file.

    Args:
        input_h5_path (str): Path to the input HDF5 file.
        output_csv_path (str): Path to the output CSV file.
        datasets_to_include (list[str] | None): List of dataset names to include as columns.
                                                If None, uses default ['SMILES', 'NAME', 'CLUSTER_k2', 'DISTANCE_k2'].
        chunk_size (int): Number of rows to process at a time.

    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"Converting HDF5 details {input_h5_path} to CSV file {output_csv_path}")

    if datasets_to_include is None:
        # Define default datasets expected in the 'details' files from stage 2
        datasets_to_include = ['SMILES', 'NAME', 'CLUSTER_k2', 'DISTANCE_k2']

    try:
        with h5py.File(input_h5_path, 'r') as h5f:
            # Validate that all requested datasets exist and get total size
            num_entries = -1
            dset_handles = {}
            for ds_name in datasets_to_include:
                if ds_name not in h5f:
                    logger.error(f"Required dataset '{ds_name}' not found in {input_h5_path}")
                    return False
                dset_handles[ds_name] = h5f[ds_name]
                current_len = dset_handles[ds_name].shape[0]
                if num_entries == -1:
                    num_entries = current_len
                elif num_entries != current_len:
                    logger.error(f"Dataset length mismatch: '{ds_name}' ({current_len}) vs previous ({num_entries}) in {input_h5_path}")
                    return False

            if num_entries == -1 or num_entries == 0:
                logger.warning(f"No entries found or datasets missing in {input_h5_path}. Creating empty CSV.")
                 # Create empty CSV with header
                with open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(datasets_to_include) # Write header
                return True

            logger.info(f"Found {num_entries} entries across {len(datasets_to_include)} datasets.")

            # Write CSV chunk by chunk
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile)
                # Write header
                writer.writerow(datasets_to_include)

                # Iterate through chunks
                for i in range(0, num_entries, chunk_size):
                    end_index = min(i + chunk_size, num_entries)
                    logger.debug(f"Processing rows {i} to {end_index-1}")
                    chunk_data = []
                    # Read chunk for each dataset
                    for ds_name in datasets_to_include:
                        data_chunk = dset_handles[ds_name][i:end_index]
                        # Handle potential string decoding (assuming stored as bytes or string dtype)
                        if data_chunk.dtype == h5py.special_dtype(vlen=bytes):
                             chunk_data.append([val.decode('utf-8') for val in data_chunk])
                        elif data_chunk.dtype == h5py.string_dtype(encoding='utf-8'):
                             chunk_data.append(data_chunk.tolist()) # Convert string array to list
                        elif isinstance(data_chunk, np.ndarray) and data_chunk.ndim > 1 and data_chunk.shape[1] == 1:
                             # Handle [[b'BYTES']] or [['STRING']] case
                             flat_chunk = data_chunk.flatten()
                             if flat_chunk.dtype == 'O': # Likely object array containing bytes/strings
                                if len(flat_chunk) > 0 and isinstance(flat_chunk[0], bytes):
                                     chunk_data.append([val.decode('utf-8') for val in flat_chunk])
                                else:
                                     chunk_data.append(flat_chunk.astype(str).tolist())
                             else: # Standard numeric etc
                                 chunk_data.append(flat_chunk.tolist())
                        else:
                             # Assume numeric or simple types
                             chunk_data.append(data_chunk.tolist())

                    # Transpose data and write rows
                    # zip(*chunk_data) transposes the list of lists/arrays
                    rows_to_write = zip(*chunk_data)
                    writer.writerows(rows_to_write)

        logger.info(f"Successfully wrote CSV file: {output_csv_path}")
        return True

    except FileNotFoundError:
        logger.error(f"Input HDF5 file not found: {input_h5_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to convert HDF5 details to CSV: {e}", exc_info=True)
        # Clean up potentially partially written file
        if os.path.exists(output_csv_path):
            try:
                os.remove(output_csv_path)
                logger.warning(f"Removed partially written CSV file: {output_csv_path}")
            except OSError as rm_err:
                logger.error(f"Could not remove partially written CSV file {output_csv_path}: {rm_err}")
        return False
