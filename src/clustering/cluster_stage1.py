# src/clustering/cluster_stage1.py
import logging
import os
import h5py
import numpy as np
from .faiss_kmeans_cpu import run_faiss_kmeans_cpu
from ..io.writers import save_fingerprints_h5 # Use refactored writer

logger = logging.getLogger(__name__)

def run_clustering_stage1(
    input_h5_path: str,
    output_dir: str,
    n_clusters_k1: int,
    faiss_n_init: int = 1,
    faiss_max_iter: int = 25,
    faiss_seed: int | None = None,
    faiss_verbose: bool = True,
    output_fp_dtype: np.dtype = np.int8,
    compression: str | None = "gzip",
    compression_opts: int = 4
    ) -> bool:
    """
    Performs the first stage of hierarchical clustering.

    Reads a large HDF5 file containing fingerprints and metadata, runs FAISS K-Means (CPU)
    to partition the data into k1 clusters, and saves each cluster's data into
    separate HDF5 files in the specified output directory.

    Args:
        input_h5_path (str): Path to the large input HDF5 file (containing ['fp_list', 'smiles_list', 'name_list']).
        output_dir (str): Directory to save the individual cluster_*.h5 files.
        n_clusters_k1 (int): Number of initial clusters (k1) for this stage.
        faiss_n_init (int): n_init/nredo parameter for FAISS K-Means.
        faiss_max_iter (int): max_iter parameter for FAISS K-Means.
        faiss_seed (int | None): Random seed for FAISS K-Means.
        faiss_verbose (bool): Verbosity flag for FAISS K-Means.
        output_fp_dtype (np.dtype): Dtype for saving fingerprints in output files.
        compression (str | None): Compression for output HDF5 files.
        compression_opts (int): Compression level.

    Returns:
        bool: True if the stage completes successfully, False otherwise.
    """
    logger.info(f"--- Starting Clustering Stage 1 ---")
    logger.info(f"Input HDF5: {input_h5_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Number of Initial Clusters (k1): {n_clusters_k1}")

    # --- Load Data ---
    # CRITICAL: This loads the entire dataset. Document the high RAM requirement.
    logger.warning(f"Loading entire dataset from {input_h5_path} into memory. This requires significant RAM!")
    try:
        with h5py.File(input_h5_path, 'r') as h5f:
            # Check if datasets exist
            if 'fp_list' not in h5f or 'smiles_list' not in h5f or 'name_list' not in h5f:
                 logger.error("Input HDF5 file must contain 'fp_list', 'smiles_list', and 'name_list' datasets.")
                 return False

            fp_data = h5f['fp_list'][:]
            # Load metadata ensuring correct handling of bytes/strings based on storage
            smiles_data = h5f['smiles_list'][:] # Assuming stored as bytes based on previous scripts
            name_data = h5f['name_list'][:]   # Assuming stored as bytes based on previous scripts

        logger.info(f"Loaded data: Fingerprints shape={fp_data.shape}, SMILES count={len(smiles_data)}, Names count={len(name_data)}")
        # Basic validation
        if not (fp_data.shape[0] == len(smiles_data) == len(name_data)):
             logger.error("Mismatch in number of entries between fingerprints, SMILES, and names.")
             return False
        if fp_data.shape[0] == 0:
            logger.warning("Input dataset is empty. Stage 1 finished with no output.")
            return True

    except Exception as e:
        logger.error(f"Failed to load data from {input_h5_path}: {e}", exc_info=True)
        return False

    # --- Run K-Means ---
    # Ensure data is float32 for FAISS
    if fp_data.dtype != np.float32:
        logger.debug("Converting fingerprint data to float32 for FAISS.")
        fp_data = fp_data.astype(np.float32)

    kmeans_obj, D, I = run_faiss_kmeans_cpu(
        data=fp_data,
        n_centroids=n_clusters_k1,
        n_init=faiss_n_init,
        max_iter=faiss_max_iter,
        verbose=faiss_verbose,
        seed=faiss_seed
    )

    if kmeans_obj is None or I is None:
        logger.error("FAISS K-Means execution failed for Stage 1.")
        # Clean up large arrays
        del fp_data, smiles_data, name_data
        import gc; gc.collect()
        return False

    cluster_indices = I.flatten() # Ensure it's a 1D array

    # --- Prepare Output Directory ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        # Clean up large arrays
        del fp_data, smiles_data, name_data, cluster_indices, kmeans_obj, D, I
        import gc; gc.collect()
        return False

    # --- Save Clusters to Separate Files ---
    logger.info(f"Saving data points to individual cluster files in {output_dir}...")
    clusters_saved = 0
    total_points_saved = 0
    start_save_time = time.perf_counter() # Import time standard library

    for i in range(n_clusters_k1):
        # Find indices of data points belonging to the current cluster
        indices_in_cluster = np.where(cluster_indices == i)[0]
        num_points_in_cluster = len(indices_in_cluster)
        logger.debug(f"Cluster {i}: Found {num_points_in_cluster} data points.")

        if num_points_in_cluster > 0:
            # Define output path for this cluster's HDF5 file
            cluster_file_path = os.path.join(output_dir, f'cluster_{i}.h5')

            # Extract data for this cluster
            # Use advanced indexing for potentially better memory efficiency if needed,
            # but direct indexing is usually fine here after the main load.
            cluster_fp = fp_data[indices_in_cluster]
            cluster_smiles = smiles_data[indices_in_cluster]
            cluster_names = name_data[indices_in_cluster]

            # Save using the refactored writer function
            save_success = save_fingerprints_h5(
                output_filename=cluster_file_path,
                fp_array=cluster_fp,
                smiles_list=cluster_smiles.tolist(), # Convert back to list of bytes for writer
                name_list=cluster_names.tolist(),   # Convert back to list of bytes for writer
                fp_dtype=output_fp_dtype,
                compression=compression,
                compression_opts=compression_opts
            )

            if save_success:
                clusters_saved += 1
                total_points_saved += num_points_in_cluster
                logger.debug(f"Successfully saved cluster {i} to {cluster_file_path}")
            else:
                logger.error(f"Failed to save cluster {i} to {cluster_file_path}")
                # Optionally: Decide whether to continue or abort on single file save failure
        else:
            logger.warning(f"Cluster {i} is empty. No file created.")

    end_save_time = time.perf_counter()
    logger.info(f"Finished saving cluster data in {end_save_time - start_save_time:.4f} seconds.")
    logger.info(f"Saved {clusters_saved} non-empty clusters containing a total of {total_points_saved} data points.")
    if total_points_saved != fp_data.shape[0]:
         logger.warning(f"Mismatch between total points saved ({total_points_saved}) and original data points ({fp_data.shape[0]}). Check logs.")

    # --- Cleanup ---
    logger.debug("Cleaning up large arrays from memory.")
    del fp_data, smiles_data, name_data, cluster_indices, kmeans_obj, D, I
    import gc
    gc.collect()

    logger.info(f"--- Clustering Stage 1 Finished ---")
    return True # Indicate overall success (even if some files failed saving, maybe?) Depends on desired strictness.
