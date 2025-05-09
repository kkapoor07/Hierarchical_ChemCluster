# src/clustering/cluster_stage2.py
import logging
import os
import h5py
import numpy as np
import math
import time # Ensure time is imported for timing

from .faiss_kmeans_cpu import run_faiss_kmeans_cpu, find_nearest_data_points_to_centroids
from ..io.writers import save_fingerprints_h5 # Re-use writer for centers

logger = logging.getLogger(__name__)

def run_clustering_stage2_single_file(
    input_cluster_h5_path: str,
    output_results_dir: str,
    k2_ निर्धारण_method: str = 'ratio', # 'ratio', 'sqrt', 'fixed'
    k2_value: float = 8.0, # Ratio divisor, fixed k, or ignored for sqrt
    base_k: int = 2, # Minimum k value
    faiss_n_init: int = 1,
    faiss_max_iter: int = 25,
    faiss_seed: int | None = None,
    faiss_verbose: bool = False, # Typically less verbose for stage 2
    output_fp_dtype: np.dtype = np.int8,
    compression: str | None = "gzip",
    compression_opts: int = 4
    ) -> bool:
    """
    Performs the second stage of hierarchical clustering on a single input cluster file.

    Reads an HDF5 file (containing data for one cluster from Stage 1), determines the
    number of sub-clusters (k2) based on the chosen method, runs FAISS K-Means (CPU),
    finds representative points, and saves detailed assignments and representative centers
    to separate HDF5 files.

    Args:
        input_cluster_h5_path (str): Path to the input cluster HDF5 file (e.g., 'cluster_0.h5').
        output_results_dir (str): Directory where 'details_' and 'centers_' HDF5 files will be saved.
        k2_determination_method (str): Method to determine k2 ('ratio', 'sqrt', 'fixed').
        k2_value (float): Value used by the k2 determination method (ratio divisor or fixed k).
        base_k (int): Minimum number of sub-clusters to create. Defaults to 2.
        faiss_n_init (int): n_init/nredo parameter for FAISS K-Means.
        faiss_max_iter (int): max_iter parameter for FAISS K-Means.
        faiss_seed (int | None): Random seed for FAISS K-Means.
        faiss_verbose (bool): Verbosity flag for FAISS K-Means.
        output_fp_dtype (np.dtype): Dtype for saving center fingerprints.
        compression (str | None): Compression for output HDF5 files.
        compression_opts (int): Compression level.

    Returns:
        bool: True if the stage completes successfully for this file, False otherwise.
    """
    file_basename = os.path.basename(input_cluster_h5_path)
    logger.info(f"--- Starting Clustering Stage 2 for file: {file_basename} ---")

    # --- Load Data for this Cluster ---
    try:
        with h5py.File(input_cluster_h5_path, 'r') as h5f:
            if 'fp_list' not in h5f: # Basic check
                 logger.error(f"Dataset 'fp_list' not found in {input_cluster_h5_path}")
                 return False
            fp_data_stage2 = h5f['fp_list'][:]
            smiles_data_stage2 = h5f['smiles_list'][:] # Load metadata for saving centers/details
            name_data_stage2 = h5f['name_list'][:]
        num_compounds = fp_data_stage2.shape[0]
        logger.info(f"Loaded {num_compounds} data points for sub-clustering.")

        if num_compounds == 0:
            logger.warning(f"Input file {file_basename} is empty. Skipping Stage 2 clustering.")
            return True # Consider empty file processed successfully

    except Exception as e:
        logger.error(f"Failed to load data from {input_cluster_h5_path}: {e}", exc_info=True)
        return False

    # --- Determine Number of Sub-clusters (k2) ---
    n_clusters_k2 = 0
    if k2_determination_method == 'ratio':
        k_calc = math.ceil(num_compounds / max(1.0, k2_value)) # Avoid division by zero
    elif k2_determination_method == 'sqrt':
        k_calc = math.ceil(math.sqrt(num_compounds))
    elif k2_determination_method == 'fixed':
        k_calc = int(k2_value)
    else:
        logger.error(f"Invalid k2_determination_method: {k2_determination_method}. Using default ratio.")
        k_calc = math.ceil(num_compounds / 8.0) # Default to original script's logic

    # Ensure k2 is at least base_k and not more than the number of points
    n_clusters_k2 = max(base_k, k_calc)
    n_clusters_k2 = min(n_clusters_k2, num_compounds) # Cannot have more clusters than points

    if num_compounds < base_k:
         logger.warning(f"Number of compounds ({num_compounds}) is less than base_k ({base_k}). Setting k2 to {num_compounds}.")
         n_clusters_k2 = num_compounds # Each point is its own cluster

    logger.info(f"Determined number of sub-clusters (k2) = {n_clusters_k2} using method '{k2_determination_method}' (value={k2_value}, base_k={base_k}).")

    if n_clusters_k2 == 0: # Should only happen if num_compounds was 0 initially
         logger.warning("k2 is zero, cannot perform clustering.")
         return True # Already handled empty file case

    # --- Run K-Means (Stage 2) ---
    if fp_data_stage2.dtype != np.float32:
        logger.debug("Converting fingerprint data to float32 for FAISS.")
        fp_data_stage2 = fp_data_stage2.astype(np.float32)

    kmeans_obj_k2, D_k2, I_k2 = run_faiss_kmeans_cpu(
        data=fp_data_stage2,
        n_centroids=n_clusters_k2,
        n_init=faiss_n_init,
        max_iter=faiss_max_iter,
        verbose=faiss_verbose,
        seed=faiss_seed
    )

    if kmeans_obj_k2 is None or I_k2 is None or D_k2 is None:
        logger.error(f"FAISS K-Means execution failed for Stage 2 on file {file_basename}.")
        del fp_data_stage2, smiles_data_stage2, name_data_stage2 # Cleanup
        import gc; gc.collect()
        return False

    cluster_indices_k2 = I_k2.flatten()
    cluster_distances_k2 = D_k2.flatten()

    # --- Find Representative Points for Sub-clusters ---
    nearest_indices_orig, nearest_distances = find_nearest_data_points_to_centroids(
        data=fp_data_stage2, # Pass original data for correct indexing
        cluster_assignments=cluster_indices_k2,
        cluster_distances=cluster_distances_k2,
        n_centroids=n_clusters_k2
    )

    # Filter out entries for empty clusters if any occurred (-1 index)
    valid_center_indices = [idx for idx in nearest_indices_orig if idx != -1]
    if len(valid_center_indices) != n_clusters_k2:
         logger.warning(f"Found representatives for {len(valid_center_indices)} out of {n_clusters_k2} sub-clusters.")

    # --- Prepare Output Files ---
    try:
        os.makedirs(output_results_dir, exist_ok=True)
        # Define output filenames based on the input cluster file name
        detail_filename = os.path.join(output_results_dir, f"details_{file_basename}")
        centers_filename = os.path.join(output_results_dir, f"centers_{file_basename}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_results_dir}: {e}")
        del fp_data_stage2, smiles_data_stage2, name_data_stage2, kmeans_obj_k2, D_k2, I_k2 # Cleanup
        import gc; gc.collect()
        return False

    # --- Save Detailed Cluster Assignments ---
    try:
        start_time = time.perf_counter()
        with h5py.File(detail_filename, "w") as h5f_out:
            # Use modern string dtype for better compatibility if possible downstream
            smiles_dtype = h5py.string_dtype(encoding='utf-8')
            name_dtype = h5py.string_dtype(encoding='utf-8')

            h5f_out.create_dataset("SMILES", data=smiles_data_stage2, dtype=smiles_dtype, compression=compression, compression_opts=compression_opts) # Assumes bytes input, save as string
            h5f_out.create_dataset("NAME", data=name_data_stage2, dtype=name_dtype, compression=compression, compression_opts=compression_opts) # Assumes bytes input, save as string
            h5f_out.create_dataset("CLUSTER_k2", data=cluster_indices_k2, dtype=np.int32, compression=compression, compression_opts=compression_opts) # Save sub-cluster index
            h5f_out.create_dataset("DISTANCE_k2", data=cluster_distances_k2, dtype=np.float32, compression=compression, compression_opts=compression_opts) # Save distance

        end_time = time.perf_counter()
        logger.info(f"Saved detailed assignments ({num_compounds} points) to {detail_filename} in {end_time - start_time:.4f}s.")

    except Exception as e:
        logger.error(f"Failed to save details file {detail_filename}: {e}", exc_info=True)
        # Decide if this is a fatal error for the whole process? For now, log and continue.
        # return False # Uncomment to make it fatal

    # --- Save Representative Centers ---
    try:
        # Extract data only for the valid representative points
        center_fp = fp_data_stage2[valid_center_indices]
        # Ensure smiles/names are correctly indexed and converted to list of bytes for writer
        center_smiles = [smiles_data_stage2[i].tolist() for i in valid_center_indices] # Convert from array slice
        center_names = [name_data_stage2[i].tolist() for i in valid_center_indices]

        save_centers_success = save_fingerprints_h5(
            output_filename=centers_filename,
            fp_array=center_fp, # Pass the actual fingerprint data for centers
            smiles_list=center_smiles,
            name_list=center_names,
            fp_dtype=output_fp_dtype,
            compression=compression,
            compression_opts=compression_opts
        )

        if save_centers_success:
            logger.info(f"Saved {len(valid_center_indices)} representative centers to {centers_filename}")
        else:
            logger.error(f"Failed to save centers file {centers_filename}")
            # return False # Uncomment to make it fatal

    except Exception as e:
        logger.error(f"Failed during center data extraction or saving for {centers_filename}: {e}", exc_info=True)
        # return False # Uncomment to make it fatal

    # --- Cleanup ---
    del fp_data_stage2, smiles_data_stage2, name_data_stage2, kmeans_obj_k2, D_k2, I_k2
    import gc
    gc.collect()

    logger.info(f"--- Finished Clustering Stage 2 for file: {file_basename} ---")
    return True # Indicate success for this file
