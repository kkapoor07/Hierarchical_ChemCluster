# src/clustering/faiss_kmeans_cpu.py
import logging
import sys
import faiss
import numpy as np

logger = logging.getLogger(__name__)

def run_faiss_kmeans_cpu(
    data: np.ndarray,
    n_centroids: int,
    n_init: int = 1, # Default from faiss Kmeans is 1
    max_iter: int = 25, # Default from faiss Kmeans is 25
    verbose: bool = True,
    seed: int | None = None # Add seed for reproducibility
    ) -> tuple[faiss.Kmeans | None, np.ndarray | None, np.ndarray | None]:
    """
    Performs K-Means clustering using faiss-cpu.

    Args:
        data (np.ndarray): Input data array (N x Dimensions). Must be float32 for FAISS.
        n_centroids (int): Number of clusters (k).
        n_init (int): Number of times to run K-Means with different centroid seeds.
                      The result with the lowest inertia is chosen. Defaults to 1.
        max_iter (int): Maximum number of iterations for a single K-Means run. Defaults to 25.
        verbose (bool): Whether FAISS K-Means should be verbose. Defaults to True.
        seed (int | None): Random seed for FAISS K-Means initialization. Defaults to None (random).

    Returns:
        tuple[faiss.Kmeans | None, np.ndarray | None, np.ndarray | None]:
            - kmeans (faiss.Kmeans object): The trained K-Means object, or None on failure.
            - D (np.ndarray): Distances of each point to its assigned centroid (N x 1), or None.
            - I (np.ndarray): Index of the cluster assigned to each point (N x 1), or None.
            Returns (None, None, None) if input data is invalid or clustering fails.
    """
    if data is None or data.ndim != 2 or data.shape[0] == 0:
        logger.error("Invalid input data array provided for K-Means.")
        return None, None, None
    if n_centroids <= 0:
         logger.error(f"Number of centroids must be positive, got {n_centroids}.")
         return None, None, None
    if n_centroids > data.shape[0]:
        logger.warning(f"Number of centroids ({n_centroids}) is greater than the number of data points ({data.shape[0]}). Setting k to number of points.")
        n_centroids = data.shape[0]

    n_samples, n_features = data.shape
    logger.info(f"Running FAISS K-Means (CPU) on {n_samples} points, {n_features} dimensions, k={n_centroids}, n_init={n_init}, max_iter={max_iter}.")

    # Ensure data is float32
    if data.dtype != np.float32:
        logger.debug("Input data is not float32. Converting...")
        data = data.astype(np.float32)

    try:
        kmeans = faiss.Kmeans(
            d=n_features,
            k=n_centroids,
            niter=max_iter,
            nredo=n_init, # nredo corresponds to n_init
            verbose=verbose,
            seed=seed if seed is not None else np.random.randint(0, 10000) # FAISS needs explicit seed or uses default
        )
        kmeans.train(data)

        if not kmeans.index:
             logger.error("FAISS K-Means training completed but index is not available.")
             return None, None, None

        logger.info("FAISS K-Means training completed.")

        # Assign points to clusters
        # index.search returns distances (D) and indices (I) of the nearest cluster(s)
        D, I = kmeans.index.search(data, 1) # Search for the 1 nearest cluster for each point
        logger.info("Cluster assignment (search) completed.")

        return kmeans, D, I

    except Exception as e:
        logger.error(f"FAISS K-Means failed: {e}", exc_info=True)
        return None, None, None


def find_nearest_data_points_to_centroids(
    data: np.ndarray,
    cluster_assignments: np.ndarray, # Cluster index for each data point (I from search)
    cluster_distances: np.ndarray,   # Distance to assigned cluster (D from search)
    n_centroids: int
    ) -> tuple[list[int], list[float]]:
    """
    Finds the index of the data point closest to each cluster centroid.

    This identifies a representative data point for each cluster based on minimum
    distance to the centroid it's assigned to.

    Args:
        data (np.ndarray): The original input data array (N x Dim).
        cluster_assignments (np.ndarray): Array of cluster indices for each point (shape N x 1 or N,).
        cluster_distances (np.ndarray): Array of distances to assigned cluster (shape N x 1 or N,).
        n_centroids (int): The total number of centroids (k).

    Returns:
        tuple[list[int], list[float]]:
            - nearest_indices (list[int]): List where index `i` contains the original index
                                           (in `data`) of the point closest to centroid `i`.
                                           Contains -1 if a cluster is empty.
            - nearest_distances (list[float]): List where index `i` contains the distance
                                               of the closest point to centroid `i`.
                                               Contains sys.float_info.max if cluster is empty.
    """
    if data is None or cluster_assignments is None or cluster_distances is None:
        logger.error("Invalid input for finding nearest points.")
        return [], []
    if len(data) != len(cluster_assignments) or len(data) != len(cluster_distances):
         logger.error("Input data, assignments, and distances must have the same length.")
         return [], []

    assignments_flat = cluster_assignments.flatten()
    distances_flat = cluster_distances.flatten()

    # Initialize lists to store the index and distance of the nearest point for each centroid
    nearest_indices = [-1] * n_centroids
    nearest_distances = [sys.float_info.max] * n_centroids

    logger.debug(f"Finding nearest data points for {n_centroids} centroids...")
    start_time = time.perf_counter() # Use time from stdlib if not already imported

    for data_idx, (distance, cluster_idx) in enumerate(zip(distances_flat, assignments_flat)):
        # Check if this point is closer than the current closest point found for this cluster
        if 0 <= cluster_idx < n_centroids: # Ensure cluster_idx is valid
            if distance < nearest_distances[cluster_idx]:
                nearest_distances[cluster_idx] = distance
                nearest_indices[cluster_idx] = data_idx
        else:
            logger.warning(f"Data point {data_idx} assigned to invalid cluster index {cluster_idx}. Skipping.")

    end_time = time.perf_counter()
    logger.debug(f"Finished finding nearest points in {end_time - start_time:.4f} seconds.")

    # Check for empty clusters
    empty_clusters = [i for i, idx in enumerate(nearest_indices) if idx == -1]
    if empty_clusters:
        logger.warning(f"Found {len(empty_clusters)} empty clusters: {empty_clusters}")

    return nearest_indices, nearest_distances
