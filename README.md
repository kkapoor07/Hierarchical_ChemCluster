# Hierarchical Cheminformatics Clustering Pipeline

## Overview

This repository provides a Python-based pipeline for performing K-Means clustering on extremely large chemical datasets (demonstrated on ~0.5 billion compounds) using a hierarchical approach with FAISS (CPU). It is particularly designed to handle situations where the desired number of final clusters (*k*) is very large, making a single K-Means run computationally infeasible due to algorithmic scaling, even if the dataset fits in memory.

The core strategy involves:
1.  **Parallel Preprocessing:** Canonicalizes tautomers and removes duplicates from input SMILES files using RDKit and multiprocessing.
2.  **Parallel Fingerprint Generation:** Efficiently calculates chemical fingerprints (e.g., Morgan) across multiple CPU cores using RDKit and multiprocessing.
3.  **Hierarchical K-Means Clustering (CPU):**
    *   **Stage 1:** Clusters the entire dataset into a manageable number of initial clusters (`k1`) using `faiss-cpu`. **This stage requires a high-memory machine** as it loads the full fingerprint dataset. It outputs data for each initial cluster into separate files.
    *   **Stage 2:** Processes each smaller cluster file independently (using multiprocessing across files), performing another `faiss-cpu` K-Means run to generate sub-clusters (`k2i`). This distributes the computational load.
4.  **Aggregation & Conversion:** Gathers the representative molecules (nearest data points to centroids) from all Stage 2 sub-clusters and provides utilities to convert outputs to standard formats (SMI, CSV).

This pipeline enables the generation of large, diverse compound sets (representatives) from massive chemical libraries for applications like virtual screening library preparation.

## Key Features

*   **Handles Massive Datasets:** Successfully applied to ~0.5 billion compounds.
*   **Addresses Large *k* Scaling:** Uses a two-stage hierarchical approach to make clustering into millions of final clusters computationally feasible compared to single-stage K-Means.
*   **Parallel Processing:** Leverages multiprocessing for efficient preprocessing (tautomer handling), fingerprint generation, and Stage 2 clustering across multiple CPU cores.
*   **FAISS (CPU) Integration:** Utilizes the efficient `faiss-cpu` library for the core K-Means steps.
*   **Modular Pipeline:** Organized into distinct steps (preprocessing, fingerprinting, clustering stages, aggregation, conversion) executed via dedicated scripts with underlying logic in `src/`.
*   **Configurable:** Key parameters like cluster numbers, file paths, fingerprint settings, and parallelism are configurable via command-line arguments.

## Requirements

*   **Python:** >= 3.9
*   **Conda:** Recommended for managing dependencies, especially `faiss-cpu` and `rdkit`.
*   **Core Libraries:** `faiss-cpu`, `rdkit`, `h5py`, `numpy`, `pandas`, `tqdm` (See `environment.yml`).
*   **Hardware:**
    *   **High Memory Node REQUIRED** for Clustering Stage 1. Memory needs are proportional to the size of the full fingerprint dataset (e.g., >200GB RAM might be needed for 0.5 billion 256-bit fingerprints, plus overhead).
    *   Multi-core CPU beneficial for preprocessing, fingerprint generation, and Stage 2 clustering.

## Installation

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/kkapoor07/Hierarchical_Cheminformatics_Clustering.git
    cd Hierarchical_Cheminformatics_Clustering
    ```

2.  **Create Conda Environment:**
    ```bash
    # Create the environment from the file
    conda env create -f environment.yml

    # Activate the environment
    conda activate hier_chem_cluster
    ```

## Pipeline Workflow & Usage

The pipeline consists of several scripts found in the `scripts/` directory, typically run in sequence. It's recommended to create separate directories for logs and various output stages.

```bash
# Example directory setup in your working area
mkdir logs
mkdir work_output
mkdir work_output/preprocessed_smiles
mkdir work_output/fingerprints
mkdir work_output/clustering_results
mkdir data
# Assume raw input SMILES files are placed inside 'data/raw_smiles/' (possibly in subdirs)
```

**Step 1: Preprocessing (Tautomer Canonicalization & Deduplication)**

*   **Script:** `scripts/preprocess_data.py`
*   **Input:** Directory containing raw SMILES files (e.g., `data/raw_smiles/`).
*   **Output:** Directory (`work_output/preprocessed_smiles/`) containing `*_taut.smi` files with unique canonical tautomers.

```bash
python scripts/preprocess_data.py \
    data/raw_smiles \
    work_output/preprocessed_smiles \
    --log_file logs/preprocess.log \
    --n_workers -1 \
    --delimiter " " \
    --smiles_col smiles \
    --id_col zinc_id \
    --pattern "*.smi" # Adjust pattern if needed
    # Add --verbose for more detailed logs
```

**Step 2: Fingerprint Generation (Parallel)**

*   **Script:** `scripts/generate_fingerprints.py`
*   **Input:** Directory containing preprocessed `*_taut.smi` files (`work_output/preprocessed_smiles/`).
*   **Output:** Directory (`work_output/fingerprints/`) containing individual `.h5` fingerprint files. Crucially, use `--aggregate` to create the single large HDF5 file needed for clustering.

```bash
python scripts/generate_fingerprints.py \
    work_output/preprocessed_smiles \
    work_output/fingerprints \
    --pattern "*_taut.smi" \
    --nbits 256 --radius 2 --fp_dtype int8 \
    --aggregate \
    --aggregated_output_file "fingerprints_aggregated.h5" \
    --log_file logs/gen_fp.log \
    --n_workers -1 \
    --skip_header # Add if your *_taut.smi files have headers
    # Add --verbose for more detailed logs
```
*Result: `work_output/fingerprints/fingerprints_aggregated.h5` is created containing all fingerprints.*

**Step 3: Hierarchical Clustering**

*   **Script:** `scripts/run_hierarchical_clustering.py`
*   **Input:** The single, large, aggregated fingerprint HDF5 file (`work_output/fingerprints/fingerprints_aggregated.h5`).
*   **Output:** Intermediate `cluster_*.h5` files (Stage 1), final `details_cluster_*.h5` and `centers_cluster_*.h5` files (Stage 2), and a final aggregated representatives file (`representatives_final.h5`) - all within the specified working directory (`work_output/clustering_results/`).
*   **Requires High Memory Node for Stage 1!** Ensure this script is run on appropriate hardware.

```bash
# Example: Aim for ~1M final representatives from 0.5B molecules
# Stage 1: Split into 1000 initial clusters
# Stage 2: Aim for ~1000 reps per initial cluster (adjust k2_value/base_k)
python scripts/run_hierarchical_clustering.py \
    --aggregated_fp_input_file work_output/fingerprints/fingerprints_aggregated.h5 \
    --work_dir work_output/clustering_results \
    --k1 1000 \
    --k2_method ratio --k2_value 500.0 --k2_base_k 1000 \
    --log_file logs/clustering_pipeline.log \
    --n_workers_stage2 -1 \
    --seed 42 # For reproducibility
    # Add --verbose for more detailed logs
```
*(Adjust `k1`, `k2_*` parameters based on your dataset size and desired final number of representatives. `k1` determines the size/number of intermediate files and RAM needed for Stage 2 jobs. The total number of final representatives will be roughly sum(k2i) across all initial clusters (i.e., summing the sub-clusters found within each initial cluster).)*

*Result: `work_output/clustering_results/representatives_final.h5` contains the aggregated representatives.*

**Step 4: Convert Output (Optional)**

*   **Script:** `scripts/convert_output.py`
*   **Input:** Aggregated representatives HDF5 file (`representatives_final.h5`) OR directory containing Stage 2 detail files (`work_output/clustering_results/stage2_results/`).
*   **Output:** SMILES file (`.smi`) for representatives OR CSV files for cluster details.

```bash
# Example 1: Convert final representatives to SMILES format
python scripts/convert_output.py \
    work_output/clustering_results/representatives_final.h5 \
    work_output/clustering_results/final_representatives.smi \
    --mode centers_to_smi \
    --log_file logs/convert_reps.log

# Example 2: Convert all Stage 2 detail files to CSVs
python scripts/convert_output.py \
    work_output/clustering_results/stage2_results \
    work_output/clustering_results/stage2_details_csv \
    --mode details_to_csv \
    --details_pattern "details_cluster_*.h5" \
    --log_file logs/convert_details.log
```

## File Formats

*   **Input SMILES:** Text files with columns separated by `--delimiter` (default space). Expected columns defined by `--smiles_col` / `--id_col` (preprocessing) or `--smiles_col_idx` / `--name_col_idx` (fingerprinting). Header handling via `--skip_header`.
*   **Intermediate/Output HDF5:** Uses `h5py`. Compression (default gzip) is used.
    *   Individual/Aggregated Fingerprints: Datasets `fp_list` (N x Bits, `int8`), `smiles_list` (N x 1, variable-length bytes), `name_list` (N x 1, variable-length bytes).
    *   Stage 1 Clusters (`cluster_*.h5`): Same format as aggregated fingerprints, containing only members of that cluster.
    *   Stage 2 Details (`details_cluster_*.h5`): Datasets `SMILES` (string), `NAME` (string), `CLUSTER_k2` (int32, sub-cluster index), `DISTANCE_k2` (float32, distance to sub-cluster centroid).
    *   Stage 2 Centers (`centers_cluster_*.h5`): Same format as aggregated fingerprints (`fp_list`, `smiles_list` [bytes], `name_list` [bytes]), containing only the representative points for the sub-clusters.
    *   Final Representatives (`representatives_final.h5`): Same format as Stage 2 Centers, containing all representatives aggregated from Stage 2.
*   **Output SMI/CSV:** Standard text formats. SMI file is space-delimited by default. CSV includes headers.

## Utility Scripts

The `scripts/utils/` directory contains potentially helpful helper scripts derived from the development process:
*   `split_smiles_file.py`: Splitting large SMILES text files.
*   `compress_h5_file.py`: Post-compressing HDF5 files (though pipeline scripts apply compression).
*   `check_h5_contents.py`: Inspecting HDF5 file structure and data.
*   `find_missing_downloads.sh`: Context-specific script for verifying ZINC download structure.
*   `delete_intermediate_files.sh`: Context-specific script for cleaning up intermediate files (use with caution).

Run these scripts with `-h` for usage details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This work was developed starting from code provided by Pat Walters (www.github.com/PatWalters/faiss_kmeans).
Uses RDKit (www.rdkit.org), FAISS (www.github.com/facebookresearch/faiss), H5py, NumPy, Pandas, Tqdm. Please cite these libraries appropriately.
