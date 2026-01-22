from __future__ import annotations
import os
from typing import Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import HDBSCAN
from umap import UMAP
import json
import h5py
from sklearn.feature_selection import VarianceThreshold
import time


def _iter_batches_permod(X: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield modality-specific batches: returns x0,x1,x2 each shaped (B,2000)."""
    N = X.shape[0]
    for start in range(0, N, batch_size):
        stop = min(N, start + batch_size)
        xb = X[start:stop].astype(np.float32)  # (B,3,5,20,20)
        
        # Split modalities and flatten last 3 dims
        x0 = xb[:, 0].reshape(xb.shape[0], -1)  # (B,2000)
        x1 = xb[:, 1].reshape(xb.shape[0], -1)
        x2 = xb[:, 2].reshape(xb.shape[0], -1)
        
        # Clean non-finite values
        for arr in (x0, x1, x2):
            arr[~np.isfinite(arr)] = 0.0
            
        yield x0, x1, x2


def run_permod_pca_umap(
    X: np.ndarray,
    ncomp_per_mod: int = 32,
    batch_size: int = 4096,
    out_dir: str = "pca_permod_umap",
    whiten: bool = False,
    seed: int = 0,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    savename: str = "umap_2d",
    variance_threshold = 1e-8
) -> str:
    """
    Per-modality IncrementalPCA, concatenate embeddings, then UMAP to 2D.
    """
    assert X.ndim == 5 and X.shape[1:] == (3, 5, 20, 20), "X must have shape (N,3,5,20,20)"
    os.makedirs(out_dir, exist_ok=True)

    N = X.shape[0]

    # Initialize scalers and PCA per modality
    scalers = [StandardScaler() for _ in range(3)]
    variance_selectors = [None for _ in range(3)]
    ipcas = [IncrementalPCA(n_components=ncomp_per_mod, batch_size=batch_size) for _ in range(3)]

    # Pass 1: fit scalers
    print("Fitting scalers...")
    for x0, x1, x2 in _iter_batches_permod(X, batch_size):
        scalers[0].partial_fit(x0)
        scalers[1].partial_fit(x1)
        scalers[2].partial_fit(x2)
    # Create variance selectors to remove low-variance features
    print("Setting up variance filtering...")
    
    # Use a small sample to determine feature variance
    sample_size = min(1000, N)
    sample_idx = np.random.choice(N, sample_size, replace=False)
    X_sample = X[sample_idx]
    
    for mod_idx in range(3):
        # Get sample data for this modality
        x_sample = X_sample[:, mod_idx].reshape(sample_size, -1)
        x_sample_scaled = scalers[mod_idx].transform(x_sample)
        
        # Create variance threshold selector
        variance_selectors[mod_idx] = VarianceThreshold(threshold=variance_threshold)
        variance_selectors[mod_idx].fit(x_sample_scaled)
        
        # Determine effective number of components
        n_features_remaining = variance_selectors[mod_idx].transform(x_sample_scaled).shape[1]
        effective_ncomp = min(ncomp_per_mod, n_features_remaining - 1, sample_size - 1)
        
        print(f"Modality {mod_idx}: {n_features_remaining} features after variance filtering, "
              f"using {effective_ncomp} PCA components")
        
        # Initialize IncrementalPCA with appropriate number of components
        ipcas[mod_idx] = IncrementalPCA(
            n_components=effective_ncomp, 
            batch_size=min(batch_size, sample_size)
        )

    # Calculate total embedding dimension
    K = sum(ipca.n_components for ipca in ipcas)

    # Pass 2: fit IncrementalPCA per modality
    print("Fitting PCA models...")
    for x0, x1, x2 in _iter_batches_permod(X, batch_size):
        # Scale and filter variance for each modality
        xs_filtered = []
        for mod_idx, x in enumerate([x0, x1, x2]):
            x_scaled = scalers[mod_idx].transform(x)
            x_filtered = variance_selectors[mod_idx].transform(x_scaled)
            xs_filtered.append(x_filtered)
            
            # Partial fit PCA
            ipcas[mod_idx].partial_fit(x_filtered)

    # Pass 3: transform and concatenate
    print("Transforming data...")
    emb_path = os.path.join(out_dir, "embeddings_permod.npy")
    Z = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, K))

    i = 0
    final_scaler = None
    
    for x0, x1, x2 in _iter_batches_permod(X, batch_size):
        # Process each modality
        embeddings = []
        
        for mod_idx, x in enumerate([x0, x1, x2]):
            # Scale and filter variance
            x_scaled = scalers[mod_idx].transform(x)
            x_filtered = variance_selectors[mod_idx].transform(x_scaled)
            
            # PCA transform
            z = ipcas[mod_idx].transform(x_filtered)
            
            # Optional whitening with numerical stability
            if whiten:
                explained_var = ipcas[mod_idx].explained_variance_
                # Add small epsilon to prevent division by zero
                z = z / np.sqrt(np.maximum(explained_var, 1e-12))
            
            embeddings.append(z)
        
        # Concatenate embeddings from all modalities
        Zb = np.concatenate(embeddings, axis=1).astype(np.float32)
        
        # Apply final scaling
        if i == 0:  # First batch - fit scaler
            final_scaler = StandardScaler()
            Zb_scaled = final_scaler.fit_transform(Zb)
        else:
            Zb_scaled = final_scaler.transform(Zb)
        
        # Store batch
        b = Zb_scaled.shape[0]
        Z[i:i+b] = Zb_scaled
        i += b

    #save PCA embeddings
    np.save(os.path.join(out_dir, "pca_embeddings.npy"), Z)

    # UMAP on concatenated embeddings
    print("Running UMAP...")
    umap_model = UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=seed,
    )

    # Convert memmap to array for UMAP
    Z_array = np.array(Z)
    Y = umap_model.fit_transform(Z_array)
    
    # Save results
    np.save(os.path.join(out_dir, f"{savename}.npy"), Y.astype(np.float32))
    save_scatter(os.path.join(out_dir, f"{savename}.png"), Y)

    print(f"Pipeline complete. Results saved to: {out_dir}")
    return out_dir


def PCAfunction(
    X: np.ndarray,
    ncomp: int = 32,
    out_dir: str = "pca_permod_umap",
    whiten: bool = False,
    seed: int = 0,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    savename: str = "umap_2d",
    visualize_pcs: bool = True,
    n_pcs_to_visualize: int = 16,
    image_shape: tuple = None,
    channel_names: list = None,  # e.g., ['slice1', 'slice2', ...] or ['R', 'G', 'B', ...]
) -> str:
    """
    PCA reduction followed by UMAP to 2D.
    """    
    os.makedirs(out_dir, exist_ok=True)

    N = X.shape[0]
    X = X.reshape(N, -1).astype(np.float32)
    
    # Fit PCA on all data at once
    print(f"Fitting PCA with {ncomp} components...")
    pca = PCA(n_components=ncomp, whiten=whiten, random_state=seed)
    Z = pca.fit_transform(X).astype(np.float32)
    
    print(f"PCA reduced data to {pca.n_components_} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Save PCA embeddings
    emb_path = os.path.join(out_dir, "pca_embeddings.npy")
    np.save(emb_path, Z)
    
    # Visualize principal components
    if visualize_pcs:
        print("Visualizing principal components...")
        n_viz = min(n_pcs_to_visualize, pca.n_components_)
        
        # Save raw PC components
        pc_components = pca.components_[:n_viz]
        np.save(os.path.join(out_dir, "pc_components.npy"), pc_components)
        
        if image_shape is not None:
            if len(image_shape) == 3 and image_shape[0] <= 10:
                _visualize_pcs_multichannel(pc_components, image_shape, pca.explained_variance_ratio_, 
                                           out_dir, n_viz, channel_names)
        
        # Always save explained variance plot
        _plot_explained_variance(pca, out_dir)
        
        print(f"PC visualizations saved to: {out_dir}")
    
    # Run UMAP
    print("Running UMAP...")
    start_time = time.time()
    umap_model = UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=seed,
    )
    Y = umap_model.fit_transform(Z)
    end_time = time.time()
    UMAPtime = np.round(end_time - start_time, 0)
    
    # Save UMAP results
    np.save(os.path.join(out_dir, f"{savename}.npy"), Y.astype(np.float32))
    
    print(f"Pipeline complete. Results saved to: {out_dir}")
    return UMAPtime


def _visualize_pcs_multichannel(pc_components, image_shape, var_ratios, out_dir, n_viz, channel_names=None):
    """
    Visualize PCs for multi-channel/multi-slice data (e.g., shape (5, 20, 20)).
    Shows each channel/slice as a separate 2D heatmap.
    """    
    n_channels, height, width = image_shape
    
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    # Create a large figure showing all PCs and channels
    fig = plt.figure(figsize=(n_channels * 3, n_viz * 2.5))
    
    for pc_idx in range(n_viz):
        pc = pc_components[pc_idx].reshape(image_shape)
        
        # Normalize across all channels for consistent color scale
        vmin, vmax = pc.min(), pc.max()
        
        for ch_idx in range(n_channels):
            ax = plt.subplot(n_viz, n_channels, pc_idx * n_channels + ch_idx + 1)
            
            im = ax.imshow(pc[ch_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
            
            # Add title only on top row
            if pc_idx == 0:
                ax.set_title(channel_names[ch_idx], fontsize=10)
            
            # Add PC label only on left column
            if ch_idx == 0:
                ax.set_ylabel(f'PC{pc_idx+1}\n({var_ratios[pc_idx]:.1%})', fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pc_channels_grid.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create individual PC visualizations with colorbars
    pc_dir = os.path.join(out_dir, "individual_pcs")
    os.makedirs(pc_dir, exist_ok=True)
    
    for pc_idx in range(min(n_viz, 8)):  # Save detailed view for first 8 PCs
        pc = pc_components[pc_idx].reshape(image_shape)
        
        fig, axes = plt.subplots(1, n_channels, figsize=(n_channels * 3, 3))
        if n_channels == 1:
            axes = [axes]
        
        vmin, vmax = pc.min(), pc.max()
        
        for ch_idx in range(n_channels):
            ax = axes[ch_idx]
            im = ax.imshow(pc[ch_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'{channel_names[ch_idx]}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(f'PC{pc_idx+1} ({var_ratios[pc_idx]:.2%} variance explained)', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(pc_dir, f"pc{pc_idx+1}_channels.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def _plot_explained_variance(pca, out_dir):
    """Plot explained variance."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Explained by Each PC')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.grid(alpha=0.3)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "explained_variance.png"), dpi=150, bbox_inches='tight')
    plt.close()


def save_scatter(p: str, Y: np.ndarray, labels=None):
    """Save a simple scatter plot of UMAP results."""
    plt.figure(figsize=(7, 6), dpi=120)
    
    # Default case: no coloring
    colours = None
    show_colorbar = False
    
    # to colour by session
    # if os.path.exists(os.path.join(os.path.dirname(p), "all_sessions_info.npy")):
    #     info = np.load(os.path.join(os.path.dirname(p), "all_sessions_info.npy"), allow_pickle=True).item()
    #     cell_counts = info.get('session_cell_counts', [])
    #     names = info.get('session_names', [])
    #     if cell_counts and names:
    #         colours = np.concatenate([np.ones(c) * i for i, c in enumerate(cell_counts)])

    # Color by cluster labels
    if os.path.exists(os.path.join(os.path.dirname(p), "umap_cluster_labels.npy")):
        cluster_labels = np.load(os.path.join(os.path.dirname(p), "umap_cluster_labels_within7.npy"))
        # if clustering within cluster 7
        # Y = Y[cluster_labels == 7]
        # cluster_labels = np.load(os.path.join(os.path.dirname(p), "umap_cluster_labels_within7.npy"))
        colours = cluster_labels
        show_colorbar = True

    scatter = plt.scatter(Y[:, 0], Y[:, 1], s=0.1, alpha=0.1, c=labels, cmap='viridis')
    # plt.legend(*scatter.legend_elements(num=9), title="Session", loc="best", markerscale=1, fontsize='small')

    # Add colorbar for cluster labels
    if show_colorbar or labels is not None:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise points (-1)
        plt.colorbar(scatter, label=f'Cluster ID ({n_clusters} clusters)', shrink=0.8, alpha=1)
    
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP 2D Embedding")
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def clustering(OUT_DIR):

    Y = np.load(os.path.join(OUT_DIR, "umap_2d_spatfilt.npy"))
    print(f"UMAP shape: {Y.shape}")
    # original_labels = np.load(os.path.join(OUT_DIR, "umap_cluster_labels.npy"))
    # Y = Y[original_labels == 7]
    # Perform clustering on Y
    clusterer = HDBSCAN(min_cluster_size=500)
    cluster_labels = clusterer.fit_predict(Y)
    # Save cluster labels
    np.save(os.path.join(OUT_DIR, "umap_cluster_labels_spatfilt.npy"), cluster_labels)


def cluster_representatives(umap_2d, labels, full_features, save_path=None, save_format='json'):
    """
    For each cluster, compute:
    - mean: mean of full_features for cluster
    - median: real datapoint closest to cluster centroid in UMAP space
    Returns a dict with keys 'mean' and 'median', each a list of arrays (len=n_clusters).
    Optionally saves the result to file (json or npy).
    """
    import numpy as np
    unique_labels = np.unique(labels)
    means = []
    medians = []
    for cl in unique_labels:
        idx = np.where(labels == cl)[0]
        umap_cluster = umap_2d[idx]
        features_cluster = full_features[idx]
        # Mean: mean of full_features in cluster
        mean_feat = features_cluster.mean(axis=0)
        means.append(mean_feat.tolist())
        # Median: find point closest to centroid in UMAP space
        centroid = umap_cluster.mean(axis=0)
        dists = np.linalg.norm(umap_cluster - centroid, axis=1)
        median_idx = idx[np.argmin(dists)]
        median_feat = full_features[median_idx]
        medians.append(median_feat.tolist())
    result = {'mean': means, 'median': medians}
    if save_path:
        if save_format == 'json':
            with open(save_path, 'w') as f:
                json.dump(result, f)
        elif save_format == 'npy':
            np.save(save_path, result)
    return result



if __name__ == "__main__":


    H5_PATH = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\data\dataset.h5"
    OUT_DIR = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output"

    with h5py.File(H5_PATH, 'r') as f:
        X = f["data"][:, 2, :, :, :]
    
    # run_permod_pca_umap(
    #     X=X,
    #     ncomp_per_mod=32,
    #     batch_size=4096,
    #     out_dir=OUT_DIR,
    #     whiten=True,
    #     seed=None,
    #     umap_neighbors=30,
    #     umap_min_dist=0.1,
    #     savename="umap_2d",
    # )

    UMAPtime = PCAfunction(
        X=X,
        ncomp=16,
        out_dir=OUT_DIR,
        image_shape=(X.shape[1], X.shape[2], X.shape[3])
        )

