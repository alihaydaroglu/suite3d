import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.getcwd())
from suite3d import quality_metrics
from matplotlib.cm import ScalarMappable


def compute_shot_noise_for_all_sessions():
    raw_root = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash"


    for session in os.listdir(raw_root):

        session_path = os.path.join(raw_root, session)
        if not os.path.isdir(session_path):
            continue

        if not os.path.exists(os.path.join(session_path, "F.npy")):
            continue
        fnpy = np.load(os.path.join(session_path, "F.npy"))
        shot = quality_metrics.shot_noise_pct(fnpy, 4)

        np.save(os.path.join(session_path, "shot_noise.npy"), shot)

        print(f"Session: {session}, Shot Noise Shape: {shot.shape}")


def umap_part():
    umap_embeddings = np.load(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\figures\umap_2d.npy")
    info = np.load(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\data\all_sessions_info.npy", allow_pickle=True).item()
    raw_root = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash"
    all_cells_shot = []


    for i, session in enumerate(info['session_names']):
        session_path = os.path.join(raw_root, session)
        n_cells = info['session_cell_counts'][i]

        if not os.path.exists(os.path.join(session_path, "shot_noise.npy")):
            continue

        shot = np.load(os.path.join(session_path, "shot_noise.npy"))
        if shot.shape[0] != n_cells:
            print(session, n_cells, shot.shape)

        all_cells_shot.append(shot)

    c = np.concatenate(all_cells_shot)
    umap_embeddings = umap_embeddings[c < 1]
    c = c[c < 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=c, cmap='viridis', alpha=0.1, s=0.01)
    fig.colorbar(ScalarMappable(cmap=scatter.get_cmap(), norm=scatter.norm), label="Shot Noise Level", cax=ax.inset_axes([0.95, 0.1, 0.02, 0.8]))
    fig.tight_layout()
    plt.show()


# compute_shot_noise_for_all_sessions()

umap_part()
