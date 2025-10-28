import numpy as np
import h5py
import os
from curation.dimensionality_reduction import run_permod_pca_umap, save_scatter
from sklearn.preprocessing import StandardScaler


data_file = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\data\dataset.h5"
OUT_DIR = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output"

with h5py.File(data_file, 'r') as f:
    edge_cells = f['edge_cells'][:]
#     spatfilt = f['data'][edge_cells==False, :, :, :, :]

# print(spatfilt.shape)


# run_permod_pca_umap(
#     X=spatfilt,
#     ncomp_per_mod=32,
#     batch_size=4096,
#     out_dir=OUT_DIR,
#     whiten=True,
#     seed=None,
#     umap_neighbors=30,
#     umap_min_dist=0.1,
#     savename="umap_2d_spatfilt",
# )


info = np.load(os.path.join(OUT_DIR, "all_sessions_info.npy"), allow_pickle=True).item()
cell_counts = info.get('session_cell_counts', [])
names = info.get('session_names', [])
if cell_counts and names:
    colours = np.concatenate([np.ones(c) * i for i, c in enumerate(cell_counts)])

colours = colours[edge_cells==False]

save_scatter(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output\umap_2d_test.png", np.load(os.path.join(OUT_DIR, "umap_2d_spatfilt.npy")).astype(np.float32), labels=colours)

