import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.getcwd())
import h5py


data = np.load(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output\all_sessions_patches.npy", allow_pickle=True)
info = np.load(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output\all_sessions_info.npy", allow_pickle=True).item()

root = r'\\znas.cortexlab.net\Lab\Share\Ali\for-suyash'
shot_noise, edge_cells = [], []

for session in os.listdir(root):
    session_path = os.path.join(root, session)
    if not os.path.exists(os.path.join(session_path, "shot_noise.npy")):
        continue
    shot = np.load(os.path.join(session_path, "shot_noise.npy"))
    edge = np.load(os.path.join(session_path, "edge_cells.npy"))
    shot_noise.append(shot)
    edge_cells.append(edge)


with h5py.File(r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\data\dataset.h5", 'w') as hf:
    hf.create_dataset("data",  data=data)
    hf.create_dataset("shot_noise", data=np.concatenate(shot_noise))
    hf.create_dataset("edge_cells", data=np.concatenate(edge_cells))


