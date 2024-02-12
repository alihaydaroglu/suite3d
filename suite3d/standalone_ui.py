import napari
import os
import numpy as n
from pathlib import Path

def load_outputs(base_path = None, verbose=True):
    """ 
    Loads the outputs of cell detection. If base_path is not specified,
    will load from the current working directory. Directory should include
    stats.npy, info.npy and iscell.npy at a minimum.
    """
    if base_path is None:
        base_dir = Path('.')
    else:
        base_dir = Path(base_path)

    # every output dir should have stats.npy, info.npy
    stats_path = base_dir / 'stats.npy'
    info_path = base_dir / 'info.npy'
    assert stats_path.exists(), "Could not find: %s" % stats_path.absolute()
    assert info_path.exists(), "Could not find: %s" % info_path.absolute()

    if verbose: print("Loading stats.npy and info.npy")
    stats = n.load(stats_path, allow_pickle=True).item()
    info = n.load(info_path, allow_pickle=True).item()

    # other  files to look for in the path
    activity_files = ['F', 'Fneu', 'spks'] # these will be memmapped
    curation_files = ['iscell', 'iscell_extracted', 'iscell_curated', 'iscell_curated_slider'] # load entirely
    activity = {}
    curation = {}

    if verbose: print("Loading curation files")
    for file in curation_files:
        path = base_dir / (file + '.npy')
        if path.exists():
            print("    Loading %s" % (file,))
            curation[file] = n.load(path, allow_pickle=True)
        else: 
            if file == 'iscell_extracted':
                curation[file] = None
            else:
                curation[file] = n.copy(curation['iscell'])

    if verbose: print("Looking for activity files")
    for file in activity_files:
        path = base_dir / (file + '.npy')
        if path.exists():
            activity[file] = n.load(path, mmap_mode = 'r')
            print("    Memory-mapped %s" % file)
        else:
            activity[file] = None
            print("    Could not find %s" % file)

    return stats, info, curation, activity

def make_label_vol(stats, shape=None, iscell_1d=None, cmap='Set3'):
    '''
    Make two volumes: one with RGBA colours randomly assigned for each cell, and one with

    Args:
        stats (list): output of suite3d. Each element is a dict with elements 'coords' and 'lam'
        shape (tuple): shape of the volume to fill up
        iscell_1d (ndarray, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to 'Set3'.
    '''
    coords = [stat['coords'] for stat in stats]
    lams = [stat['lam'] for stat in stats]
