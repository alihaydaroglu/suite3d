# This script will run the full curation pipeline. It can be run after running the Suite3D pipeline (as in the demo notebook).
# You will need the relevant job directory. 
import os
import h5py
import numpy as np



def preprocess(jobdir):

    from curation.dataloader import main
    from curation.dimensionality_reduction import PCAfunction

    output_directory = os.path.join(jobdir, 'rois', 'curation')
    os.makedirs(output_directory, exist_ok=True)
    main(data_directory=jobdir, output_directory=output_directory)


    # Step 2: Run the dimensionality reduction on the footprints
    h5_path = os.path.join(output_directory, 'dataset.h5')
    with h5py.File(h5_path, 'r') as f:
        X = f["data"][:, 2, :, :, :]
    UMAPtime = PCAfunction(
        X=X,
        ncomp=8,
        out_dir=output_directory,
        image_shape=(X.shape[1], X.shape[2], X.shape[3])
        )
    
    all_sessions_info = np.load(os.path.join(output_directory, 'all_sessions_info.npy'), allow_pickle=True).item()
    all_sessions_info['UMAPtime'] = UMAPtime
    np.save(os.path.join(output_directory, 'all_sessions_info.npy'), all_sessions_info)


if __name__ == "__main__":

    from dataloader import main
    from dimensionality_reduction import PCAfunction

    # Step 1: Run the dataloader to load all the ROIs
    jobdir = r"C:\Users\suyash\UCL\tinya_data"
    output_directory = os.path.join(jobdir, 'rois', 'curation')
    os.makedirs(output_directory, exist_ok=True)
    main(data_directory=jobdir, output_directory=output_directory)


    # Step 2: Run the dimensionality reduction on the footprints
    h5_path = os.path.join(output_directory, 'dataset.h5')
    with h5py.File(h5_path, 'r') as f:
        X = f["data"][:, 2, :, :, :]
    UMAPtime = PCAfunction(
        X=X,
        ncomp=8,
        out_dir=output_directory,
        image_shape=(X.shape[1], X.shape[2], X.shape[3])
        )
    
    all_sessions_info = np.load(os.path.join(output_directory, 'all_sessions_info.npy'), allow_pickle=True).item()
    all_sessions_info['UMAPtime'] = UMAPtime
    np.save(os.path.join(output_directory, 'all_sessions_info.npy'), all_sessions_info)


