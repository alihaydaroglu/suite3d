How the Suite3D GUI Works
=========================

Suite3D includes an interactive desktop viewer for inspecting ``info.npy`` and
ROI outputs, plus a separate portable HTML viewer that can be exported from a
finished job. The desktop GUI is implemented in
``suite3d/viewer/info_viewer_gui.py``. It uses PyQt5 for windows, controls, and
threading, and embeds Matplotlib canvases for image, curation, trace, motion,
and 3D ROI plots.

Starting the GUI
----------------

The Windows launcher at the repository root, ``info_viewer.bat``, activates the
``s3d`` conda environment and runs:

.. code-block:: bat

   python suite3d\viewer\info_viewer_gui.py --no-default %*

The Python entry point also accepts an optional path:

.. code-block:: bash

   python suite3d/viewer/info_viewer_gui.py /path/to/s3d-job
   python suite3d/viewer/info_viewer_gui.py /path/to/s3d-job/rois
   python suite3d/viewer/info_viewer_gui.py /path/to/s3d-job/rois/info.npy

If a directory is provided, the GUI resolves it to either ``rois/info.npy`` or
``info.npy``. If no path is provided and default loading is enabled, it tries the
``DEFAULT_INFO_PATH`` constant in ``info_viewer_gui.py``.

Main Objects
------------

``InfoViewer`` is the main ``QMainWindow``. It owns the top-level state for the
loaded Suite3D run:

* ``info`` and ``current_array`` hold the selected 3D image volume.
* ``stats`` holds ROI dictionaries loaded from ``stats.npy``.
* ``iscell`` holds accepted and rejected ROI state loaded from or saved to
  ``iscell.npy``.
* ``traces`` memory-maps ``F.npy``, ``Fneu.npy``, and ``spks.npy`` when those
  files are present.
* ``recording_files`` memory-map registered movie chunks from
  ``registered_fused_data/fused_reg_data*.npy``.
* ``motion_shifts`` loads motion correction offsets from ``offsets*.npy``.

``Suite3DAnalysisWorker`` is a ``QThread`` used by the Analyze Data dialog. It
runs pipeline stages outside the GUI thread, emits progress updates, forwards
job log messages into the GUI, and loads the resulting ``rois/info.npy`` when
the run completes.

``Roi3DWindow`` is a child dialog for a selected ROI. It renders either the ROI
mask or neuropil mask as points or as a smoothed surface, using voxel size
metadata where available.

Layout and Controls
-------------------

The main window is built in ``InfoViewer._build_ui()``. It contains four working
areas:

* File controls for opening a run folder, opening an ``info.npy`` directly, or
  starting analysis from raw TIFF data.
* Display controls for image source, z-plane projection, ROI mask overlays,
  ROI color mode, right-panel mode, motion correction display, z-plane playback,
  and registered-frame playback.
* An image canvas that shows accepted ROIs on the left and either non-accepted
  ROIs or accepted neuropil on the right.
* A trace canvas that shows ``F``, ``Fneu``, and ``spks`` for the selected ROI,
  or motion correction shifts when motion display is enabled.

The GUI uses Qt signals to connect controls to update methods. For example, the
image selector, projection controls, mask checkbox, right-panel selector, and
sliders all trigger display refreshes through ``on_display_changed()`` or the
plane and frame handlers.

Loading a Suite3D Run
---------------------

Loading starts in ``load_info()``:

1. ``resolve_info_path()`` normalizes the user-selected path.
2. ``np.load(..., allow_pickle=True).item()`` reads the Suite3D info dictionary.
3. The viewer lists displayable 3D arrays from ``max_img``, ``mean_img``,
   ``vmap``, and ``vmap_raw``.
4. ``load_roi_files()`` reads ``stats.npy`` and optional ``iscell.npy``.
5. ``load_trace_file()`` memory-maps traces from ``F.npy``, ``Fneu.npy``, and
   ``spks.npy`` when present.
6. ``load_recording_files()`` discovers registered movie chunks and enables
   frame playback if they are available.

When the image source changes, ``update_array_selection()`` updates slider
ranges, playback availability, status text, projection controls, and the current
image shape. ``update_image()`` then redraws the Matplotlib figure.

Image Display and ROI Overlays
------------------------------

The image view can show a single z-plane, a max or mean projection across
planes, a black background, or a registered movie frame. Contrast is computed
from the 1st and 99.8th percentiles unless the black background is selected.

ROI overlays are generated from the coordinate arrays stored in each ROI's
``stats.npy`` entry:

* ``coords`` supplies cell ROI voxels.
* ``npcoords`` supplies neuropil voxels when available.
* ``iscell[:, 0]`` decides whether each ROI appears in the accepted or
  non-accepted panel.

``build_mask_overlay()`` creates two arrays for each panel: an RGBA overlay and
an integer ROI id map. The id map allows mouse clicks to map image pixels back
to ROI indices. Overlays are cached by image shape, panel type, plane,
projection state, mask kind, and color mode so repeated redraws are faster.

Selection, Navigation, and Zoom
-------------------------------

Mouse and keyboard actions are handled directly on the Matplotlib canvases:

* Left click selects the ROI under the cursor and updates the trace panel.
* Left drag pans all image panels together.
* Mouse wheel zooms around the cursor.
* Right click toggles the ROI under the cursor between accepted and
  non-accepted.
* Right drag draws a rectangle and moves every ROI in that rectangle to the
  other curation panel.
* Left and right arrow keys step through ROIs in the current accepted or
  non-accepted pool.
* ``Ctrl+Z`` restores the previous curation state from the undo stack.

The selected ROI is drawn in white on top of the regular mask overlay.
``Zoom selected ROI`` computes a bounding box around the selected mask and
applies that view to all image panels. ``Reset zoom`` restores the full image.

Curation Workflow
-----------------

The right-side curation panel computes metrics from ``stats.npy`` and trace
files:

* voxel count
* number of z-planes touched by the ROI
* peak detection value
* median voxel SNR
* ``F`` trace skewness

Each visible histogram has draggable low and high threshold lines. Once a
threshold is changed, ``apply_curation_filters()`` recomputes
``iscell[:, 0]`` from all active metric ranges. Manual ROI overrides are kept
separately and are applied through click or rectangle actions. The GUI stores up
to 50 undo snapshots for curation operations.

The ``Reject Fneu > F`` command rejects ROIs whose mean neuropil trace is larger
than the mean ROI fluorescence trace. Those rejected ROIs are tracked separately
so the adjacent ``Undo`` button can restore them without disturbing other
curation edits.

``Save to iscell.npy`` writes the current curation state back to the loaded ROI
directory. It does not rewrite ``stats.npy`` or trace files.

Trace and Motion Views
----------------------

When an ROI is selected, ``update_trace_plot()`` extracts the matching row or
column from each loaded trace file. The trace panel can show:

* ``F`` fluorescence trace
* ``Fneu`` neuropil trace
* ``spks`` deconvolved activity estimate

The trace canvas supports scroll zoom, drag panning, full-view reset, and a
vertical frame cursor tied to the registered movie frame slider.

If registered movie chunks and ``offsets*.npy`` files are available, the Motion
Correction checkbox switches the trace canvas to a motion plot. That plot shows
y shift, x shift, and combined xy motion across frames, with the current frame
highlighted.

Running Analysis from the GUI
-----------------------------

The Analyze Data dialog lets a user choose a TIFF directory, choose an output
directory, set key Suite3D parameter overrides, and launch the pipeline. Before
starting, the GUI tries to infer:

* volume or frame rate from ScanImage TIFF metadata
* y and x voxel size from ScanImage ROI metadata
* planes from ``n_ch_tif`` unless a comma-separated plane list is provided

The worker creates a ``Job`` named ``gui-analysis`` and runs:

1. initialization
2. registration
3. correlation map calculation
4. ROI segmentation
5. neuropil mask computation
6. trace extraction and deconvolution

Progress is coarse-grained by stage. Job log messages are mirrored into the GUI
log panel. On success, the generated ``rois/info.npy`` is loaded automatically.

Portable HTML Viewer
--------------------

The desktop GUI is different from the exportable HTML viewer in
``suite3d/viewer/html_viewer.py``. The HTML viewer is generated from a completed
``Job``:

.. code-block:: python

   viewer_path = job.make_html_viewer()

It writes ``viewer.html`` plus a ``viewer/`` data directory containing
pre-rendered plane PNGs, encoded ROI masks, metadata, optional trace chunks, and
optional movie snippets. The exported directory is designed to work offline by
opening ``viewer.html`` directly in a browser. Curation from the browser can be
downloaded as ``curation.json`` and applied back to a Suite3D job with
``Job.import_curation()``.

File Expectations
-----------------

For the richest GUI experience, a completed job directory should contain:

.. code-block:: text

   s3d-<job-id>/
     rois/
       info.npy
       stats.npy
       iscell.npy
       F.npy
       Fneu.npy
       spks.npy
     registered_fused_data/
       fused_reg_data*.npy
       offsets*.npy

Only ``info.npy`` with at least one displayable 3D array is strictly required
for image viewing. ROI overlays require ``stats.npy``. Curation persistence
requires ``iscell.npy`` or lets the GUI create a default accepted array in
memory before saving. Trace, registered movie, neuropil, and motion features
activate only when their corresponding files are found.
