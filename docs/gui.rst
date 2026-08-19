How the Suite3D GUI Works
=========================

Suite3D includes an interactive desktop viewer for inspecting ``info.npy`` and
ROI outputs, plus a separate portable HTML viewer that can be exported from a
finished job. The desktop GUI is implemented in
``suite3d/viewer/info_viewer_gui.py``.

Running Suite3D Without the GUI
-------------------------------

To analyze data from the command line instead of opening the GUI, use
``suite3d-pipeline``.

The command asks for a few paths and names:

* ``job-dir`` is the base output folder. This is where Suite3D stores analysis
  runs. You can choose any folder where you want the results to be saved.
* ``job-id`` is a short name for one analysis run. Suite3D adds ``s3d-`` before
  this name to create the final run folder.
* ``tif-dir`` is the folder that contains the raw ScanImage TIFF files that you
  want to analyze.
* ``--all`` tells Suite3D to run the main analysis stages instead of only
  creating or loading a job.

For example, if ``job-dir`` is ``C:\example\suite3d-runs`` and ``job-id``
is ``test1``, Suite3D creates this output folder:

.. code-block:: console

   C:\example\suite3d-runs\s3d-test1

1. Activate the Suite3D conda environment:

   .. code-block:: console

      conda activate s3d

2. Start the command-line pipeline:

   .. code-block:: console

      suite3d-pipeline

3. When prompted for the base job directory, enter the ``job-dir`` folder where
   Suite3D should save the output. For example:

   .. code-block:: console

      C:\example\suite3d-runs

4. When prompted for the job ID, enter the ``job-id`` name for this analysis.
   For example:

   .. code-block:: console

      test1

   Suite3D will save the run in:

   .. code-block:: console

      C:\example\suite3d-runs\s3d-test1

5. When prompted for the raw ScanImage TIFF folder, enter the ``tif-dir``
   directory that contains the TIFF files. For example:

   .. code-block:: console

      C:\example\raw-tiffs\session01

6. Choose which stages to run. To run the full non-GUI pipeline directly without
   prompts, pass all values on the command line:

   .. code-block:: console

      suite3d-pipeline --job-dir C:\example\suite3d-runs --job-id test1 --tif-dir C:\example\raw-tiffs\session01 --all

   The ``--all`` option runs initialization, registration, correlation-map
   calculation, and ROI segmentation.

Starting the GUI
----------------

After installing Suite3D into a conda environment, activate that environment and
run the ``suite3d`` command:

.. code-block:: console

   conda activate s3d
   suite3d

Suite3D Analyze
---------------

Running Analysis from the GUI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

To Be Filled
^^^^^^^^^^^^


Suite3D Data Viewer
-------------------

Input
~~~~~

The data viewer opens a completed Suite3D output. The user can select a
Suite3D job directory, its ``rois`` folder, or the ``info.npy`` file directly.
When a directory is selected, the GUI looks for ``rois/info.npy``.

Input for image viewing is:

* ``info.npy``: provides the background images shown in the viewer. The file
  should include at least one 3D image volume, such as ``max_img``,
  ``mean_img``, ``vmap``, or ``vmap_raw``. Any available volumes appear in the
  image selector.
* ``stats.npy``: enables ROI mask overlays and ROI selection. It stores the ROI
  coordinate dictionaries, including ``coords`` for cell masks and optionally
  ``npcoords`` for neuropil masks.
* ``iscell.npy``: provides the accepted/non-accepted ROI labels used by the curation panels. If it is missing, the GUI can still display ROI
  masks from ``stats.npy`` and creates an in-memory accepted array before
  saving.
* ``F.npy``: enables the ROI fluorescence trace.
* ``Fneu.npy``: enables the neuropil trace and the ``Reject Fneu > F`` curation
  action.
* ``spks.npy``: enables the deconvolved activity trace.
* ``registered_fused_data/fused_reg_data*.npy``: enables registered movie frame
  playback.
* ``registered_fused_data/offsets*.npy``: enables the motion correction plot.


Layout and Controls
~~~~~~~~~~~~~~~~~~~

The main window is built in ``InfoViewer._build_ui()``. It contains five working
areas:

* File controls for opening a run folder, opening an ``info.npy`` directly, or
  starting analysis from raw TIFF data.
* Display controls for image source, z-plane projection, ROI mask overlays,
  ROI color mode, right-panel mode, motion correction display, z-plane playback,
  and registered-frame playback.
* An image canvas that shows accepted ROIs on the left and either non-accepted
  ROIs or accepted neuropil on the right.
* An ROI curation panel with metric histograms, draggable thresholds, manual
  curation actions, undo, and saving to ``iscell.npy``.
* A trace canvas that shows ``F``, ``Fneu``, and ``spks`` for the selected ROI,
  or motion correction shifts when motion display is enabled.


Display Panel
~~~~~~~~~~~~~

The ``Display`` panel controls what image is shown, which ROI overlays are
visible, and how the viewer navigates through planes and movie frames.

* ``Image``: chooses the background image. The available entries come from
  ``info.npy`` and can include ``max_img``, ``mean_img``, ``vmap``, and
  ``vmap_raw``. ``Black`` shows ROI masks on a plain black background.
  ``registered movie`` appears when registered movie chunks are available.
* ``Project across planes``: shows a projection across all z-planes instead of
  a single plane. When this is enabled, the plane slider is disabled because the
  image no longer represents one plane.
* ``ROI colors``: chooses how ROI masks are colored. ``Random`` assigns each ROI
  a stable random color. Other modes color ROIs by computed values, such as
  correlation, skewness, voxel count, peak value, or voxel SNR, when those
  values are available.
* ``Show cell ROIs``: toggles the ROI mask overlay on top of the background
  image.
* ``Show right panel``: shows or hides the second image panel. When it is hidden,
  only the accepted-cell panel is displayed.
* ``Right panel``: chooses what the second panel displays. ``Non-accepted
  cells`` shows rejected/non-cell ROIs. ``Accepted neuropil`` shows neuropil
  masks for accepted ROIs when ``npcoords`` are present in ``stats.npy``.
* ``Motion correction``: switches the lower trace plot to motion correction
  shifts when ``offsets*.npy`` files are available.
* ``Plane`` slider and ``Play``: select or animate the z-plane shown in the
  image panels.
* ``Frame`` slider and ``Play``: select or animate registered movie frames when
  registered movie chunks are available.
* ``Zoom selected ROI``: zooms all image panels to the selected ROI mask.
* ``Reset zoom``: restores the full image view.
* ``Show selected ROI in 3D``: opens a separate 3D view for the selected ROI or
  neuropil mask.

Image Display and ROI Overlays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The image view can show a single z-plane, a max or mean projection across
planes, a black background, or a registered movie frame. Contrast is computed
from the 1st and 99.8th percentiles unless the black background is selected.

ROI overlays are generated from the coordinate arrays stored in each ROI's
``stats.npy`` entry:

* ``coords`` supplies cell ROI voxels.
* ``npcoords`` supplies neuropil voxels when available.
* ``iscell[:, 0]`` decides whether each ROI appears in the accepted or
  non-accepted panel.

Selection, Navigation, and Zoom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mouse and keyboard actions are handled directly on the Matplotlib canvases:

* Mouse wheel zooms around the cursor.
* Right click toggles the ROI under the cursor between accepted and
  non-accepted.
* Right drag draws a rectangle and moves every ROI in that rectangle between the
  accepted and non-accepted curation panels.
* Left and right arrow keys step through ROIs in the current accepted or
  non-accepted pool.
* ``Ctrl+Z`` restores the previous curation state from the undo stack.

The selected ROI is drawn in white on top of the regular mask overlay.


Curation Workflow
~~~~~~~~~~~~~~~~~

The right-side curation panel computes metrics from ``stats.npy`` and trace
files:

* voxel count: ROI footprint size in voxels.
* number of z-planes touched by the ROI: how many imaging planes contain ROI
  voxels.
* peak detection value: strength of the seed peak in the Suite3D detection map.
* median voxel SNR: median per-voxel signal-to-noise ratio inside the ROI,
  where higher values mean the ROI-related signal stands out more clearly above
  voxel-level noise or residual background variation.
* ``F`` trace skewness: skewness of the ROI fluorescence trace computed by the
  viewer.

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
~~~~~~~~~~~~~~~~~~~~~~

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

Output
~~~~~~

The desktop GUI can produce or update a small number of files:

* Running analysis from the GUI writes a normal Suite3D job into the selected
  output directory. On success, the GUI automatically loads the generated
  ``rois/info.npy`` from that job.
* ``Save to iscell.npy`` writes the current accepted or non-accepted ROI labels
  to ``rois/iscell.npy``. This is the main persistent output of manual or
  histogram-based curation in the desktop viewer.
* If ``iscell.npy`` was missing when the run was opened, the GUI creates the
  accepted-array state in memory first and writes it only when the user saves.
* Actions such as ROI selection, zooming, trace inspection, motion viewing,
  ROI color changes, and 3D ROI display update only the current session unless
  they are followed by saving ``iscell.npy``.
