import napari
import os
import numpy as n
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import sys
import copy
import functools
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsProxyWidget, QSlider, QPushButton, QVBoxLayout, QLabel, QLineEdit, QShortcut, QCheckBox, QComboBox
from PyQt5.QtGui import QKeySequence
from warnings import warn


default_display_params = {
    'lam_max' : 0.3, # Voxels with cell values above lam_max will have alpha=1
    'cmap' : 'Set3', # colormap to use
    'scale' : (15,3,3), # size of a voxel in z,y,x in microns
    'contrast_percentiles' : (20,99.9), # the contrast limits (expressed as percentile) on startup
    'histogram_nbins' : 100,
    'F_color' : 'red',
    'Fneu_color' : 'blue',
    'spks_color' : 'white',
    'line_width' : 2,
    'n_frames_plotted' : 3000,
}


dropdown_style="""
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}
"""

basicButtonStyle = """
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}

QPushButton:hover {
    background-color: #45a049;
    font-size: 10px;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 5px 5px;
}
"""

qCheckedStyle = """
QWidget {
    background-color: #1F1F1F;
    color: green;
    font-family: Arial, sans-serif;
}
"""

qNotCheckedStyle = """
QWidget {
    background-color: #1F1F1F;
    color: #F0F0F0;
    font-family: Arial, sans-serif;
}
"""


warnings = {
    'missing_iscell_extracted' : 'No iscell_extracted.npy found. Assuming all ROIs were extracted\n',
    'activity_mismatch' : 'Mismatch between number of extracted ROIs and the number of 1s in iscell_extracted.npy[:,0]. This is bad - you have no way of corresponding cells and traces. Re-run extraction. Will not display traces. \n'
}

class GenericNapariUI:
    '''
    Generic class to create a napari-based UI from output files in a directory
    '''
    def __init__(self, base_path=None, verbose=True, display_params = {}):
        '''
        Create and initialize a generic UI class without launching the UI yet

        Args:
            base_path (_type_, optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.
            display_params (dict, optional): _description_. Defaults to {}.
        '''

        if base_path is None:
            self.base_dir = Path('.')
        else:
            self.base_dir = Path(base_path)

        self.verbose = verbose

        self.display_params = copy.deepcopy(default_display_params)
        for key in display_params.keys():
            if key in self.display_params.keys():
                self.log("Display params: Updating %s" % key, 1)
                self.display_params[key] = display_params[key]
            else:
                self.log("Display params: Invalid key %s!" % display_params, 1)

        self.layers = {}
        self.viewer = None
        self.display_roi_labels = None
        self.roi_features = {}
        self.roi_feature_names = {}
        self.roi_feature_ranges = {}
        # these will contain the plot areas, and graphs themselves
        self.hist_plots = {}
        self.hist_graphs = {}
        self.hist_range_lines = {}

        # 
        self.display_activity = False
        self.stats    = []

        self.coords   = []
        self.meds = []
        self.lams     = []
        self.n_roi    = 0 

        self.shape    = (0,0,0)

        self.label_vols = {}

        self.viewer = None

        self.toggle_buttons = {}
        self.toggle_buttons_proxy = {}
        self.filter_toggles = {}

    def start_viewer(self):
        '''
        Start a napari viewer instance
        '''
        if self.viewer is not None:
            try: self.viewer.close()
            except: print("Couldn't close old viewer")
            self.viewer = None
        self.viewer = napari.Viewer(title="Suite3D: %s" % self.base_dir.absolute())

    def close(self):
        '''
        close the viewer
        ## TODO add a warning if things aren't saved
        '''
        self.viewer.close()

    def log(self, text, level=0):
        if self.verbose:
            print(level * '    ', text)

    def load_file(self, filename,allow_pickle=True, mmap_mode=None):
        '''
        Light wrapper around n.load() to load an arbitrary .npy file from self.base_dir
        '''
        filepath = self.base_dir / filename
        if not filepath.exists():
            print("Did not find %s" % filepath)
            return None
        self.log("Loading from %s" % filepath)
        file = n.load(filepath, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        if file.dtype == 'O' and file.ndim < 1: file = file.item()
        return file
        
    def save_file(self, filename,data, overwrite=True):
        '''
        Light wrapper around n.save() to save an arbitrary data to a .npy file 
        to self.basedir, with overwrite protection

        Args:
            filename (str): name + extension of file
        '''

        filepath = self.base_dir / filename
        if filepath.exists():
            if not overwrite:
                print("File %s already exists. Not overwriting." % filepath)
                return
            print("Overwriting existing %s" % filepath)
        n.save(filepath, data)

        
    def make_all_label_vols(self):
        '''
        Makes four volumes of size self.shape. Two of them are RGB volumes with each cell colored differently, one for ROIs labeled cells and one for non-cells. Two are integers, with -1 in all voxels that aren't part of an ROI, and cell_idx in all voxels that are part of an ROI, separated by cells and non-cells.
        '''
        lam_max = self.display_params['lam_max']
        cmap = self.display_params['cmap']
        if self.display_roi_labels is None:
            self.display_roi_labels = n.ones(self.n_roi)
        cell_idxs, cell_rgb = make_label_vols(self.coords, self.lams, self.shape, lam_max = lam_max, 
                        iscell_1d = self.display_roi_labels, cmap=cmap)
        non_cell_idxs, non_cell_rgb = make_label_vols(self.coords, self.lams, self.shape, lam_max = lam_max, iscell_1d = 1 - self.display_roi_labels, cmap=cmap)
        
        self.label_vols = {
        'cell_idxs' : cell_idxs,
        'cell_rgb': cell_rgb,
        'non_cell_idxs' : non_cell_idxs,
        'non_cell_rgb' : non_cell_rgb
        }
    def add_cells_to_viewer(self):
        '''
        Once self.make_all_label_vols() is called, use this to add the cells to the UI
        '''
        scale = self.display_params['scale']
        self.layers['non_cell_rgb'] = self.viewer.add_image(self.label_vols['non_cell_rgb'], name='Non-cells',rgb=True, scale=scale, visible=False)
        self.layers['cell_rgb'] = self.viewer.add_image(self.label_vols['cell_rgb'], name='Cells', rgb=True, scale=scale)

        
    def update_cells_in_viewer(self):
        '''
        Once self.label_vols is updated, use this to update the data displayed on the viewer
        '''
        self.layers['cell_rgb'].data = self.label_vols['cell_rgb']
        self.layers['non_cell_rgb'].data = self.label_vols['non_cell_rgb']

        self.layers['cell_rgb'].refresh()
        self.layers['non_cell_rgb'].refresh()
    
    def compute_roi_features(self):
        '''
        All features that appear on the histograms are computed here. Each feature should have a 1d 
        array of size n_roi in self.roi_features, and a human-readable name in self.roi_feature_names.
        Any feature computed here will automatically get added to histograms in create_and_update_histograms
        '''
        self.roi_features['npix'] = n.array([len(stat['lam']) for stat in self.stats])
        self.roi_features['corrmap_val'] = n.array([stat['peak_val'] for stat in  self.stats])
        self.roi_features['act_thresh'] = n.array([stat['threshold'] for stat in  self.stats])

        self.roi_feature_names['npix'] = '# Pixels in ROI'
        self.roi_feature_names['corrmap_val'] = 'Max val of ROI in corr. map'
        self.roi_feature_names['act_thresh'] = 'Activity threshold'

        saved_ranges = self.load_file('histogram_curation.npy')

        for key in self.roi_features.keys():
            if saved_ranges is not None and key in saved_ranges.keys():
                if self.verbose: print("Loading saved ranges for %s" % key)
                self.roi_feature_ranges[key] = saved_ranges[key]
            else:
                vals = self.roi_features[key]
                self.roi_feature_ranges[key] = vals.min(), vals.max()
            
            self.filter_toggles[key] = False

    def create_histograms(self, plot_area):
        # if we are including the click histogram, leave the first row empty
        if 'click' in self.hist_graphs.keys():
            start_idx = 1
        else: start_idx = 0
        for idx, key in enumerate(self.roi_features.keys()):
            # get the name of the feature, and array of vals
            feature_name = self.roi_feature_names[key]
            self.hist_graphs[key] = pg.BarGraphItem(x=[0], height=[1], width=1)
            # add the histogram to the plot area
            print("Adding plot to %s" % str(plot_area))
            self.hist_plots[key] = plot_area.addPlot(row=idx+start_idx, col=1, title=feature_name)
            self.hist_plots[key].addItem(self.hist_graphs[key])

            
            # add draggable max/min lines to the histograms
            min_line = pg.InfiniteLine(pos=0, movable=True)
            max_line = pg.InfiniteLine(pos=1, movable=True)            
            self.hist_plots[key].addItem(min_line)
            self.hist_plots[key].addItem(max_line)
            self.hist_range_lines[key] = [min_line, max_line]            
            # when a movement of the line is finished, call the update_histogram_ranges function
            min_line.sigPositionChangeFinished.connect(functools.partial(self.update_feature_ranges, key))
            max_line.sigPositionChangeFinished.connect(functools.partial(self.update_feature_ranges, key))

    def update_histograms(self):
        '''
        update the histograms. Show only ROIs that are selected in the 'base_labels'
        '''
        nbins = self.display_params['histogram_nbins']
        
        # loop over all of the computer ROI features
        n_hist = len(self.roi_features.keys())
        for idx,key in enumerate(self.roi_features.keys()):
            feature = self.roi_features[key]
            feature = feature[self.base_labels]
            # compute histogram and plot it 
            vals, bins = n.histogram(feature, bins=nbins)
            bin_centers = (bins[1:] + bins[:-1])/2
            width = n.diff(bins).mean()

            self.hist_graphs[key].setOpts(x = bin_centers, height=vals, width=width)

            min_line, max_line = self.hist_range_lines[key]
            min_line.setPos(self.roi_feature_ranges[key][0])
            max_line.setPos(self.roi_feature_ranges[key][1])
            min_line.setBounds((bins[0]-width, bins[-1]+width))
            max_line.setBounds((bins[0]-width, bins[-1]+width))
        self.update_histogram_titles()

        
    def update_histogram_titles(self):
        for key in self.roi_features.keys():
            self.hist_plots[key].setTitle(self.roi_feature_names[key] + '. Range: %.1f - %.1f' % self.roi_feature_ranges[key])

    def update_feature_ranges(self, key, save=True):
        '''
        read the position of lines on the curation histograms, and update the displayed ROIs

        Args:
            key (_type_): _description_
        '''
        min_val = self.hist_range_lines[key][0].pos()[0]
        max_val = self.hist_range_lines[key][1].pos()[0]
        self.roi_feature_ranges[key] = min_val, max_val
        self.update_histogram_titles()
        if save:
            self.save_file('histogram_curation.npy', self.roi_feature_ranges)
            if self.verbose: print("Saving curation ranges")
        self.update_displayed_roi_labels()
    
    def update_displayed_roi_labels(self):
        '''
        Update the cell/not-cell labels of displayed ROIs with the various filters.
        Only uses the filter sources that are toggled on in self.filter_toggles
        '''
        # print("UPDATING DISPLAY")
        old_labels = self.display_roi_labels.copy()
        self.display_roi_labels[:] = self.base_labels.copy()
        n_total_roi = self.display_roi_labels.sum()
        if 'click' in self.filter_toggles.keys() and self.filter_toggles['click']:
            # get the ROIs that have been click-marked as cells/non-cells
            click_noncells = (self.click_curations['current'] == 0)
            click_cells = (self.click_curations['current'] == 1)
            self.display_roi_labels[click_noncells] = False
            self.display_roi_labels[click_cells] = True
        for key in self.roi_features.keys():
            if self.filter_toggles[key]:
                vals = self.roi_features[key]
                vmin, vmax = self.roi_feature_ranges[key]
                feature_cells = (vals >= vmin) & (vals <= vmax)
                self.display_roi_labels &= feature_cells
        n_final_roi = self.display_roi_labels.sum()

        update_label_vols(self.label_vols, old_labels, self.display_roi_labels,
                          self.coords, self.lams, self.display_params['lam_max'], 
                          self.display_params['cmap'])
        self.update_cells_in_viewer()

    def create_toggles(self, plot_area):
        if 'click' in self.hist_graphs.keys():
            toggle_button_keys = ['click'] + list(self.roi_features.keys())
        else: 
            toggle_button_keys = list(self.roi_features.keys())
        for idx,key in enumerate(toggle_button_keys):
            self.toggle_buttons[key] = QCheckBox()
            self.toggle_buttons[key].setCheckable(True)
            self.toggle_buttons[key].setChecked(self.filter_toggles[key])
            self.toggle_buttons[key].clicked.connect(functools.partial(self.toggle_filter, filter_key=key))
            self.toggle_buttons[key].setStyleSheet(qCheckedStyle)

            self.toggle_buttons_proxy[key] = QGraphicsProxyWidget()
            self.toggle_buttons_proxy[key].setWidget(self.toggle_buttons[key])
            # self.toggle_area.addItem(self.toggle_buttons_proxy[key], row=idx, col=0)
            
            plot_area.addItem(self.toggle_buttons_proxy[key], row=idx, col=0)
    
    def toggle_filter(self, filter_key):
        self.filter_toggles[filter_key] = self.toggle_buttons[filter_key].isChecked()
        # print("Set filter %s to %s" % (filter_key, str(self.filter_toggles[filter_key])))
        self.update_displayed_roi_labels()

class CurationUI(GenericNapariUI):
    '''
    UI object
    '''
    def __init__(self, base_path = None, verbose=True, display_params={}):
        '''
        Create a UI object in a given directory

        Args:
            base_path (str, optional): Path to directory containing stats.npy. Defaults to None.
            verbose (bool, optional): Defaults to True.
        '''
        
        super().__init__(base_path, verbose, display_params)
        self.info     = {}
        self.activity = {}
        self.nframes_total = 0
        self.iscells  = {}
        self.display_roi_labels = n.empty(0)
        self.click_curations = {}
        self.base_labels = None
        



        self.n_roi_activity = None

    def load_outputs(self):
        """ 
        Loads the outputs of cell detection. If base_path is not specified,
        will load from the current working directory. Directory should include
        stats.npy, info.npy and iscell.npy at a minimum.
        """
        # load the stats.npy file
        # First, attempt to load in stats_small. This file doesn't have the neuropil
        # coordinates, which inflate the size by 3-4x. 
        self.stats = self.load_file('stats_small.npy')
        if self.stats is None:
            self.stats = self.load_file('stats.npy')
            self.log("Loaded full stats.npy")
        else:
            self.log("Loaded stats_small.npy")


        
        # unpack coords and lams from stats
        self.coords = [stat['coords'] for stat in self.stats]
        self.lams = [stat['lam'] for stat in self.stats]
        self.meds = n.array([stat['med'] for stat in self.stats])
        # number of ROIs 
        self.n_roi = len(self.coords)

        # by default, display all ROIs 
        # this is done by setting base_labels to all ones
        # users have the option to load the iscell file from the dropdown
        self.base_labels = n.ones(self.n_roi, bool)
        self.display_roi_labels = self.base_labels.copy()

        # load the info.npy file
        self.info = self.load_file('info.npy')
        self.log("Loaded info.npy")
        # save the shape of the volume 
        self.shape = self.info['vmap'].shape

        # other  files to look for in the path
        activity_files = ['F', 'Fneu', 'spks'] # these will be memmapped
        iscell_files = ['iscell', 'iscell_extracted']
        self.activity = {}
        self.iscells = {}

        if self.verbose: print("Looking for activity files")
        for file in activity_files:
            self.activity[file] = self.load_file(file + '.npy', mmap_mode='r')
            if self.activity[file] is not None:
                self.display_activity = True

                # check the number of n_rois in the activity files
                file_nroi = self.activity[file].shape[0]
                self.nframes_total = self.activity[file].shape[1]
                if self.n_roi_activity is not None:
                    # make sure all activity.npy files have the same number of ROIs
                    if file_nroi != self.n_roi_activity:
                        warn("Mismatch between the number of extracted ROIs. Found %d ROIs in %s.npy, expected %d from other activity files. Re-run extraction" % (file_nroi, file, self.n_roi_activity))
                        self.display_activity=False
                        break
                else:
                    self.n_roi_activity = file_nroi

        if self.verbose: print("Loading iscell files")
        for file in iscell_files:
            self.iscells[file] = self.load_file(file + '.npy')
        
        # if we don't find iscell_extracted, that probably means extraction hasn't happened
        # maybe this should mean that no traces should be displayed?
        # for now, if iscell_extracted isn't found, assume all ROIs are extracted. We'll
        # double check that self.n_roi is equal to the number of rois in the activity files
        # the displayed cell/not-cell labels, a 1D array of size n_roi     
        if self.display_activity:
            if self.iscells['iscell_extracted'] is None:
                warn(warnings['missing_iscell_extracted'], RuntimeWarning)
                self.iscells['iscell_extracted'] = n.ones((self.n_roi,2))
            self.n_roi_extracted = self.iscells['iscell_extracted'][:,0].sum()
            self.extracted_roi_idxs = n.where(self.iscells['iscell_extracted'][:,0])[0]
            self.extracted_roi_flag = self.iscells['iscell_extracted'][:,0]
            if self.n_roi_extracted != self.n_roi_activity:
                warn(warnings['activity_mismatch'],RuntimeWarning)
                self.display_activity=False



    def create_ui(self):
        self.make_all_label_vols()
        self.load_click_curations()
        self.compute_roi_features()

        self.start_viewer()
        self.add_images_to_viewer()
        self.add_cells_to_viewer()

        self.build_curation_window()
        self.add_click_curation_callbacks()
        self.create_click_plot()
        self.update_click_plot()

        self.create_histograms(self.curation_plot_area)
        self.update_histograms()
        self.create_save_button()
        self.create_toggles(self.curation_plot_area)
        self.create_base_labels_dropdown()

        self.dock_curation_window()

        if self.display_activity:
            self.build_activity_window()
            self.create_activity_plot()
            self.add_activity_callbacks()
            self.dock_activity_window()

    def load_click_curations(self):
        '''
        Cells curated by clicking are saved into click_curations.npy. This contains a dictionary. The 'current' element of the dictionary contains a 1D array of size n_rois which is 1 for ROIs that have been marked as cells, 0 for ROIs marked as non-cells, and None for ROIs that have not been marked. Other elements of the dictionary can correspond to previously saved curations in future versions.

        Args:
            iscell (ndarray, optional): (n_roi, 2) array. Defaults to None.
            base_path (Path, optional): Defaults to None.

        Returns:
            _type_: _description_
        '''
        self.click_curations = self.load_file('click_curations.npy')
        if self.click_curations is None:
            print("Creating click_curations.npy")
            click_curation = n.empty(self.n_roi, float)
            click_curation[:] = n.nan
            self.click_curations = {'current' : click_curation}
            self.save_file('click_curations.npy',self.click_curations)
        else:
            print("Loaded click-curations with %d cells and %d non-cells marked" \
                  % ((self.click_curations['current'] == 1).sum(), (self.click_curations['current'] == 0).sum()))
            
        self.filter_toggles['click'] = False
        
    def add_images_to_viewer(self):
        '''
        Take the mean, maximum and correlation map images from info.npy and add them to the UI
        '''
        scale = self.display_params['scale']
        pmin, pmax = self.display_params['contrast_percentiles']
        image_keys = [ 'max_img', 'mean_img' ,'vmap']
        image_labels = ['Max Image', 'Mean Image', 'Corr. Map', ]
        for image_key, image_label in zip(image_keys, image_labels):
            image = self.info[image_key]
            clims = get_percentiles(image, pmin=pmin,pmax=pmax)
            self.layers[image_key] = self.viewer.add_image(image, name=image_label, 
                                    contrast_limits=clims, scale=scale)

        
        # self.update_displayed_roi_labels()

    def add_activity_callbacks(self):
        '''
        Add callbacks to the cell_rgb and non_cell_rgb layers to allow right-click labelling
        '''
        cell_layer = self.layers['cell_rgb']
        non_cell_layer = self.layers['non_cell_rgb']

        @cell_layer.mouse_drag_callbacks.append
        def click_handler_cell_layer(layer, event):
            if event.button == 1: 
                cz,cy,cx = n.array(layer.world_to_data(event.position)).astype(int)
                roi_idx = self.get_roi_idx_from_position(cz,cy,cx, cell=True)
                self.update_activity_plot(roi_idx)
        @non_cell_layer.mouse_drag_callbacks.append
        def click_handler_non_cell_layer(layer, event):
            if event.button == 1:
                cz,cy,cx = n.array(layer.world_to_data(event.position)).astype(int)
                roi_idx = self.get_roi_idx_from_position(cz,cy,cx, cell=False)
                self.update_activity_plot(roi_idx)

    def add_click_curation_callbacks(self):
        '''
        Add callbacks to the cell_rgb and non_cell_rgb layers to allow right-click labelling
        '''
        cell_layer = self.layers['cell_rgb']
        non_cell_layer = self.layers['non_cell_rgb']

        @cell_layer.mouse_drag_callbacks.append
        def click_handler_cell_layer(layer, event):
            if event.button == 2: 
                cz,cy,cx = n.array(layer.world_to_data(event.position)).astype(int)
                self.mark_roi(cz,cy,cx, mark_as_cell=False)
        @non_cell_layer.mouse_drag_callbacks.append
        def click_handler_non_cell_layer(layer, event):
            if event.button == 2:
                cz,cy,cx = n.array(layer.world_to_data(event.position)).astype(int)
                self.mark_roi(cz,cy,cx, mark_as_cell=True)

    def get_roi_idx_from_position(self, cz, cy, cx, cell=True):
        '''
        Return the roi_idx that exists at a given voxel

        Args:
            cz (int): z coord
            cy (int): y coord
            cx (int): x coord
            cell (True, optional): If true, looks for ROI in cells. If False, looks for it in non-cells. Defaults to True.
        '''
        if cell: 
            idxs = self.label_vols['cell_idxs']
        else:
            idxs = self.label_vols['non_cell_idxs']
        roi_idx = idxs[cz, cy, cx]
        return roi_idx

    def mark_roi(self, cz, cy, cx, mark_as_cell=True):
        '''
        Given the coordinate of a voxel, label the cell in this voxel as a cell/non-cell
        and update the displays. cell=True means mark as cell, cell=False means mark as non-cell
        '''
        # if we are marking it as a cell, look for it in non-cells, so pass ~mark_as_cell
        roi_idx = self.get_roi_idx_from_position(cz,cy,cx,~mark_as_cell)
        if roi_idx < 1: # no ROI at this location
            return
        if self.verbose:
            print("Marking ROI %d as %d" % (roi_idx, int(mark_as_cell)))
        self.click_curations['current'][roi_idx] = int(mark_as_cell)
        self.save_file('click_curations.npy', self.click_curations, overwrite=True)
        self.update_click_plot()
        self.update_displayed_roi_labels()


    def build_activity_window(self):
        # this window contains the activity traces for the selected cell
        self.activity_window = pg.GraphicsLayoutWidget()
        self.activity_plot_area = pg.GraphicsLayout()

        self.activity_window.addItem(self.activity_plot_area, row=0, col=0)

    def create_activity_plot(self):
        self.activity_curves = {}
        self.activity_pens = {}
        self.activity_plot_bounds = [0,min(self.display_params['n_frames_plotted'], self.nframes_total)]
        self.activity_plot = self.activity_plot_area.addPlot(row=0, col=0)
        self.activity_plot.addLegend()
        self.activity_plot.setTitle("Click on an ROI to see its activity traces.")
        for key in self.activity.keys():
            # pen = pg.mkPen(color = self.display_params[key + "_color"],
                        #    width =  self.display_params['line_width'])
            # self.activity_pens[key] = pen
            # self.activity_curves[key] = pg.PlotCurveItem([0],[0], pen=self.display_params[key + '_color'])
            self.activity_curves[key] = pg.PlotCurveItem([0],[0], name = key,
                                             pen=pg.mkPen(self.display_params[key + '_color'],
                                                          width = self.display_params['line_width']))
            self.activity_plot.addItem(self.activity_curves[key])

                    
        xax = self.activity_plot.getAxis('bottom')
        xax.setLabel("Time (s)")

    def update_activity_plot_title(self, roi_idx, extracted=True):
        roi_is_cell = self.display_roi_labels[roi_idx]
        if not extracted:
            self.activity_plot.setTitle('ROI %05d was not extracted' % roi_idx)
        else:
            cell_str = 'cell' if roi_is_cell else 'non-cell'
            cz,cy,cx = self.meds[roi_idx]

            self.activity_plot.setTitle("ROI %05d. Position (pix): %02d, %03d, %03d (%s)" % (roi_idx,cz, cy,cx, cell_str)) 
    def dock_activity_window(self):
        # dock the activity window to napari
        self.docked_activity_window = self.viewer.window.add_dock_widget(self.activity_window, name='Activity', area='bottom')

    def update_activity_plot(self, roi_idx):

        # first and last frames to plot
        fmin, fmax = self.activity_plot_bounds
        frame_times = n.arange(fmin, fmax) / self.info['all_params']['fs']

        corrected_roi_idx = roi_idx
        if roi_idx == -1 or not self.extracted_roi_flag[roi_idx]:
            # if selected voxel is not in an ROI, or if the selected ROI was not extracted
            for key in self.activity.keys():
                self.activity_curves[key].setData(x=[0],y=[0])
            self.update_activity_plot_title(roi_idx, extracted=False)
            return
        
        if self.n_roi_extracted != self.n_roi:
            # If not all ROIs were extracted, we need to do this correction:
            # If ROI 5 was not extracted, then ROI 6 will correspond to row 5 of activity .npy files
            # correct for this, so roi_idx is the original idx in stats.npy, and corrected_ is the 
            # corresponding one in spks.npy, etc
            corrected_roi_idx = n.where(self.extracted_roi_idxs == roi_idx)
            if self.verbose:
                print("ROI %d is the %d-th extracted ROI since not all detected ROIs were extracted. Double check this code!")
            
        for key in self.activity.keys():
            if self.activity[key] is None:
                continue
            self.activity_curves[key].setData(x = frame_times,
                                              y = self.activity[key][corrected_roi_idx, fmin:fmax])
        
        self.update_activity_plot_title(roi_idx, extracted=True)
    
    def build_curation_window(self):

        # this is the window that will contain the histograms, buttons etc.
        self.curation_window = pg.GraphicsLayoutWidget()

        # create the plot and button areas and attach them to curation_window
        self.curation_plot_area = pg.GraphicsLayout()
        self.button_area = pg.GraphicsLayout()
        self.dropdown_area = pg.GraphicsLayout()
        self.curation_window.addItem(self.dropdown_area, row=0,col=0)
        self.curation_window.addItem(self.curation_plot_area, row=1,col=0)
        self.curation_window.addItem(self.button_area, row=2, col=0)

    def dock_curation_window(self):
        # dock the curation window to napari
        self.docked_curation_window = self.viewer.window.add_dock_widget(self.curation_window, name='ROI Features', area='right')


    def create_click_plot(self):
        '''
        make an empty plot showing the number of marked and unmarked ROIs and add it to self.curation_plot_area
        '''
        self.hist_graphs['click'] = pg.BarGraphItem(x=[1,2,3], height=[1,1,1], width=1)
        self.hist_plots['click'] = self.curation_plot_area.addPlot(row=0, col=1, title='# click-curated ROIs')
        self.hist_plots['click'].addItem(self.hist_graphs['click'])

    def update_click_plot(self):
        '''
        update the bar graph showing # of click-labelled cells, non-cells and unmarked ROIs
        '''
        click_cells = (self.click_curations['current'] == 1).sum()
        click_non_cells = (self.click_curations['current'] == 0).sum()
        click_unmarked = n.isnan(self.click_curations['current']).sum()
        ys = n.log10([click_cells + 0.1, click_non_cells + 0.1, click_unmarked + 0.1])

        # self.hist_graphs['click'].setData('x', xs)
        self.hist_graphs['click'].setOpts(height = ys)
        
        self.hist_plots['click'].setYRange(0,ys[-1]+0.5)
        
        xax = self.hist_plots['click'].getAxis('bottom')
        xax.setTicks([[(1, 'Cells'), (2, 'Non-cells'), (3, 'Unmarked')], []])
    
        yax = self.hist_plots['click'].getAxis('left')
        yax.setTicks([[(ys[0], str(click_cells)), (ys[1], str(click_non_cells)), (ys[2], str(click_unmarked))], []])


    def create_base_labels_dropdown(self):
        self.dropdown = QComboBox()
        self.dropdown.addItems(['all', 'iscell.npy'])
        self.dropdown.setStyleSheet(dropdown_style)
        self.dropdown.currentTextChanged.connect(self.update_base_labels)
        self.dropdown_proxy = QGraphicsProxyWidget()
        self.dropdown_proxy.setWidget(self.dropdown)
        self.dropdown_area.addItem(self.dropdown_proxy, row=0, col=0)
    
    def update_base_labels(self, label):
        '''
        Update the base labels. Load the labels and disable all filters.
        '''
        if label == 'all':
            self.base_labels = n.ones(self.n_roi, bool)
        elif label == 'iscell.npy':
            self.base_labels = self.iscells['iscell'][:,0].astype(bool)
        elif label == 'iscell_extracted.npy':
            self.base_labels = self.iscells['iscell_extracted'][:,0].astype(bool)
        
        for key in self.filter_toggles.keys():
            # print("Setting button %s to False" % key)
            # set the filter button, and toggle the filter for the display
            self.toggle_buttons[key].setChecked(False)
            self.toggle_filter(key)
        self.update_displayed_roi_labels()
        self.update_histograms()

    def create_save_button(self):
        self.save_button = QPushButton('button', text='save to iscell.npy')
        self.save_button.clicked.connect(self.update_iscell)
        self.save_button.setStyleSheet(basicButtonStyle)
        self.save_button_proxy = QGraphicsProxyWidget()
        self.save_button_proxy.setWidget(self.save_button)
        self.button_area.addItem(self.save_button_proxy, row=0, col=0)

    def update_iscell(self):
        self.iscells['iscell'][:] = self.display_roi_labels.astype(int)[:,n.newaxis]
        if self.verbose:
            print("Updating iscell with %d / %d ROIs marked as cells" % (self.display_roi_labels.sum(),self.n_roi))
        self.save_file('iscell.npy', self.iscells['iscell'], overwrite=True)
        
        if self.verbose:
            print("Saving new iscell.npy with %d / %d cells" % (self.display_roi_labels.sum(), self.n_roi))
    


def make_label_vols(coords, lams, shape, lam_max = 0.3, iscell_1d=None, cmap='Set3'):
    
    '''
    Make an RGBA volume with voxels occupied by cells having random colours

    Args:
        coords (list): list of size n_rois, each element is a list of size 3, with lists of z,y,x coordinates
        lams (list) : similar to coords, each element is a list of lams for each cell
        shape (tuple): shape of the volume to fill up
        iscell_1d (ndarray, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to 'Set3'.
    '''

    cmap = plt.get_cmap(cmap)
    n_cmap = cmap.N

    cell_idxs_vol = -n.ones(shape, int)
    cell_rgb_vol = n.zeros(shape + (4,))

    n_cells = len(coords)
    if iscell_1d is None:
        iscell_1d = n.ones(n_cells)
    for i in range(len(coords)):
        # get the coordinates and cell projection values for this cell
        cz,cy,cx = coords[i]
        lam = copy.copy(lams[i])
        lam /= lam_max; lam[lam > 1] = 1
        if iscell_1d[i]: # if is cell, add to cell volumes
            cell_idxs_vol[cz,cy,cx] = i
            cell_rgb_vol[cz,cy,cx, :3] += cmap(i % n_cmap)[:3]
            cell_rgb_vol[cz,cy,cx, 3] += lam

    # RGB values are summed for voxels in multiple cells
    # make sure the maximum is 1
    # cell_rgb_vol[cell_rgb_vol > 1][:,:3] = cell_rgb_vol[cell_rgb_vol > 1][:,:3] % 1
    cell_rgb_vol[cell_rgb_vol > 1] = 1
    return cell_idxs_vol, cell_rgb_vol

def update_label_vols(label_vols, old_roi_labels, new_roi_labels, coords, lams, 
                      lam_max=0.3, cmap='Set3'):
    '''
    Change the label volumes based on updated roi labels of cell/not-cell

    Args:
        label_vols (dict): output of make_all_label_vols
        old_roi_labels (ndarray): of size n_roi; 1 for cell, 0 for non-cell
        new_roi_labels (ndarray): same as old_roi_labels
    '''
    cmap = plt.get_cmap(cmap)
    n_cmap = cmap.N
    # compare the old and new labels, and find the cell indices that have changed labels
    changed_idxs = n.where((new_roi_labels != old_roi_labels))[0]
    n_changed_rois = len(changed_idxs)
    if n_changed_rois == 0: 
        return label_vols
    
    # loop through all changed cells
    for roi_idx in changed_idxs:
        cz,cy,cx = coords[roi_idx]
        lam = copy.copy(lams[roi_idx])
        lam /= lam_max; lam[lam > 1] = 1
        if new_roi_labels[roi_idx] == 1: # if this ROI is marked a cell
            label_vols['non_cell_idxs'][cz,cy,cx] = -1 # remove from non-cell idxs
            label_vols['cell_idxs'][cz,cy,cx] = roi_idx # add to cell idxs
            label_vols['non_cell_rgb'][cz,cy,cx, :3] -= cmap(roi_idx % n_cmap)[:3]
            label_vols['non_cell_rgb'][cz,cy,cx, 3] -= lam # remove from non-cell rgb
            label_vols['cell_rgb'][cz,cy,cx, :3] += cmap(roi_idx % n_cmap)[:3]
            label_vols['cell_rgb'][cz,cy,cx, 3] += lam # add to cell rgb
        elif new_roi_labels[roi_idx] == 0: # if this ROI is marked a non-cell
            label_vols['cell_idxs'][cz,cy,cx] = -1 # remove from cell idxs
            label_vols['non_cell_idxs'][cz,cy,cx] = roi_idx # add to non-cell idxs
            label_vols['cell_rgb'][cz,cy,cx, :3] -= cmap(roi_idx % n_cmap)[:3]
            label_vols['cell_rgb'][cz,cy,cx, 3] -= lam # remove from cell rgb
            label_vols['non_cell_rgb'][cz,cy,cx, :3] += cmap(roi_idx % n_cmap)[:3]
            label_vols['non_cell_rgb'][cz,cy,cx, 3] += lam # add to non-cell rgb

    return label_vols


class SweepUI(GenericNapariUI):
    def __init__(self, base_path=None, verbose=True, display_params = {}):       
        '''
        Create a UI object in a given directory

        Args:
            base_path (str, optional): Path to directory containing stats.npy. Defaults to None.
            verbose (bool, optional): Defaults to True.
        '''
        super().__init__(base_path, verbose, display_params)
        self.sweep_type = None

    def load_outputs(self):
        self.sweep_summary = self.load_file('sweep_summary.npy')

        self.get_sweep_params()
        self.log("Loaded summary for sweep of type: %s" % self.sweep_type)
        self.log("Sweep over %d total combinations, varying the following parameters:" % self.n_combinations)

        for param in self.param_names:
            self.log("%2d values for %s: %s" % (len(self.param_dict[param]), param, str(self.param_dict[param])), 1)

        if self.sweep_type == 'corrmap':
            self.sweep_results, self.sweep_params = collate_sweep_results(self.sweep_summary,
                                                                          result_key='corrmap')
            self.vol_shape = self.sweep_summary['mean_img'].shape
            self.mean_img = self.sweep_summary['mean_img']
            self.max_img = self.sweep_summary['max_img']

        elif self.sweep_type == 'segmentation':
            self.sweep_results, self.sweep_params = collate_sweep_results(self.sweep_summary,
                                                                          result_key='stats')
            self.info = self.sweep_summary['results'][0]['info']
            self.mean_img = self.info['mean_img']
            self.max_img = self.info['max_img']
            self.corr_map = self.info['vmap']
            self.shape = self.corr_map.shape

        if self.all_combinations:
            self.current_index = n.zeros(self.n_params, int)
            self.current_result = self.sweep_results[tuple(self.current_index)]
            self.current_params = self.sweep_params[tuple(self.current_index)]
        else:
            self.current_param = self.sweep_params[0]
            self.current_index = 0
            self.current_result = self.sweep_results[self.current_param][self.current_index]
            self.current_params = self.sweep_params[self.current_param][self.current_index]
        
        if self.sweep_type == 'corrmap':
            self.corr_map = self.current_result

        if self.sweep_type == 'segmentation':
            self.stats = self.current_result
            self.unpack_stats()
            self.compute_roi_features()
            
            

    def create_ui(self):
        self.start_viewer()
        self.add_background_images_to_viewer()
        self.add_corrmap_to_viewer()
    
        if self.sweep_type == 'segmentation':
            self.make_all_label_vols()
            self.build_histogram_window()
            self.add_cells_to_viewer()
            self.build_histogram_window()
            self.create_histograms(self.histogram_plot_area)
            self.create_toggles(self.histogram_plot_area)
            self.update_histograms()
            self.dock_histogram_window()

        self.create_param_display()

    def build_histogram_window(self):
        self.histogram_window = pg.GraphicsLayoutWidget()
        self.histogram_plot_area = pg.GraphicsLayout()
        self.histogram_window.addItem(self.histogram_plot_area)
    
    def dock_histogram_window(self):
        # dock the curation window to napari
        self.docked_histogram_window = self.viewer.window.add_dock_widget(self.histogram_window, name='ROI Features', area='right')

    def display_current_result(self):
        if self.sweep_type == 'corrmap':
            self.update_corr_map(self.current_result)
        elif self.sweep_type == 'segmentation':
            self.stats = self.current_result
            # get the coordaintes of each ROI, and compute ROI features
            self.unpack_stats()
            self.compute_roi_features()
            # create the cell label volume and display it 
            self.make_all_label_vols()
            self.update_cells_in_viewer()
            # update the histograms with new features
            # and apply the current filters 
            self.update_histograms()
            self.update_displayed_roi_labels()
            


    def get_sweep_params(self):
        self.param_dict = self.sweep_summary['param_sweep_dict']
        self.param_names = self.sweep_summary['param_names']
        self.n_params = len(self.param_names)
        self.combinations = self.sweep_summary['combinations']
        
        self.sweep_type = self.sweep_summary.get('sweep_type', 'segmentation')
        self.all_combinations = self.sweep_summary['all_combinations']

        self.n_combinations = len(self.combinations)
        self.n_vals_per_param = tuple([len(self.param_dict[k]) for k in self.param_names])
        # self.disp_shape = self.n_vals_per_param + self.vol_shape
        # self.disp_scale = tuple([1] * self.n_params) + self.display_params['scale']
        # self.axis_labels = tuple(self.param_names + ['z','y','x'])

    def add_background_images_to_viewer(self):
        scale = self.display_params['scale']
        pmin, pmax = self.display_params['contrast_percentiles']

        clims = get_percentiles(self.mean_img, pmin, pmax)
        self.viewer.add_image(self.mean_img, name='Mean Image', scale = scale, contrast_limits=clims)

        clims = get_percentiles(self.max_img, pmin, pmax)
        self.viewer.add_image(self.max_img, name='Max Image', scale = scale, contrast_limits=clims)

    def add_corrmap_to_viewer(self):
        scale = self.display_params['scale']
        pmin, pmax = self.display_params['contrast_percentiles']
        clims = get_percentiles(self.corr_map, pmin, pmax)
        self.layers['corr_map'] = self.viewer.add_image(self.corr_map, name='Correlation Map', scale = scale, contrast_limits=clims)

    def update_corr_map(self, new_corr_map):
        self.corr_map = new_corr_map
        self.layers['corr_map'].data = self.corr_map
        self.layers['corr_map'].refresh()

    def unpack_stats(self):
        # unpack coords and lams from stats
        self.coords = [stat['coords'] for stat in self.stats]
        self.lams = [stat['lam'] for stat in self.stats]
        self.meds = n.array([stat['med'] for stat in self.stats])
        # number of ROIs 
        self.n_roi = len(self.coords)
        self.display_roi_labels = n.ones(self.n_roi, bool)
        self.base_labels = n.ones(self.n_roi, bool)


    def create_param_display(self):
        self.param_display_window = pg.GraphicsLayoutWidget()
        self.all_param_display_area = pg.GraphicsLayout()
        self.param_display_window.addItem(self.all_param_display_area)

        self.param_labels = {}
        self.param_dropdowns = {}
        self.param_display_areas = {}
        for pidx, param_name in enumerate(self.param_names):
            possible_vals = self.param_dict[param_name]
            dropdown = QComboBox()
            label = QLabel()            
            dropdown.setStyleSheet(dropdown_style)
            label.setStyleSheet(dropdown_style)

            param_display_area = pg.GraphicsLayout()

            label_proxy = QGraphicsProxyWidget()
            label_proxy.setWidget(label)
            dropdown_proxy = QGraphicsProxyWidget()
            dropdown_proxy.setWidget(dropdown)

            label.setText(param_name)
            dropdown.addItems([str(val) for val in possible_vals])
            
            param_display_area.addItem(label_proxy, row=1, col=0)
            param_display_area.addItem(dropdown_proxy, row=0, col=0)
            self.all_param_display_area.addItem(param_display_area, row=0, col=pidx)

            callback = functools.partial(self.select_param, param_index=pidx )
            dropdown.activated.connect(callback)

            self.param_dropdowns[param_name] = dropdown
            self.param_labels[param_name] = label 
        self.log("Docking params")
        self.docked_param_window = self.viewer.window.add_dock_widget(self.param_display_window, name='Params', area='left')

    def select_param(self, param_val_index,param_index):
        self.log("Setting param %s to %s" % (self.param_names[param_index], 
                                                str(self.param_dict[self.param_names[param_index]][param_val_index])))
        if self.all_combinations:
            self.current_index[param_index] = param_val_index 
            self.current_result = self.sweep_results[tuple(self.current_index)]
            self.current_params = self.sweep_params[tuple(self.current_index)]
        else:
            self.current_index = param_val_index
            self.current_param = self.param_names[param_index]
            self.current_result = self.sweep_results[self.current_param][self.current_index]
            self.current_params = self.sweep_params[self.current_param][self.current_index]
            self.fix_dropdown_values()
        self.display_current_result()

    def fix_dropdown_values(self):
        if not self.all_combinations:
            for param_name in self.param_names:
                if param_name == self.current_param: continue
                param_val = self.current_params[param_name]
                param_val_idx = n.where(self.param_dict[param_name] == param_val)[0]
                self.param_dropdowns[param_name].setCurrentIndex(param_val_idx)
            

def get_percentiles(image, pmin=1, pmax=99, eps = 0.0001):
    '''
    return the lower and higher percentiles of an image

    Args:
        image (ndarray): image (arbitrary dimensions & size)
        pmin (float, optional): lower percentile, 0-100. Defaults to 1.
        pmax (float, optional): higher percentile, 0-100. Defaults to 99.
        eps (float, optional): Defaults to 0.0001.

    Returns:
        _type_: _description_
    '''
    im_f = image.flatten()
    vmin = n.percentile(im_f, pmin)
    vmax = n.percentile(im_f, pmax) + eps
    return vmin, vmax


def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Standalone viewer of Suite3D outputs.')
    parser.add_argument('type', type=str, default='curation')
    parser.add_argument('--output_dir', type=Path,default=None,
                    help='Path to directory containing the Suite3D output, including stats.npy.')
    return parser


def collate_sweep_results(sweep_summary, result_key = 'stats'):
    '''
    Load the results of a sweep and arrange in a way that is easy to visualize

    Args:
        sweep_summary (dict): Loaded from sweep_summary.npy
        result_key (str, optional): Key of the saved result from each run we care about. For a corrmap sweep, it should be corrmap. For a segmentation sweep, it should be stats.

    Returns:
        if all combinations of parameters were swept, returns a P-dim array where P is the number of parameters that were swept, and each coordinate corresponds to combinations of coordinate values
        else, it returns a dictionary, where each key is a parameter name, and the values are arrays of results for a sweep over the given parameter
    '''
    param_dict = sweep_summary['param_sweep_dict']
    param_names = sweep_summary['param_names']
    combinations = sweep_summary['combinations']
    results = sweep_summary['results']
    n_val_per_param = [len(param_dict[k]) for k in param_names]
    n_params = len(param_names)
    if sweep_summary['all_combinations']:
        collated_results = n.empty(n_val_per_param, dtype='O')
        param_values = n.empty(n_val_per_param, dtype='O')

        for cidx, combination in enumerate(combinations):
            param_idxs = [n.where(param_dict[param_names[pidx]] == combination[pidx])[0][0] \
                                for pidx in range(n_params)]
            collated_results[tuple(param_idxs)] = results[cidx][result_key]
            parvals = {}
            for param_idx, param_name in enumerate(param_names):
                parvals[param_name] = combinations[cidx][param_idx]
            param_values[tuple(param_idxs)] = parvals
    else:
        collated_results = {}
        param_values = {}
        cidx = 0
        for pidx in range(len(param_names)):
            collated_results[param_names[pidx]] = n.empty(n_val_per_param[pidx], dtype='O')
            param_values[param_names[pidx]] = n.empty(n_val_per_param[pidx], dtype='O')
            for vidx in range(n_val_per_param[pidx]):
                collated_results[param_names[pidx]][vidx] = results[cidx][result_key]
                parvals = {}
                for param_idx, param_name in enumerate(param_names):
                    parvals[param_name] = combinations[cidx][param_idx]
                param_values[param_names[pidx]][vidx] = parvals
                cidx += 1
        assert cidx == n.sum(n_val_per_param) - 1

    return collated_results, param_values


if __name__ == '__main__':
    
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    base_dir = parsed_args.output_dir
    if base_dir is not None:
        print("Running UI in %s" % base_dir.absolute())
    else:
        print("Running UI in current working dir")

    if parsed_args.type == 'curation':
        ui = CurationUI(base_dir)
    elif parsed_args.type == 'sweep':
        ui = SweepUI(base_dir)
    else:
        warn("Invalid argument")
    ui.load_outputs()
    ui.create_ui()
        

    napari.run()


