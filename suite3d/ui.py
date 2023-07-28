import os
import numpy as n
from matplotlib import pyplot as plt
import pyqtgraph as pg
import copy

try: 
    import napari
except:
    print("No Napari")

try: from napari._qt.widgets.qt_range_slider_popup import QRangeSliderPopup
except: pass

def load_outputs(output_dir, load_traces = False, trace_names = ['F', 'Fneu','spks']):
    files = os.listdir(output_dir)
    outputs = {}
    outputs['dir'] = output_dir
    info = n.load(os.path.join(output_dir, 'info.npy'),allow_pickle=True).item()
    outputs['vmap'] = info['vmap']
    outputs['max_img'] = info.get('max_img', n.zeros_like(outputs['vmap']))
    outputs['mean_img'] = info.get('mean_img', n.zeros_like(outputs['vmap']))
    outputs['fs'] = info['all_params']['fs']
    outputs['stats'] = n.load(os.path.join(output_dir, 'stats.npy'),allow_pickle=True)

    if 'iscell.npy' in files:
        outputs['iscell'] = n.load(os.path.join(output_dir, 'iscell.npy'))
    else:
        outputs['iscell'] = n.ones((len(outputs['stats'],2)))
    if 'iscell_extracted.npy' in files:
        outputs['iscell_extracted'] = n.load(os.path.join(output_dir, 'iscell_extracted.npy'))
    if 'iscell_curated.npy' in files:
        outputs['iscell_curated'] = n.load(os.path.join(output_dir, 'iscell_curated.npy'))
    if 'iscell_curated_slider.npy' in files:
        outputs['iscell_curated_slider'] = n.load(os.path.join(output_dir, 'iscell_curated_slider.npy'))

    if load_traces:
        traces = {}
        for filename in (trace_names):
            if filename + '.npy' in files:
                traces[filename] = n.load(os.path.join(output_dir, filename + '.npy'))
            else: print("Not found: %s.npy" % filename)
            assert len(traces) > 0
        outputs.update(traces)
    return outputs


def get_percentiles(image, pmin=1, pmax=99, eps = 0.0001):
    im_f = image.flatten()
    vmin = n.percentile(im_f, pmin)
    vmax = n.percentile(im_f, pmax) + eps
    return vmin, vmax

def make_label_vols(stats, shape, lam_max = 0.3, iscell=None, cmap='Set3'):
    coords = [stat['coords'] for stat in stats]
    lams = [stat['lam'] for stat in stats]
    cmap = plt.get_cmap(cmap)
    n_cmap = cmap.N
    cell_id_vol = n.zeros(shape, int)
    cell_rgb_vol = n.zeros(shape + (4,))
    if iscell is None: iscell = n.ones((len(stats),2))
    for i in range(len(stats)):
        if iscell[i,0]:
            cz,cy,cx = coords[i]
            lam = copy.copy(lams[i])
            # print(lam)3
            lam /= lam_max; lam[lam > 1] = 1
            cell_id_vol[cz,cy,cx] = i  + 1
            cell_rgb_vol[cz,cy,cx, :3] = cmap(i % n_cmap)[:3]
            # print(lam)
            cell_rgb_vol[cz,cy,cx, 3] = lam
    return cell_id_vol, cell_rgb_vol


def create_ui(outputs, cmap='Set3', lam_max = 0.3, scale=(15,4,4), iscell_label='iscell_extracted'):
    stats = outputs['stats']; mean_img = outputs['mean_img'];
    vmap = outputs['vmap']; max_img = outputs['max_img']
    iscell = outputs[iscell_label] if iscell_label in outputs.keys() else outputs['iscell']
    if len(iscell.shape) < 2: iscell = iscell[:,n.newaxis]
    shape = vmap.shape
    coords = [stat['coords'] for stat in stats]
    lams = [stat['lam'] for stat in stats]
    layers = {}
    cell_id_vol, noncell_id_vol, cell_rgb_vol, noncell_rgb_vol = update_vols(stats, shape,
                                        iscell, layers, cmap, lam_max, scale, update_layers=False)
    clims_mean = get_percentiles(mean_img, 0, 99.5)
    clims_max = get_percentiles(max_img, 0, 99.5)
    clims_vmap = get_percentiles(vmap, 0, 99.9)

    v = napari.Viewer()
    layers['max_img_layer'] = v.add_image(max_img,name='Max Image',contrast_limits=clims_max, scale=scale)
    layers['vmap_layer'] = v.add_image(vmap, name='Corr Map', contrast_limits=clims_vmap, scale=scale)
    layers['mimg_layer'] = v.add_image(mean_img, name='Mean Img', contrast_limits=clims_mean, scale=scale)
    layers['nvol_layer'] = v.add_image(noncell_rgb_vol, name='Non cells', rgb=True, 
                                       scale=scale, visible=False)
    layers['cvol_layer'] = v.add_image(cell_rgb_vol, name='Cells', rgb=True, scale=scale)
    layers['clabel'] = cell_id_vol.astype(int)
    layers['nclabel'] = noncell_id_vol.astype(int)
    # layers['nclabel_layer'] = v.add_labels(noncell_id_vol.astype(int), name='Non cells (labels)', 
                                        #    scale=scale, opacity=0)
    # layers['clabel_layer'] = v.add_labels(cell_id_vol.astype(int), name='Cells (labels)', 
                                        #   scale=scale, opacity=0)

    return v, layers

import datetime
def add_callbacks_to_ui(v, layers, outputs, savedir, add_sliders = True, filters=None, add_curation=True):
    spks = outputs['spks']; F = outputs.get('F', n.zeros_like(spks)); Fneu = outputs.get('Fneu', n.zeros_like(spks))

    if add_curation:
        iscell = outputs['iscell_extracted']
        if len(iscell.shape) < 2:
            iscell = iscell[:, n.newaxis]
        iscell_savepath = os.path.join(savedir, 'iscell_curated.npy')
        if os.path.exists(iscell_savepath):
            string = datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')
            iscell_curated = n.load(iscell_savepath)
            print("Found old curated iscell with %d of %d marked as cells" %
                (iscell_curated[:, 0].sum(), iscell_curated.shape[0]))
            backup_path = os.path.join(savedir, 'iscell_curated_old_%s.npy' % string)
            print("Saving old iscell_curated to backup path %s" % backup_path)
            n.save(backup_path, iscell_curated)
        else:
            iscell_curated = iscell.copy()
        n.save(iscell_savepath, iscell_curated)
    
    n_roi, nt = spks.shape
    ts = n.arange(nt) / outputs['fs']
    widgets = {}
    widgets['plot_widget'] = pg.PlotWidget()
    widgets['plot_widget'].addLegend()
    widgets['f_line'] = widgets['plot_widget'].plot(ts, F[0],name='F', pen='b')
    widgets['fneu_line'] = widgets['plot_widget'].plot(ts, Fneu[0], name='Fneu', pen='r')
    widgets['spks_line'] = widgets['plot_widget'].plot(ts, spks[0], name='Deconv', pen='w')
    widgets['dock_widget'] = v.window.add_dock_widget(widgets['plot_widget'],
                                            name='traces', area='bottom')
    cell_layer = layers['cvol_layer']
    not_cell_layer = layers['nvol_layer']
    
    if add_curation:
        update_vols(outputs['stats'], outputs['vmap'].shape, iscell_curated, 
        layers, update_layers=True)

    if add_sliders:
        iscell_slider_path = os.path.join(savedir, 'iscell_curated_slider.npy')
        n.save(iscell_slider_path, iscell_curated)
        if os.path.exists(iscell_slider_path):
            string = datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')
            iscell_curated_slider_old = n.load(iscell_slider_path)
            print("Found old curated + slider-ed iscell with %d of %d marked as cells" %
                  (iscell_curated_slider_old[:, 0].sum(), iscell_curated.shape[0]))
            backup_path = os.path.join(savedir, 'iscell_curated_slider_old_%s.npy' % string)
            print("Saving old iscell_curated to backup path %s" % backup_path)
            n.save(backup_path, iscell_curated_slider_old)
            n.save(iscell_slider_path, iscell_curated_slider_old)
        sliders, values, ranges = add_curation_sliders(v, iscell_savepath, outputs, layers,
                             iscell_save_path=iscell_slider_path, filters=filters,)

    def get_traces(cidx):
        return ts, spks[cidx], F[cidx], Fneu[cidx]
    def update_plot(widg_dict, label_idx):
        if label_idx == -1:
            ts = [0]; ss = [0]; fx = [0]; fn = [0]
        else:
            ts, ss, fx, fn = get_traces(label_idx - 1)
        widg_dict['f_line'].setData(ts, fx)
        widg_dict['fneu_line'].setData(ts, fn)
        widg_dict['spks_line'].setData(ts, ss)
    
    @cell_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        handle_event(layer,event)
    @not_cell_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        handle_event(layer,event)
    def handle_event(layer, event):
        coords = n.array(layer.world_to_data(event.position)).astype(int)
        value_c = layers['clabel'][coords[0], coords[1], coords[2]]
        value_nc = layers['nclabel'][coords[0], coords[1], coords[2]]
        value = max(value_c, value_nc)
        if event.button == 1:
            update_plot(widgets,value-1)
        elif add_curation and event.button == 2:
            if value > 0:
                iscell_curated[value-1] = 1-iscell_curated[value-1]
                n.save(iscell_savepath, iscell_curated)
                print("Updating cell %d" % (value-1))
                if add_sliders:
                    slider_callback(v, sliders, ranges, iscell_savepath, iscell_slider_path, values, outputs, layers)
                else: 
                    update_vols(outputs['stats'], outputs['vmap'].shape, iscell_curated, 
                            layers, update_layers=True)
                
def update_vols(stats, shape, iscell, layers, cmap='Set3', 
                lam_max=0.3, scale=(15,4,4), update_layers=True):
    cell_id_vol, cell_rgb_vol = make_label_vols(stats, shape, iscell=iscell,
                                               cmap=cmap, lam_max=lam_max)
    noncell_id_vol, noncell_rgb_vol = make_label_vols(stats, shape, iscell=1-iscell,
                                                   cmap=cmap, lam_max=lam_max)
    if update_layers:
        print(layers['cvol_layer'])
        layers['cvol_layer'].data = cell_rgb_vol;layers['cvol_layer'].refresh()
        layers['nvol_layer'].data = noncell_rgb_vol;layers['nvol_layer'].refresh()
        layers['clabel'] = cell_id_vol
        layers['nclabel'] = noncell_id_vol

    return cell_id_vol, noncell_id_vol, cell_rgb_vol, noncell_rgb_vol


def add_curation_sliders(v, iscell_path, outputs, layers, iscell_save_path = None,filters = None,):
    if filters is None: 
        peak_vals = [stat['peak_val'] for stat in outputs['stats']]
        filters = [('vmap_peak', (n.min(peak_vals),n.percentile(peak_vals, 80)), 'peak_val', lambda x : x),
                   ('npix', (0,250), 'lam', lambda x : len(x))]
        print(filters[0])

    # iscell = n.load(iscell_path)
    if iscell_save_path is None: 
        iscell_save_path = iscell_path

    ranges = [filt[1] for filt in filters]
    sliders, values = add_filters(v, filters, outputs)
    for slider in sliders:
     slider.slider.sliderReleased.connect(lambda x=0 : slider_callback(v, sliders, 
                ranges, iscell_path, iscell_save_path, values, outputs, layers))
    return sliders, values, ranges

def slider_callback(v, sliders, ranges, iscell_path, iscell_save_path, values, outputs, layers):
    iscell = n.load(iscell_path)
    iscell_out = iscell.copy()
    for i,slider in enumerate(sliders):
        rng = list(slider.slider.value())
        if rng[0] == ranges[i][0]: rng[0] = values[i].min()
        if rng[1] == ranges[i][1]: rng[1] = values[i].max()
        valid = get_valid_cells(values[i], rng)
        iscell_out[:,0] = n.logical_and(iscell_out[:,0], valid)
        iscell_out[:,1] = iscell_out[:,0]
    n.save(iscell_save_path, iscell_out)
    print("%d, %d cells valid" % (iscell_out[:,0].sum(),iscell_out[:,1].sum() ))
    update_vols(outputs['stats'], outputs['vmap'].shape, iscell_out, 
                layers, update_layers=True)

def add_slider(v, name, srange=(0, 1), callback=None):
    slider = QRangeSliderPopup()
    slider.slider.setRange(*srange)
    slider.slider.setSingleStep(0.1)
    slider.slider.setValue(srange)
    slider.slider._slider.sliderReleased.connect(
        slider.slider.sliderReleased.emit)
    if callback is not None:
        slider.slider.sliderReleased.connect(callback)
    widget = v.window.add_dock_widget(slider, name=name,
                                      area='left', add_vertical_stretch=False)
    return widget, slider


def add_filters(v, filters, outputs, callback=None):
    sliders = []
    all_values = []
    for filt in filters:
        values = n.array([filt[3](stat[filt[2]]) for stat in outputs['stats']])
        widget, slider = add_slider(
            v, filt[0], srange=filt[1], callback=callback)
        sliders.append(slider)
        all_values.append(values)
    return sliders, all_values


def get_valid_cells(prop, limits):
    good_cells = n.logical_and(prop > limits[0], prop < limits[1])
    return good_cells




# OLD STUFF BELOW HERE


def load_outputs_old(dir, files = ['stats.npy', 'F.npy', 'Fneu.npy', 'spks.npy', 'info.npy', 'iscell_filtered.npy', 'iscell_extracted.npy', 'iscell.npy', 'vmap.npy', 'im3d.npy', 'vmap_patch.npy'], return_namespace=False, additional_outputs = {}, regen_iscell=False):
    outputs = {'dir' : dir}
    for filename in files:
        if filename in os.listdir(dir):
            tag = filename.split('.')[0]
            outputs[tag] = n.load(os.path.join(dir, filename),allow_pickle=True)
            if tag == 'info':
                outputs[tag] = outputs[tag].item()
                outputs['vmap'] = outputs['info']['vmap']
                outputs['fs'] = outputs['info']['all_params']['fs']
                if 'vmap_patch' in outputs['info'].keys():
                    outputs['vmap_patch'] = outputs['info']['vmap_patch']
    if 'iscell' not in outputs.keys() or regen_iscell:
        iscell = n.ones((len(outputs['stats']), 2), dtype=int)
        n.save(os.path.join(dir, 'iscell.npy'), iscell)
        outputs['iscell'] = iscell
    if additional_outputs is not None:
        outputs.update(additional_outputs)
    if 'F' in outputs.keys(): 
        outputs['ts'] = n.arange(outputs['F'].shape[-1]) / outputs['fs']
    return outputs

def normalize_planes(im3d, axes = (1,2), normalize_by='mean'):
    imnorm = im3d - im3d.min(axis=axes, keepdims=True)
    if normalize_by == 'mean':
        imnorm /= imnorm.mean(axis=axes, keepdims=True)
    else:
        assert False
    return imnorm

def make_cell_label_vol(stats, iscell, shape, lam_thresh = 0.5, use_patch_coords = False,labels=None, dtype=int, bg_nan=False, max_lam_only = False):
    cell_labels = n.zeros(shape, dtype=dtype)
    if bg_nan:
        cell_labels[:] = n.nan
    n_cells = iscell.sum()
    label_to_idx = {}

    iscell_idxs = n.where(iscell)[0]

    for i, cell_idx in enumerate(iscell_idxs):
        if cell_idx >= len(stats):
            print("Warning - iscell.npy has more cells than stats.npy")
            break
        stat = stats[cell_idx]
        if stat is None: continue

        lam = stat['lam'][stat['lam'] > 0]
        if len(lam) < 1: continue
        lam = stat['lam'] / lam.sum()
        npix = len(lam)
        if max_lam_only: 
            valid_pix = lam == lam.max()
        else: 
            valid_pix = lam > lam_thresh / npix
        if use_patch_coords: coords = stat['coords_patch']
        else: coords = stat['coords']
        zs, ys, xs = [xx[valid_pix] for xx in coords]
        if labels is None:
            cell_labels[zs, ys, xs] = cell_idx + 1
        else:
            # print(cell_idx, xs, labels[i])
            cell_labels[zs, ys, xs] = labels[i]
            
    return cell_labels


def simple_filter_cells(stats, max_w = 30):
    plot_cell_idxs = []
    for cell_idx in range(len(stats)):
        stat = stats[cell_idx]
        lams = stat['lam']
        if n.isnan(lams).sum() > 0:
            continue
        zs, ys, xs = stat['coords']
        if ys.max() - ys.min() > max_w or xs.max() - xs.min() > max_w: 
            # print(cell_idx)
            continue
        plot_cell_idxs.append(cell_idx)
    return n.array(plot_cell_idxs)
        
def update_iscell(iscell, dir):
    iscell_path = os.path.join(dir, 'iscell.npy')
    n.save(iscell_path, iscell)


def create_napari_ui(outputs, lam_thresh=0.3, title='3D Viewer', use_patch_coords=False, scale=(15,4,4), theme='dark', extra_cells=None, extra_cells_names=None, vmap_limits=None,
                     extra_images = None, extra_images_names = None, cell_label_name='cells', vmap_name='corr map', use_filtered_iscell=True, v=None):
    if use_patch_coords:
        vmap = outputs['vmap_patch']
    else: 
        vmap = outputs['vmap']
    if use_filtered_iscell and 'iscell_filtered' in outputs.keys():
        iscell = outputs['iscell_filtered']
    else:
        iscell = outputs['iscell']
    if len(iscell.shape) > 1:
        iscell = iscell[:,0]
    cell_labels = make_cell_label_vol(outputs['stats'], iscell, vmap.shape,
                                         lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
    not_cell_labels = make_cell_label_vol(outputs['stats'], 1-iscell, vmap.shape,
                                             lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
    if v is None:
        v = napari.view_image(
        vmap, title=title, name=vmap_name, opacity=1.0, scale=scale, contrast_limits=vmap_limits)
    else:
        v.add_image(vmap, title=title, name=vmap_name, opacity=1.0, scale=scale, contrast_limits=vmap_limits)

    if extra_images is not None:
        for i, extra_image in enumerate(extra_images):
            v.add_image(
                extra_image, name=extra_images_names[i], opacity=1.0, scale=scale)

    if 'im3d' in outputs.keys():
        v.add_image(outputs['im3d'], name='Image', scale=scale)
    cell_layer = v.add_labels(cell_labels, name=cell_label_name, opacity=0.5, scale=scale)

    if extra_cells is not None:
        for i, extra_cell in enumerate(extra_cells):
            extra_cell_labels = make_cell_label_vol(extra_cell, n.ones(len(extra_cell)), vmap.shape,
                                                    lam_thresh=lam_thresh, use_patch_coords=use_patch_coords)
            v.add_labels(extra_cell_labels,
                         name=extra_cells_names[i], scale=scale, opacity=0.5)

    not_cell_layer = v.add_labels(
        not_cell_labels, name='not-' +cell_label_name, opacity=0.5, scale=scale)
    
    if 'F' in outputs.keys():
        if outputs['F'].shape[0] != len(iscell):
            assert outputs['F'].shape[0] == iscell.sum()
            trace_idxs = n.cumsum(iscell) - 1
        else:
            trace_idxs = n.arange(len(iscell))

    v.theme = theme
    widg_dict = {}
    widg_dict['plot_widget'] = pg.PlotWidget()
    widg_dict['plot_widget'].addLegend()
    widg_dict['f_line'] = widg_dict['plot_widget'].plot(
        [0], [0], pen='b', name='F')
    widg_dict['fneu_line'] = widg_dict['plot_widget'].plot(
        [0], [0], pen='r', name='Npil')
    widg_dict['spks_line'] = widg_dict['plot_widget'].plot(
        [0], [0], pen='w', name='Deconv')
    widg_dict['dock_widget'] = v.window.add_dock_widget(
        widg_dict['plot_widget'], name='activity', area='bottom')



    def get_traces(cell_idx):
        trace_idx = trace_idxs[cell_idx]
        fx = outputs['F'][trace_idx]
        fn = outputs['Fneu'][trace_idx]
        ss = outputs['spks'][trace_idx]
        return outputs['ts'], fx, fn, ss

    def update_plot(widg_dict, cell_idx):
        ts, fx, fn, ss = get_traces(cell_idx)
        widg_dict['f_line'].setData(ts, fx)
        widg_dict['fneu_line'].setData(ts, fn)
        widg_dict['spks_line'].setData(ts, ss)

    @cell_layer.mouse_drag_callbacks.append
    def on_click(cell_labels, event):
        value = cell_labels.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True)
        print(value)
        if value is not None and value > 0:
            cell_idx = value - 1
            if event.button == 1:
                update_plot(widg_dict, cell_idx)
            # if event.button == 2:
            #     mark_cell(
            #         cell_idx, 0, outputs['iscell'], cell_layer, not_cell_layer)

    @not_cell_layer.mouse_drag_callbacks.append
    def on_click(not_cell_labels, event):
        value = not_cell_labels.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True)
        print('Not cell,', value)
        if value is not None and value > 0:
            cell_idx = value - 1
            if event.button == 1:
                update_plot(widg_dict, cell_idx)
            # if event.button == 2:
            #     mark_cell(
            #         cell_idx, 1, outputs['iscell'], cell_layer, not_cell_layer)

    return v






def mark_cell(cell_idx, mark_as, iscell=None, napari_cell_layer=None, napari_not_cell_layer=None, refresh=True):
    napari_idx = cell_idx + 1
    print("Marking cell %d (napari %d) as %d" %
          (cell_idx, napari_idx, int(mark_as)))
    if mark_as:
        cell_layer_val = napari_idx
        not_cell_layer_val = 0
        coords = napari_not_cell_layer.data == napari_idx
    else:
        cell_layer_val = 0
        not_cell_layer_val = napari_idx
        coords = napari_cell_layer.data == napari_idx

    if napari_cell_layer is not None:
        napari_cell_layer.data[coords] = cell_layer_val
        if refresh:
            napari_cell_layer.refresh()
    if napari_not_cell_layer is not None:
        napari_not_cell_layer.data[coords] = not_cell_layer_val
        if refresh:
            napari_not_cell_layer.refresh()
    if iscell is not None:
        iscell[cell_idx] = int(mark_as)
