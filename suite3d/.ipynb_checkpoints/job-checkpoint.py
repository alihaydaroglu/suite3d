try:
    import tifffile
except: 
    print("No tifffile")
import datetime
import os
import copy
import sys
import numpy as n
import itertools
from multiprocessing import Pool
import shutil
from matplotlib import pyplot as plt
from skimage.io import imread
try: 
    import psutil
except: print("No psutil")

from suite2p.extraction import dcnv

from . import init_pass
from . import utils 
from . import lbmio
from .iter_step import register_dataset, fuse_and_save_reg_file, calculate_corrmap, calculate_corrmap_from_svd, register_dataset_gpu
from . import extension as ext
from .default_params import get_default_params
from . import svd_utils as svu
from . import ui

class Job:
    def __init__(self, root_dir, job_id, params=None, tifs=None, overwrite=False, verbosity=10, create=True, params_path=None):
        """Create a Job object that is a wrapper to manage files, current state, log etc.

        Args:
            root_dir (str): Root directory in which job directory will be created
            job_id (str): Unique name for the job directory
            params (dict): Job parameters (see examples)
            tifs (list) : list of full paths to tif files to be used
            overwrite (bool, optional): If False, will throw error if job_dir exists. Defaults to False.
            verbosity (int, optional): Verbosity level. 0: critical only, 1: info, 2: debug. Defaults to 1.
        """

        self.verbosity = verbosity
        self.job_id = job_id


        if create:   
            self.init_job_dir(root_dir, job_id, exist_ok=overwrite)
            def_params = get_default_params()
            self.log("Loading default params")
            for k,v in params.items():
                assert k in def_params.keys(), "%s not a valid parameter" % k
                self.log("Updating param %s" % (str(k)), 2)
                def_params[k] = v
            self.params = def_params
            assert tifs is not None, 'Must provide tiff files'
            self.params['tifs'] = tifs
            self.tifs = tifs
            self.save_params()
        else:
            self.job_dir = os.path.join(root_dir,'s3d-%s' % job_id)
            self.load_dirs()
            self.load_params(params_path=params_path)
            self.tifs = self.params.get('tifs', [])


    def log(self, string='', level=1, logfile=True, log_mem_usage=False):
        """Print messages based on current verbosity level

        Args:
            string (str): String to be printed
            level (int, optional): Level equal or below self.verbosity will be printed. Defaults to 1.
        """
        if log_mem_usage:
            
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()

            vm_avail = vm.available
            vm_unavail = vm.total - vm_avail

            total = sm.used + vm_unavail
            string = "{:<20}".format(string)
            string += ("Total Used: %07.3f GB, Virtual Available: %07.3f GB, Virtual Used: %07.3f GB, Swap Used: %07.3f GB" %
                       ((total/(1024**3), vm_avail/(1024**3),  vm_unavail / (1024**3), sm.used/(1024**3))))
            
        if level <= self.verbosity:
            # print('xxx')
            print(("   " * level) + string)
        if logfile:
            logfile = os.path.join(self.job_dir, 'log.txt')
            self.logfile = logfile
            with open(logfile, 'a+') as f:
                datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = '\n[%s][%02d] ' % (datetime_string, level)
                f.write(header + '   ' * level + string)

    def save_params(self, new_params=None, copy_dir = None, params=None, update_main_params=True):
        """Update saved params in job_dir/params.npy
        """
        if params is None:
            params = self.params
        if new_params is not None:
            params.update(new_params)
        if copy_dir is not None:
            params_path = os.path.join(self.dirs[copy_dir], 'params.npy')
            n.save(params_path, params)
            self.log("Saved a copy of params at %s" % self.dirs[copy_dir])
        if update_main_params:
            n.save(os.path.join(self.dirs['job_dir'], 'params.npy'), params)
        self.log("Updated main params file")

    def load_params(self, dir = None, params_path = None):
        if params_path is None:
            if dir is None:
                dir = 'job_dir'
            params_path = os.path.join(self.dirs[dir], 'params.npy')
        self.params = n.load(params_path, allow_pickle=True).item()
        self.log("Found and loaded params from %s" % params_path)
        return self.params

    
    def show_summary_plots(self):
        summary = self.load_summary()
        f1 = plt.figure(figsize=(8,4), dpi=200)
        plt.plot(summary['plane_shifts'])
        plt.xlabel("Plane")
        plt.ylabel("# pixels of shift")
        plt.title("LBM shift between planes")
        plt.ylim(-100,100)

        crosstalk_dir = os.path.join(self.dirs['summary'], 'crosstalk_plots')
        gamma_fit_img = os.path.join(crosstalk_dir, 'gamma_fit.png')
        plane_fits_img = os.path.join(crosstalk_dir, 'plane_fits.png')

        if os.path.isfile(plane_fits_img):
            im = imread(plane_fits_img)
            f2,ax = plt.subplots(figsize=(im.shape[0] // 200, im.shape[1] // 200), dpi=400);
            ax.imshow(im); ax.set_axis_off()
        if os.path.isfile(gamma_fit_img):
            im = imread(gamma_fit_img)
            f3,ax = plt.subplots(figsize=(im.shape[0] // 200, im.shape[1] // 200), dpi=150);
            ax.imshow(im); ax.set_axis_off()

        
        if 'fuse_shifts' in summary.keys() and 'fuse_ccs' in summary.keys():
            utils.plot_fuse_shifts(summary['fuse_shifts'], summary['fuse_ccs'])

                

    def load_summary(self):
        summary_path = os.path.join(self.dirs['summary'], 'summary.npy')
        summary = n.load(summary_path,  allow_pickle=True).item()
        return summary

    def make_svd_dirs(self, n_blocks=None):
        self.make_new_dir('svd')
        self.make_new_dir('blocks', 'svd', dir_tag='svd_blocks')
        block_dirs = []
        if n_blocks is not None:
            for i in range(n_blocks):
                block_dirs.append(self.make_new_dir('%03d' % i, 'svd_blocks', dir_tag = 'svd_blocks_%03d' % i))
            return block_dirs

    def make_stack_dirs(self, n_stacks):
        stack_dirs = []
        self.make_new_dir('stacks', 'svd', dir_tag='svd_stacks')
        for i in range(n_stacks):
            stack_dirs.append(self.make_new_dir('%03d' % i, 'svd_stacks', dir_tag = 'svd_stacks_%03d' % i))
        return stack_dirs

    def make_extension_dir(self, extension_root, extension_name='ext'):
        extension_dir = os.path.join(extension_root, 's3d-extension-%s' % self.job_id)
        if extension_name in self.dirs.keys():
            self.log("Extension dir %s already exists at %s" % (extension_name, self.dirs[extension_name]))
            return self.dirs[extension_name]
        os.makedirs(extension_dir)
        self.log("Made new extension dir at %s" % extension_dir)
        self.dirs[extension_name] = extension_dir
        self.save_dirs()
        return extension_dir

    def save_dirs(self, name='dirs', dirs=None):
        if dirs is None: dirs = self.dirs
        n.save(os.path.join(self.job_dir, '%s.npy' % name), dirs)

    def load_dirs(self):
        self.dirs = n.load(os.path.join(self.job_dir , 'dirs.npy'),allow_pickle=True).item()

    def update_root_path(self, new_root):
        old_dirs = copy.deepcopy(self.dirs)
        root_len = self.dirs['summary'].find('s3d-' + self.job_id)
        self.log("Replacing %s with %s" % (self.dirs['summary'][:root_len], new_root))
        for k,v in self.dirs.items():
            self.dirs[k] = os.path.join(new_root, v[root_len:])
        self.save_dirs()
        self.save_dirs('old_dirs_%d' % n.random.randint(1,1e9), old_dirs)


    def make_new_dir(self, dir_name, parent_dir_name = None, exist_ok=True, dir_tag = None):
        if parent_dir_name is None:
            parent_dir = self.job_dir
        else: 
            parent_dir = self.dirs[parent_dir_name]
        if dir_tag is None:
            dir_tag = dir_name
        
        dir_path = os.path.join(parent_dir, dir_name)
        if os.path.exists(dir_path):
            self.log("Found dir %s with tag %s" % (dir_path, dir_tag), 2)
        else:
            os.makedirs(dir_path, exist_ok = exist_ok)
            self.log("Created dir %s with tag %s" % (dir_path, dir_tag))
        self.dirs[dir_tag] = dir_path
        n.save(os.path.join(self.job_dir, 'dirs.npy'), self.dirs)
        return dir_path

    def init_job_dir(self, root_dir, job_id, exist_ok=False):
        """Create a job directory and nested dirs

        Args:
            root_dir (str): Root directory to create job_dir in
            job_id (str): Unique name for job
            exist_ok (bool, optional): If False, throws error if job_dir exists. Defaults to False.
        """

        job_dir = os.path.join(root_dir,'s3d-%s' % job_id)
        self.job_dir = job_dir
        if os.path.isdir(job_dir):
            self.log("Job directory %s already exists" % job_dir, 0)
            assert exist_ok, "Set create=False to load existing job, or set overwrite=True to overwrite existing job"
        else:
            os.makedirs(job_dir, exist_ok=True)

        self.log("Loading job directory for %s in %s" %
                    (job_id, root_dir), 0)
        if 'dirs.npy' in os.listdir(job_dir):
            self.log("Loading dirs ")
            self.dirs = n.load(os.path.join(job_dir, 'dirs.npy'),allow_pickle=True).item()
        else:
            self.dirs = {'job_dir' : job_dir}

        if job_dir not in self.dirs.keys():
            self.dirs['job_dir'] = job_dir

        for dir_name in ['registered_data', 'summary', 'iters']:
            dir_key = dir_name
            if dir_key not in self.dirs.keys():
                new_dir = os.path.join(job_dir, dir_name) 
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir, exist_ok=True)
                    self.log("Created dir %s" % new_dir,2)
                # else:
                    # 
                    # self.log("Found dir %s" % new_dir,2)
                self.dirs[dir_key] = new_dir
                
            else:
                self.log("Found dir %s" % dir_name,2)
        n.save(os.path.join(job_dir, 'dirs.npy'), self.dirs)

    def run_init_pass(self):
        self.save_params(copy_dir='summary')
        self.log("Launching initial pass", 0)
        init_pass.run_init_pass(self)
    def copy_init_pass(self,summary_old_job):
        n.save(os.path.join(self.dirs['summary'],
               'summary.npy'), summary_old_job)
    
    def register(self, tifs=None, start_batch_idx = 0, params=None, summary=None):
        if params is None:
            params = self.params
        self.save_params(params=params, copy_dir='registered_data')
        if summary is None:
            summary = self.load_summary()
        n.save(os.path.join(self.dirs['registered_data'], 'summary.npy'), summary)
        if tifs is None:
            tifs = self.tifs
        register_dataset(tifs, params, self.dirs, summary, self.log, start_batch_idx = start_batch_idx)


    def register_gpu(self, tifs=None):
        params = self.params
        summary = self.load_summary()
        save_dir = self.make_new_dir('registered_fused_data')
        if tifs is None:
            tifs = self.tifs
        register_dataset_gpu(tifs, params, self.dirs, summary, self.log)
        
        



    def calculate_corr_map(self, mov=None, save=True, return_mov_filt=False, crop=None, svd_info=None, iter_limit=None, 
                            parent_dir = None, update_main_params=True, svs = None, us=None):
        self.save_params(copy_dir=parent_dir, update_main_params=update_main_params)
        mov_sub_dir_tag = 'mov_sub'
        iter_dir_tag = 'iters'
        if parent_dir is not None: 
            mov_sub_dir_tag = parent_dir + '-' + mov_sub_dir_tag
            iter_dir_tag = parent_dir + '-iters'
            iter_dir = self.make_new_dir('iters', parent_dir_name=parent_dir, dir_tag=iter_dir_tag)
        mov_sub_dir = self.make_new_dir('mov_sub', parent_dir_name=parent_dir, dir_tag=mov_sub_dir_tag)
        n.save(os.path.join(mov_sub_dir, 'params.npy'), self.params)
        self.log("Saving mov_sub to %s" % mov_sub_dir)
        if svd_info is not None:
            mov = svd_info
            self.log("Using SVD shortcut, loading entire V matrix to memory")
            self.log("WARNING: if you encounter very large RAM usage during this run, use mov=svd_info instead of svd_info=svd_info. If it persists, reduce your batchsizes")
            out = calculate_corrmap_from_svd(svd_info, params=self.params, log_cb=self.log, iter_limit=iter_limit, svs=svs, us=us, dirs = self.dirs, iter_dir_tag=iter_dir_tag, mov_sub_dir_tag=mov_sub_dir_tag)
        else:
            if mov is None:
                mov = self.get_registered_movie('registered_fused_data', 'fused')
            if crop is not None and svd_info is None:
                assert svd_info is None, 'cant crop with svd - easy fix'
                self.params['detection_crop'] = crop
                self.save_params(copy_dir='mov_sub', update_main_params=False)
                mov = mov[crop[0][0]:crop[0][1], :, crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
                self.log("Cropped movie to shape: %s" % str(mov.shape))
            vmap, mean_img, max_img =  calculate_corrmap(mov, self.params, self.dirs, self.log, return_mov_filt=return_mov_filt, save=save,
                                    iter_limit=iter_limit, iter_dir_tag=iter_dir_tag, mov_sub_dir_tag=mov_sub_dir_tag)
        
        return (vmap, mean_img, max_img), mov_sub_dir, self.dirs[iter_dir_tag]

    def patch_and_detect(self, corrmap_dir_tag='', do_patch_idxs=None, compute_npil_masks=False, ts=(), combined_name='combined'):
        connector = '-' if len(corrmap_dir_tag) > 0 else ''
        mov_sub = self.get_registered_movie(corrmap_dir_tag + connector + 'mov_sub', 'mov', axis=0)
        vmap = self.load_iter_results(-1, dir_tag=corrmap_dir_tag + connector + 'iters')['vmap2'] ** 0.5
        patch_size_xy = self.params['patch_size_xy']
        patch_overlap_xy = self.params['patch_overlap_xy']
        nt,nz,ny,nx = mov_sub.shape

        patches, grid_shape = svu.make_blocks((nz, ny,nx), (nz,) + patch_size_xy, (0,) + patch_overlap_xy)
        patches_vmap, __ = svu.make_blocks((nz, ny,nx), (nz,) + patch_size_xy, (0,) + patch_overlap_xy,
                                      nonoverlapping_mask=True)
        n_patches = patches.shape[1]
        patch_idxs = []

        if do_patch_idxs is None: do_patch_idxs = range(n_patches)
        for patch_idx in do_patch_idxs:
            self.log("Detecting from patch: %d/%d" % (patch_idx, len(do_patch_idxs)))
            patch_dir, patch_info, stats = self.detect_cells_from_patch(patch_idx, patches[:,patch_idx],
                                                                patches_vmap[:,patch_idx], 
                                                                        parent_dir=corrmap_dir_tag, compute_npil_masks=compute_npil_masks,
                                                                ts = self.params.get('detection_time_crop', (None,None)))
            patch_idxs.append(patch_idx)

        combined_dir = self.combine_patches(patch_idxs, parent_dir_tag = corrmap_dir_tag + connector + 'detection', combined_name=combined_name)
        
        return combined_dir
    
    def compute_npil_masks(self, stats_dir, corrmap_dir_tag=''):
        connector = '-' if len(corrmap_dir_tag) > 0 else ''
        mov_sub = self.get_registered_movie(corrmap_dir_tag + connector + 'mov_sub', 'mov', axis=0)
        nt,nz,ny,nx = mov_sub.shape
        stats = n.load(os.path.join(stats_dir, 'stats.npy'),allow_pickle=True)
        stats = ext.compute_npil_masks_mp(stats, (nz,ny,nx), n_proc=self.params['n_proc_corr'])
        n.save(os.path.join(stats_dir, 'stats.npy'), stats)
        return stats_dir


    def detect_cells_from_patch(self, patch_idx = 0, coords = None,vmap_coords = None,
                                vmap=None, mov=None, compute_npil_masks=False,n_proc = 8, ts=(None, None),
                                parent_dir = None, extra_tag = None):
        # print('test')
        self.save_params()
        det_dir_tag = 'detection'
        iter_dir_tag = 'iters'
        mov_dir_tag = 'mov_sub'
        patch_str = 'patch-%04d' % patch_idx
        patch_dir_tag = patch_str
        if parent_dir is not None:
            dir_prefix = parent_dir
            connector = '-' if len(parent_dir) > 0 else ''
            if extra_tag is not None: dir_prefix = dir_prefix + connector + extra_tag
            det_dir_tag = parent_dir + connector + det_dir_tag
            iter_dir_tag = parent_dir + connector + iter_dir_tag
            mov_dir_tag = parent_dir + connector + mov_dir_tag
            patch_dir_tag = parent_dir + connector + patch_dir_tag
        detection_dir = self.make_new_dir(det_dir_tag)
        patch_dir = self.make_new_dir(patch_str, parent_dir_name= det_dir_tag, dir_tag = patch_dir_tag)
        n.save(os.path.join(patch_dir, 'params.npy'), self.params)
        stats_path = os.path.join(patch_dir, 'stats.npy')
        info_path = os.path.join(patch_dir, 'info.npy')
        zs, ys, xs = coords
        if vmap_coords is None:
            vmap_coords = coords
        vzs, vys, vxs = vmap_coords
        self.log("Running cell detection on patch %04d at %s, max %d iters" % (patch_idx, patch_dir, self.params['max_iter']))
        self.log("Patch bounds are %s, %s, %s" % (str(zs), str(ys), str(xs)))
        self.log("Cell center bounds are %s, %s, %s" % (str(vzs), str(vys), str(vxs)))
        self.log("Time bounds are %s" % (str(ts)))

        patch_info = {'zs' : zs, 'ys' : ys, 'xs' : xs, 'ts' : ts,
                      'vzs' : vzs, 'vys' : vys, 'vxs' : vxs, 'all_params' : self.params}
        if mov is None:
            print(mov_dir_tag)
            mov = self.get_registered_movie(mov_dir_tag, 'mov_sub',axis=0)
            nt, nz,ny,nx = mov.shape
            mov = mov[ts[0]:ts[1], zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]]
        else:
            __, nz, ny, nx = mov.shape

        if self.params['detection_timebin'] > 1:
            self.log("Binning movie with a factor of %.2f" % self.params['detection_timebin'])
            mov = ext.binned_mean(mov, self.params['detection_timebin'])
        try:
            self.log("Loading %.2f GB movie to memory" % (mov.nbytes/1024**3))
            mov = mov.compute()
            self.log("Loaded")
        except:
            self.log("Not a dask array")


        if vmap is None:
            iter_results = self.load_iter_results(-1, dir_tag = iter_dir_tag)
            if 'vmap' in iter_results:
                vmap = iter_results['vmap']
            else: 
                vmap = iter_results['vmap2']**0.5
            if 'mean_img' in iter_results.keys():
                patch_info['mean_img'] = iter_results['mean_img']
            if 'max_img' in iter_results.keys():
                patch_info['max_img'] = iter_results['max_img']
        if self.params['normalize_vmap']:
            vmap = ui.normalize_planes(vmap)
        
        patch_info['vmap'] = vmap.copy()
        patch_info['vmap_unmasked'] = vmap[zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]]
        vmap_patch = n.zeros_like(mov[0])
        dz = vzs[0] - zs[0]; dy = vys[0] - ys[0]; dx = vxs[0] - xs[0]
        vmap_patch[dz:dz+(vzs[1]-vzs[0]),dy:dy+(vys[1]-vys[0]),dx:dx+(vxs[1]-vxs[0])] = \
                            vmap[vzs[0]:vzs[1], vys[0]:vys[1], vxs[0]:vxs[1]]
        vmap = vmap_patch

        patch_info['vmap_patch'] = vmap.copy()
        n.save(os.path.join(patch_dir, 'vmap_patch.npy'), vmap)
        n.save(os.path.join(patch_dir, 'vmap_patch_unmasked.npy'),
               patch_info['vmap_unmasked'])


        n.save(info_path, patch_info)
        self.log("Movie shape: %s" % str(mov.shape), 2)
        self.log("Saving cell stats and info to %s" % patch_dir)
        # print(zs, ys, xs)
        if n_proc == 1:
            stats = ext.detect_cells(mov, vmap, **self.params, log=self.log, 
                             offset = (zs[0], ys[0], xs[0]), savepath=stats_path, patch_idx=patch_idx)
        else:
            stats = ext.detect_cells_mp(mov, vmap, **self.params, log=self.log,
                                        offset=(zs[0], ys[0], xs[0]), savepath=stats_path, n_proc=n_proc, patch_idx=patch_idx)
        if compute_npil_masks and len(stats) > 0:
            self.log("Computing neuropil masks")
            if n_proc == 1:
                stats = ext.compute_npil_masks(stats, (nz,ny,nx))
            else:
                self.log("Starting MP", 1)
                stats = ext.compute_npil_masks_mp(stats, (nz,ny,nx), n_proc=n_proc)
                self.log("Ended MP", 1)
        n.save(stats_path, stats)
        patch_info['stats_path']  = stats_path
        return patch_dir, patch_info, stats


    def extract_and_deconvolve(self, patch_idx=0, mov=None, batchsize_frames = 500, stats = None, offset=None, 
                               n_frames=None, stats_dir=None, iscell = None, ts=None, load_F_from_dir=False,
                               parent_dir_tag=None, save_dir = None, crop=True, mov_shape_tfirst=False):
        self.save_params()
        if stats_dir is None:
            stats_dir = self.get_patch_dir(patch_idx, parent_dir_tag=parent_dir_tag)
            stats, info = self.get_detected_cells(
                patch_idx, parent_dir_tag=parent_dir_tag)
            offset = (info['zs'],info['ys'],info['xs'])
        else:
            if stats is not None:
                if 'stats.npy' not in os.listdir(stats_dir):
                    self.log("Saving provided stats.npy to %s" % stats_dir)
                    n.save(os.path.join(stats_dir, 'stats.npy'), stats)
                else:
                    self.log("WARNING - overwriting with provided stats.npy in %s. Old one is in old_stats.npy" % stats_dir)
                    old_stats = n.load(os.path.join(stats_dir, 'stats.npy'),allow_pickle=True)
                    n.save(os.path.join(stats_dir, 'old_stats.npy'), old_stats)
                    n.save(os.path.join(stats_dir, 'stats.npy'), stats)
            else:
                stats = n.load(os.path.join(stats_dir, 'stats.npy'),allow_pickle=True)

        # return stats
        if mov is None:
            if not mov_shape_tfirst: 
                mov = self.get_registered_movie('registered_fused_data','fused')
            else:
                mov = self.get_registered_movie('registered_fused_data','fused',axis=0)
        if crop:
            cz, cy, cx = self.params['svd_crop']
            self.log("Cropping with bounds: %s" % (str(self.params['svd_crop'])))
            if mov_shape_tfirst:
                mov = mov[:,cz[0]:cz[1], cy[0]:cy[1], cx[0]:cx[1]]
            else:
                mov = mov[cz[0]:cz[1], :, cy[0]:cy[1], cx[0]:cx[1]]
        if ts is not None:
            if mov_shape_tfirst:
                mov = mov[ts[0]:ts[1]]
            else:
                mov = mov[:,ts[0]:ts[1]]
        self.log("Movie shape: %s" % (str(mov.shape)))
        if save_dir is None: save_dir = stats_dir
        if iscell is None: 
            iscell = n.ones((len(stats), 2), int)
        if type(iscell) == str:
            if iscell[-4:] != '.npy': iscell += '.npy'
            iscell = n.load(os.path.join(stats_dir, iscell))
        if len(iscell.shape) < 2: iscell = iscell[:,n.newaxis]
        print(len(stats))
        assert iscell.shape[0] == len(stats)

        valid_stats = [stat for i,stat in enumerate(stats) if iscell[i,0]]
        save_iscell = os.path.join(save_dir, 'iscell_extracted.npy')
        self.log("Extracting %d valid cells, and saving cell flags to %s" % (len(valid_stats), save_iscell))
        stats = valid_stats
        # return stats
        n.save(save_iscell, iscell)
        # print(offset, batchsize_frames, n_frames)
        # return mov, stats
        if not load_F_from_dir:
            self.log("Extracting activity")
            F_roi, F_neu = ext.extract_activity(mov, stats, batchsize_frames=batchsize_frames, offset=offset, n_frames=n_frames, intermediate_save_dir=save_dir, mov_shape_tfirst=mov_shape_tfirst)
            n.save(os.path.join(save_dir, 'F.npy'), F_roi)
            n.save(os.path.join(save_dir, 'Fneu.npy'), F_neu)
        else:
            F_roi = n.load(os.path.join(stats_dir, 'F.npy'))
            F_neu = n.load(os.path.join(stats_dir, 'Fneu.npy'))



        self.log("Deconvolving")
        F_sub = F_roi - F_neu * self.params.get('npil_coeff',0.7)
        dcnv_baseline = self.params.get('dcnv_baseline','maximin')
        dcnv_win_baseline = self.params.get('dcnv_win_baseline',60)
        dcnv_sig_baseline = self.params.get('dcnv_sig_baseline',10)
        dcnv_prctile_baseline = self.params.get('dcnv_prctile_baseline',8)
        dcnv_batchsize = self.params.get('dcnv_batchsize',3000)
        tau = self.params.get('tau',1.3)
        F_sub = dcnv.preprocess(F_sub, dcnv_baseline, dcnv_win_baseline,
                     dcnv_sig_baseline, self.params['fs'],dcnv_prctile_baseline)
        spks = dcnv.oasis(F_sub, batch_size = dcnv_batchsize, tau=tau,
                         fs=self.params['fs'])
                         
        self.log("Saving to %s" % save_dir)
        n.save(os.path.join(save_dir, 'spks.npy'), spks)
        
        return self.get_traces(patch_dir=save_dir)

    def get_patch_dir(self, patch_idx = 0, parent_dir_tag='detection'):
        if type(patch_idx) == str:
            patch_str = patch_idx
        else:
            patch_str = 'patch-%04d' % patch_idx
        patch_dir = self.make_new_dir(patch_str, parent_dir_name= parent_dir_tag, 
                                        dir_tag = parent_dir_tag + '-' + patch_str)
        return patch_dir
    def load_patch_results(self, patch_idx=0, parent_dir_tag = 'detection'):
        patch_dir = self.get_patch_dir(patch_idx, parent_dir_tag)
        stats = n.load(os.path.join(patch_dir, 'stats.npy'), allow_pickle=True)
        info = n.load(os.path.join(patch_dir, 'info.npy'), allow_pickle=True).item()
        try: 
            iscell = n.load(os.path.join(patch_dir, 'iscell.npy'))
        except FileNotFoundError:
            iscell = n.ones((len(stats), 2), dtype=int)
            n.save(os.path.join(patch_dir, 'iscell.npy'), iscell)
        return stats, info, iscell

    def combine_patches(self, patch_idxs, combined_name, info_use_idx = -1, save=True,
                        extra_stats_keys = None, parent_dir_tag = 'detection'):

        if save:
            combined_dir = self.make_new_dir(combined_name, parent_dir_name=parent_dir_tag,
                                                  dir_tag = parent_dir_tag + '-' + combined_name)
        stats = []
        iscells = []
        keep_stats_keys = ['idx','threshold', 'coords', 'lam','med',
                           'peak_val', 'npcoords', 'patch_idx', 'med_patch']
        if extra_stats_keys is not None:
            keep_stats_keys += extra_stats_keys

        for patch_idx in patch_idxs:
            stats_patch, info_patch, iscell = self.load_patch_results(patch_idx, parent_dir_tag)
            for stat in stats_patch:
                keep_stat =  {}
                for key in keep_stats_keys:
                    if key in stat.keys():
                        keep_stat[key] = stat[key]
                stats.append(keep_stat)
            iscells.append(iscell)
            if patch_idx == patch_idxs[info_use_idx]: info = info_patch
        iscell = n.concatenate(iscells)
        # stats = n.concatenate(stats)
        self.log("Combined %d patches, %d cells" % (len(patch_idxs), len(stats)))
        if not save: 
            return stats, info, iscell
        else:
            self.log("Saving combined files to %s" % combined_dir)
            n.save(os.path.join(combined_dir, 'stats.npy'), stats)
            self.log("Saved stats", 2)
            n.save(os.path.join(combined_dir, 'iscell.npy'), iscell)
            self.log("Saved iscell", 2)
            n.save(os.path.join(combined_dir, 'info.npy'), info)
            self.log("Saved info (copied from patch) %d" % patch_idxs[info_use_idx], 2)

            return combined_dir

    def get_detected_cells(self, patch = 0, parent_dir_tag='detection'):
        patch_dir = self.get_patch_dir(patch, parent_dir_tag=parent_dir_tag)
        stats = n.load(os.path.join(patch_dir, 'stats.npy'), allow_pickle=True)
        info = n.load(os.path.join(patch_dir, 'info.npy'), allow_pickle=True).item()
        return stats, info
    def get_traces(self, patch_idx=0, parent_dir_tag='detection', patch_dir=None):
        if patch_dir is None:
            patch_dir = self.get_patch_dir(patch_idx, parent_dir_tag=parent_dir_tag)
        traces = {}
        for filename in ['F.npy', 'Fneu.npy', 'spks.npy']:
            if filename in os.listdir(patch_dir):
                traces[filename[:-4]] = n.load(os.path.join(patch_dir, filename))
        return traces
        

    def get_registered_files(self, key='registered_data', filename_filter='reg_data', sort=True):
        all_files = os.listdir(self.dirs[key])
        reg_files = [os.path.join(self.dirs[key],x) for x in all_files if x.startswith(filename_filter)]
        if sort: reg_files = sorted(reg_files)
        return reg_files
    def get_denoised_files(self):
        all_files = n.os.listdir(self.dirs['deepinterp'])
        reg_files = [os.path.join(self.dirs['deepinterp'],x) for x in all_files if x.startswith('dp')]
        return reg_files

    def get_iter_dirs(self, dir_tag = 'iters', sort=True):
        iters_dir = self.dirs[dir_tag]
        iter_dirs = [os.path.join(iters_dir, dir) for dir in os.listdir(iters_dir)]
        if sort: iter_dirs = sorted(iter_dirs)
        ret = []
        # print(iter_dirs)
        # return iter_dirs
        for dir in iter_dirs:
            # print(os.listdir(dir))
            if 'vmap.npy' in os.listdir(dir) or 'vmap2.npy' in os.listdir(dir):
                ret.append(dir)
        return ret
    
    def load_iter_results(self, iter_idx, dir_tag='iters'):
        iter_dir = self.get_iter_dirs(dir_tag=dir_tag)[iter_idx]
        self.log("Loading from %s" % iter_dir)
        res = {}
        for filename in ['vmap', 'max_img', 'mean_img', 'sum_img', 'vmap2']:
            if filename + '.npy' in os.listdir(iter_dir):
                res[filename] = n.load(os.path.join(iter_dir, filename + '.npy'))
        return res

    def fuse_registered_movie(self, files=None, save=True, n_proc=8, delete_original=False, parent_dir=None):
        n_skip = self.params['n_skip']
        if files is None:
            files = self.get_registered_files()
        __, xs = lbmio.load_and_stitch_full_tif_mp(
            self.tifs[0], channels=n.arange(1), get_roi_start_pix=True)
        centers = n.sort(xs)[1:]
        shift_xs = n.round(self.load_summary()[
                           'plane_shifts'][:, 1]).astype(int)
        if save:
            reg_fused_dir = self.make_new_dir('registered_fused_data', parent_dir_name=parent_dir)
        else:
            reg_fused_dir = ''
        if save:
            self.log("Saving to %s" % reg_fused_dir)
            self.save_params(copy_dir='registered_fused_data')

        crop = self.params.get("fuse_crop", None)
        if crop is not None:
            self.log("Cropping: %s" % str(crop))
        # if you get an assertion error here with save=False in _get_more_data, assert left > 0
        # congratulations, you have run into a bug in Python itself! 
        # https://bugs.python.org/issue34563, https://stackoverflow.com/questions/47692566/
        # the files are too big! 
        if n_proc > 1:
            with Pool(n_proc) as p:
                fused_files = p.starmap(fuse_and_save_reg_file, [(
                    file, reg_fused_dir, centers,  shift_xs, n_skip, crop, None, save, delete_original) for file in files])
        else:
            self.log("Single processor")
            fused_files = [fuse_and_save_reg_file(file, reg_fused_dir, centers,  shift_xs, n_skip, None, None, save, delete_original) for file in files]
        if not save:
            # return fused_files
            fused_files = n.concatenate(fused_files, axis=1)
        return fused_files

    def svd_decompose_movie(self, svd_dir_tag, run_svd=True, end_batch=None, mov=None,
                            mov_shape_tfirst=False):
        svd_dir = self.dirs[svd_dir_tag]
        self.save_params(copy_dir=svd_dir_tag)


        if mov is None:
            if not mov_shape_tfirst: 
                mov = self.get_registered_movie('registered_fused_data','fused')
            else:
                mov = self.get_registered_movie('registered_fused_data', 'fused', axis=0)
            self.log("Loaded mov of size %s" % str(mov.shape))
        if self.params.get('svd_crop', None) is not None:
            crop = self.params['svd_crop']
            if not mov_shape_tfirst:
                mov = mov[crop[0][0]:crop[0][1], :,crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
            else:
                mov = mov[:, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]]
            self.log("Cropped to size %s" % str(mov.shape))
        if self.params.get('svd_time_crop', None) is not None:
            svd_time_crop = self.params.get('svd_time_crop', None)
            if not mov_shape_tfirst: 
                mov = mov[:,svd_time_crop[0]:svd_time_crop[1]]
            else:
                mov = mov[svd_time_crop[0]:svd_time_crop[1]]
            self.log("Time-cropped to size %s" % str(mov.shape))
        
        if self.params.get('svd_pix_chunk') is None:
            self.params['svd_pix_chunk'] = n.product(self.params['svd_block_shape']) // 2
        if self.params.get('svd_time_chunk') is None:
            self.params['svd_save_time_chunk'] = 4000
        if self.params.get('svd_save_time_chunk') is None:
            self.params['svd_save_comp_chunk'] = 400
        if self.params.get('svd_save_comp_chunk') is None:
            self.params['svd_comp_chunk'] = 100
        if self.params.get('n_svd_blocks_per_batch') is None:
            self.params['n_svd_blocks_per_batch'] = 16
        self.save_params(copy_dir=svd_dir_tag)
        # return
        svd_info = svu.block_and_svd(mov, n_comp = self.params['n_svd_comp'], 
                               block_shape = self.params['svd_block_shape'],
                               block_overlaps = self.params['svd_block_overlaps'],
                               pix_chunk = self.params['svd_pix_chunk'],
                               t_chunk = self.params['svd_time_chunk'],
                               t_save_chunk = self.params['svd_save_time_chunk'],
                               comp_chunk = self.params['svd_save_comp_chunk'],
                               n_svd_blocks_per_batch = self.params['n_svd_blocks_per_batch'],
                               log_cb = self.log, end_batch=end_batch, flip_shape=mov_shape_tfirst,
                               svd_dir = svd_dir, run_svd=run_svd)
        return svd_info


    def get_subtracted_movie(self):
        mov_sub_paths = []
        for d in self.get_iter_dirs():
            if 'mov_sub.npy' in os.listdir(d):
                mov_sub_paths.append(os.path.join(d, 'mov_sub.npy'))

        mov_sub = utils.npy_to_dask(mov_sub_paths, axis=0)
        return mov_sub

    def get_registered_movie(self, key='registered_data', filename_filter='reg_data', axis=1):
            paths = self.get_registered_files(key, filename_filter)
            mov_reg = utils.npy_to_dask(paths, axis=axis)
            return mov_reg

    def load_frame_counts(self):
        return n.load(os.path.join(self.dirs['job_dir'],'frames.npy'), allow_pickle=True).item()


    def get_exp_frame_idxs(self,exp_idx):
        frames = self.load_frame_counts()
        idxs = n.where(frames['jobids']==exp_idx)[0]
        st,en = idxs[0], idxs[-1]
        frame_start = frames['nframes'][:st].sum()
        frame_end = frames['nframes'][:en+1].sum()
        return frame_start, frame_end

    def save_frame_counts(self):
        size_to_frames = {}
        nframes = []
        jobids = []
        for tif in self.tifs:
            jobids.append(int(tif.split(os.path.sep)[-2]))
            tifsize = int(os.path.getsize(tif))
            if tifsize in size_to_frames.keys():
                nframes.append(size_to_frames[tifsize])
            else:
                tf = tifffile.TiffFile(tif)
                nf = len(tf.pages) // self.params.get('n_ch_tif', 30)
                nframes.append(nf)
                size_to_frames[tifsize] = nf
                self.log(tif +  ' is %d frames and %d bytes' % (nf, tifsize))

        nframes = n.array(nframes)
        jobids = n.array(jobids)

        tosave = {'nframes' : nframes, 'jobids' : jobids}
        self.frames = tosave
        n.save(os.path.join(self.dirs['job_dir'],'frames.npy'), tosave)

        return nframes, jobids
    

    def sweep_params(self, params_to_sweep,svd_info=None, mov=None, testing_dir_tag='sweep', 
                             n_test_iters = 1, all_combinations=True, do_vmap=True, svs=None,us=None,
                             test_parent_dir = None, delete_mov_sub = True):
        init_params = copy.deepcopy(self.params)
        testing_dir = self.make_new_dir(testing_dir_tag, parent_dir_name=test_parent_dir)
        sweep_summary_path = os.path.join(testing_dir, 'sweep_summary.npy')
        param_per_run = {}
        n_per_param = []
        param_names = []
        param_vals_list = []
        for k in params_to_sweep.keys():
            assert k in self.params.keys()
            param_names.append(k)
            n_per_param.append(len(params_to_sweep[k]))
            param_vals_list.append(params_to_sweep[k])
            param_per_run[k] = []
        if all_combinations:
            n_combs = n.product(n_per_param)
            combinations = n.array(list(itertools.product(*param_vals_list)))
        else:
            n_combs = n.sum(n_per_param)
            base_vals = [init_params[param_name] for param_name in param_names]
            combinations = n.stack([base_vals]*n_combs)
            cidx = 0
            for pidx in range(len(param_names)):
                for vidx in range(n_per_param[pidx]):
                    combinations[cidx][pidx] = param_vals_list[pidx][vidx]
                    cidx += 1
        assert len(combinations) == n_combs

        comb_strs = []; comb_params = []; comb_dir_tags = []; comb_dirs = []
        for comb_idx, comb in enumerate(combinations):
            comb_param = copy.deepcopy(init_params)
            comb_str = 'comb%05d-params' % comb_idx
            for param_idx, param in enumerate(param_names):
                param_value = comb[param_idx]
                if type(param_value) != str:
                    val_str = '%.03f' % param_value
                else: val_str = param_value
                comb_str += '-%s_%s' % (param, val_str)
                comb_param[param] = param_value    
            comb_dir_tag = testing_dir_tag + '-comb_%05d' % comb_idx
            comb_dir = self.make_new_dir(comb_dir_tag, parent_dir_name=testing_dir_tag)
            
            comb_params.append(comb_param); comb_dirs.append(comb_dir); 
            comb_strs.append(comb_str); comb_dir_tags.append(comb_dir_tag)
        sweep_summary = {
            'comb_strs' : comb_strs,
            'comb_dir_tags' : comb_dir_tags,
            'comb_params' : comb_params,
            'comb_dirs' : comb_dirs ,
            'param_names' : param_names,
            'combinations' : combinations,
            'param_sweep_dict' : params_to_sweep}
        n.save(sweep_summary_path, sweep_summary)
        self.log("Saving summary for %d combinations to %s" % (n_combs, sweep_summary_path))

        if do_vmap:
            vmaps = []
            for comb_idx in range(n_combs):
                comb_dir_tag = comb_dir_tags[comb_idx]; comb_dir = comb_dirs[comb_idx]
                comb_str = comb_strs[comb_idx]; 
                self.log("Running combination %02d/%02d" % (comb_idx + 1, n_combs), 0, log_mem_usage=True)
                self.log("Summary dict size: %02d GB" % (sys.getsizeof(sweep_summary)/1024**3))
                self.log("Combination params: %s" % comb_str, 2) 
                self.log("Saving to tag %s at %s" % (comb_dir_tag,comb_dir), 2) 
                self.params = comb_params[comb_idx]
                (vmap, mean_img, max_img), mov_sub_dir, iter_dir = \
                                            self.calculate_corr_map(mov=mov, svd_info=svd_info, parent_dir = comb_dir_tag,
                                            iter_limit=n_test_iters, update_main_params=False, svs=svs, us=us)
                if delete_mov_sub:
                    self.log("Removing mov_sub from %s" % mov_sub_dir)
                    shutil.rmtree(mov_sub_dir)
                vmaps.append(vmap)
                sweep_summary['vmaps'] = vmaps
                sweep_summary['mean_img'] = mean_img
                sweep_summary['max_img'] = max_img
                n.save(sweep_summary_path, sweep_summary)
        return sweep_summary
    
    def vis_vmap_sweep(self,summary):
        nz,ny,nx = summary['vmaps'][0].shape
        param_dict = summary['param_sweep_dict']
        param_names = summary['param_names']
        combinations = summary['combinations']
        vmaps = summary['vmaps']
        n_val_per_param = [len(param_dict[k]) for k in param_names]
        vmap_sweep = n.zeros(tuple(n_val_per_param) + (nz,ny,nx))
        print(n_val_per_param)
        print(vmap_sweep.shape)
        n_params = len(param_names)
        for cidx, combination in enumerate(combinations):
            param_idxs = [n.where(param_dict[param_names[pidx]] == combination[pidx])[0][0] \
                                for pidx in range(n_params)]
            vmap_sweep[tuple(param_idxs)] = vmaps[cidx]
        v = ui.napari.Viewer()
        v.add_image(summary['mean_img'], name='mean_img')
        v.add_image(summary['max_img'], name='max_img')
        v.add_image(vmap_sweep, name='Corrmap Sweep')
        v.dims.axis_labels = tuple(param_names + ['z','y','x'])
        return v
    

    def get_logged_mem_usage(self):
        mem_log_lines = []
        with open(self.logfile, 'r') as logf:
            lines = logf.readlines()
            for line in lines:
                if line.find("Virtual Available") > -1:
                    mem_log_lines.append(line)

        timestamps = []
        used_mem = []
        used_swp = []
        used_vrt = []
        avail_vrt = []
        descriptors = []

        for line in mem_log_lines:
            try:
                tstamp = datetime.datetime.strptime(line[1:20], '%Y-%m-%d %H:%M:%S')
                timestamps.append(tstamp)
            except:
                #print("Could not parse line %s" % line)
                continue
            
            
            tag = 'Total Used: '
            num_len = 7
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx:num_idx + 7]
            num_float = float(num_str)
            used_mem.append(num_float)

            descriptors.append(line[25:num_idx-len(tag)].strip())
            
            tag = 'Swap Used: '
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx:num_idx + 7]
            num_float = float(num_str)
            used_swp.append(num_float)
            
            tag = 'Virtual Used: '
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx:num_idx + 7]
            num_float = float(num_str)
            used_vrt.append(num_float)
            
            tag = 'Virtual Available: '
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx:num_idx + 7]
            num_float = float(num_str)
            avail_vrt.append(num_float)

        return timestamps, used_mem, used_swp, used_vrt, avail_vrt, descriptors
    
    def plot_memory_usage(self, show_descriptors_pctile=None):
        timestamps, used_mem, used_swp, used_vrt, avail_vrt, descriptors = self.get_logged_mem_usage()
        f,axs = plt.subplots(2,1,sharex=True, figsize=(8,8))

        ax = axs[0]
        ax.plot(timestamps, used_mem, label='Used')
        ax.plot(timestamps, used_swp, label='Swap')
        ax.plot(timestamps, used_vrt, label='Used (virtual)')
        ax.plot(timestamps, avail_vrt, label=("Available (virtual)"))
        ax.set_ylabel("Memory usage (GB)")
        ax.legend()

        ax = axs[1]
        deltas = n.diff(used_mem)
        ax.scatter(timestamps[1:],deltas) 
        ax.set_ylabel("Change in memory usage (GB)")
        ax.set_xlabel("Timestamp")

        if show_descriptors_pctile is not None:
            top_deltas = n.where(deltas > n.percentile(deltas, show_descriptors_pctile))[0]
            for top_idx in top_deltas:
                ax.text(timestamps[top_idx+1],deltas[top_idx], descriptors[top_idx], rotation=-45, rotation_mode='anchor')
                ax.scatter([timestamps[top_idx+1]],[deltas[top_idx]], s=5, color='red')

        # plt.show()
        return f,axs