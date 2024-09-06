try:
    import tifffile
except:
    print("No tifffile")
import datetime
import os
import copy
import time
import sys
import numpy as n
import itertools
from multiprocessing import Pool
import shutil
from matplotlib import pyplot as plt
from skimage.io import imread

try:
    import psutil
except:
    print("No psutil")

from suite2p.extraction import dcnv

from . import init_pass
from . import utils
from . import lbmio
from .iter_step import (
    register_dataset,
    fuse_and_save_reg_file,
    calculate_corrmap,
    calculate_corrmap_from_svd,
    register_dataset_gpu,
    register_dataset_gpu_3d,
)

from . import corrmap
from . import extension as ext
from .default_params import get_default_params
from . import svd_utils as svu
from . import ui


class Job:
    def __init__(
        self,
        root_dir,
        job_id,
        params=None,
        tifs=None,
        overwrite=False,
        verbosity=10,
        create=True,
        params_path=None,
        parent_job=None,
        copy_parent_dirs=(),
        copy_parent_symlink=False,
    ):
        """Create a Job object that is a wrapper to manage files, current state, log etc.
        Args:
            root_dir (str): Root directory in which job directory will be created
            job_id (str): Unique name for the job directory
            params (dict): Job parameters (see examples)
            tifs (list) : list of full paths to tif files to be used
            overwrite (bool, optional): If True, and creat=True, this will overwrite the params and dir files if they already existed for this job. If False, and create=True, this will throw an error if job_dir exists. Defaults to False.
            params_path : if you have moved the job from somewhere else, the params_path needs to be provided explicitly. You probably also need to call update_root_path in that case TODO: fix how root directory is handled on job.dirs
            parent_job : you can create a copy of a parent job, inheriting all the parameters in a new directory
            copy_parent_dirs (tuple) : list of directories to copy from the parent job
            copy_parent_symlink (bool) : if copying dirs, you can optionally symlink them
            verbosity (int, optional): Verbosity level. 0: critical only, 1: info, 2: debug. Defaults to 1.
        """

        self.verbosity = verbosity
        self.job_id = job_id
        self.summary = None

        if create:
            if parent_job is not None:
                self.init_job_dir(root_dir, job_id, exist_ok=overwrite)
                return self.copy_parent_job(parent_job, copy_parent_dirs, copy_parent_symlink)
            self.init_job_dir(root_dir, job_id, exist_ok=overwrite)
            def_params = get_default_params()
            self.log("Loading default params")
            for k, v in params.items():
                assert k in def_params.keys(), "%s not a valid parameter" % k
                self.log("Updating param %s" % (str(k)), 2)
                def_params[k] = v
            self.params = def_params
            assert tifs is not None, "Must provide tiff files"
            self.params["tifs"] = tifs
            self.tifs = tifs
            self.save_params()
        else:
            self.job_dir = os.path.join(root_dir, "s3d-%s" % job_id)
            self.load_dirs()
            self.load_params(params_path=params_path)
            self.tifs = self.params.get("tifs", [])

    def copy_parent_job(self, parent_job, copy_dirs=(), symlink=False):
        """
        Copy the initial pass results, params and more from another job

        Args:
            parent_job (Job): A job object that you want to copy over
            copy_dirs (tuple, optional): Directories from the parent job you want to copy. Defaults to ().
            symlink (bool, optional): If True, create symlinks to the parent directories instead of copying them fully. Defaults to False.
        """
        self.params = parent_job.load_params()
        self.copy_init_pass_from_job(parent_job)
        self.log("Copied init pass and parameters from parent job")
        self.tifs = self.params["tifs"]

        for key in copy_dirs:
            self.log("Copying dir %s from parent job" % key)
            # path_suffix = parent_job.dirs[key][len(parent_job.dirs['job_dir']) + len(os.path.sep):]
            # new_path = os.path.join(self.dirs['job_dir'], path_suffix)
            old_dir_path = parent_job.dirs[key]
            new_dir_path = self.make_new_dir(key)
            if not symlink:
                shutil.copytree(old_dir_path, new_dir_path, dirs_exist_ok=True)
            else:
                os.symlink(old_dir_path, new_dir_path, target_is_directory=True)

        self.save_params()

    def log(self, string="", level=1, logfile=True, log_mem_usage=False):
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
            string += (
                "Total Used: %07.3f GB, Virtual Available: %07.3f GB, Virtual Used: %07.3f GB, Swap Used: %07.3f GB"
                % ((total / (1024**3), vm_avail / (1024**3), vm_unavail / (1024**3), sm.used / (1024**3)))
            )

        if level <= self.verbosity:
            # print('xxx')
            print(("   " * level) + string)
        if logfile:
            logfile = os.path.join(self.job_dir, "log.txt")
            self.logfile = logfile
            with open(logfile, "a+") as f:
                datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = "\n[%s][%02d] " % (datetime_string, level)
                f.write(header + "   " * level + string)

    def load_file(self, filename, dir_name=None, path=None, allow_pickle=True, mmap_mode=None):
        """
        Light wrapper around n.load() to load an arbitrary .npy file from self.base_dir
        """
        if filename[-4:] != ".npy":
            filename = filename + ".npy"
        if path is not None:
            filepath = os.path.join(path, filename)
        elif dir_name is not None:
            filepath = os.path.join(self.dirs[dir_name], filename)
        else:
            assert False
        if not os.path.exists(filepath):
            self.log("Did not find %s" % filepath, 1)
            return None
        self.log("Loading from %s" % filepath, 2)
        file = n.load(filepath, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        if file.dtype == "O" and file.ndim < 1:
            file = file.item()
        return file

    def save_file(self, filename, data, dir_name=None, path=None, overwrite=True):
        """
        Light wrapper around n.save() to save an arbitrary data to a .npy file
        to self.basedir, with overwrite protection

        Args:
            filename (str): name + extension of file
        """
        if filename[-4:] != ".npy":
            filename = filename + ".npy"
        if dir_name is not None:
            filepath = os.path.join(self.dirs[dir_name], filename)
        elif path is not None:
            filepath = os.path.join(path, filename)
        else:
            assert False
        if os.path.exists(filepath):
            if not overwrite:
                self.log("File %s already exists. Not overwriting." % filepath, 2)
                return
            self.log("Overwriting existing %s" % filepath, 2)
        n.save(filepath, data)

    def make_new_dir(
        self, dir_name, parent_dir_name=None, exist_ok=True, dir_tag=None, add_to_dirs=True, return_dir_tag=False
    ):
        """
        Create a new directory and save the full path to it in self.dirs[dir_name]

        Args:
            dir_name (str): name of the directory to create
            parent_dir_name (str, optional): Name of parent directory. If None, create new dir in the root job_dir. Defaults to None.
            exist_ok (bool, optional): If False, throw an error if the directory already exists. Defaults to True.
            dir_tag (str, optional): Typically, the path created (e.g. /data/s3d-testjob/dirname) is saved in job.dirs['dirname']. If you want the key to be something else, e.g. you want it to be saved under job.dirs['mydir'], set dir_tag = 'mydir'. Only used if you're doing something weird. Defaults to None.
            add_to_dirs (bool, optional): if False, make the dir but don't add to job.dirs

        Returns:
            str: full path to directory
        """
        if parent_dir_name is None:
            parent_dir = self.job_dir
        elif parent_dir_name in self.dirs.keys():
            parent_dir = self.dirs[parent_dir_name]
        else:
            parent_dir = self.make_new_dir(parent_dir_name, exist_ok=False)

        if dir_tag is None:
            dir_tag = dir_name
        if parent_dir_name is not None:
            dir_tag = parent_dir_name + "-" + dir_tag
        dir_path = os.path.join(parent_dir, dir_name)
        if os.path.exists(dir_path):
            self.log("Found dir %s" % (dir_path,), 2)
        else:
            os.makedirs(dir_path, exist_ok=exist_ok)
            self.log("Created dir %s with tag %s" % (dir_path, dir_tag))
        if add_to_dirs:
            self.log("Updating self.dirs tag %s" % dir_tag, 2)
            self.dirs[dir_tag] = dir_path
        n.save(os.path.join(self.job_dir, "dirs.npy"), self.dirs)
        if return_dir_tag:
            return dir_tag, dir_path
        return dir_path

    def save_dirs(self, name="dirs", dirs=None):
        """
        save dirs.npy, which contains the paths to all of the things in the job

        Args:
            name (str, optional): Name of the file to save, don't change. Defaults to 'dirs'.
            dirs (dict, optional): If you want to save a different dirs file than self.dirs. Don't change . Defaults to None.
        """
        if dirs is None:
            dirs = self.dirs
        n.save(os.path.join(self.job_dir, "%s.npy" % name), dirs)

    def load_dirs(self):
        """
        Load dirs.npy into self.dirs
        """
        self.dirs = n.load(os.path.join(self.job_dir, "dirs.npy"), allow_pickle=True).item()

    def save_params(self, new_params=None, copy_dir_tag=None, params=None, update_main_params=True, copy_dir=None):
        """
        Update saved params in job_dir/params.npy

        Args:
            new_params (dict, optional): Dictionary containing parameters to update. Defaults to None.
            copy_dir_tag (str, optional): If set, save a copy of the params file in the directory specified by the directory tag copy_dir_tag. Defaults to None.
            params (dict, optional): Params dict to update, usually set to None so we update the main params dict in self.params . Defaults to None.
            update_main_params (bool, optional): Update the params.npy file in the root job_dir. Defaults to True.
        """
        if params is None:
            params = self.params
        if new_params is not None:
            params.update(new_params)
        if copy_dir_tag is not None:
            params_path = os.path.join(self.dirs[copy_dir_tag], "params.npy")
            n.save(params_path, params)
            self.log("Saved a copy of params at %s" % self.dirs[copy_dir_tag])
        if copy_dir is not None:
            params_path = os.path.join(copy_dir, "params.npy")
            n.save(params_path, params)
            self.log("Saved a copy of params at %s" % copy_dir)
        if update_main_params:
            n.save(os.path.join(self.dirs["job_dir"], "params.npy"), params)
        self.log("Updated main params file")

    def load_params(self, dir=None, params_path=None):
        """
        Load params.npy into job.params

        Args:
            dir (str, optional): dir_tag for the directory where the params.npy is located. If None, load the one in the root directory. Ignored if params_path is not None. Defaults to None.
            params_path (str, optional): Full path the a params file to load. Only use if doing something weird (e.g. loading a params file not called params.npy, or one that isn't in ia job directory). Defaults to None.

        Returns:
            _type_: _description_
        """
        if params_path is None:
            if dir is None:
                dir = "job_dir"
            params_path = os.path.join(self.dirs[dir], "params.npy")
        self.params = n.load(params_path, allow_pickle=True).item()
        self.log("Found and loaded params from %s" % params_path)
        return self.params

    def make_extension_dir(self, extension_root, extension_name="ext"):
        extension_dir = os.path.join(extension_root, "s3d-extension-%s" % self.job_id)
        if extension_name in self.dirs.keys():
            self.log("Extension dir %s already exists at %s" % (extension_name, self.dirs[extension_name]))
            return self.dirs[extension_name]
        os.makedirs(extension_dir)
        self.log("Made new extension dir at %s" % extension_dir)
        self.dirs[extension_name] = extension_dir
        self.save_dirs()
        return extension_dir

    def update_root_path(self, new_root):
        old_dirs = copy.deepcopy(self.dirs)
        root_len = self.dirs["summary"].find("s3d-" + self.job_id)
        self.log("Replacing %s with %s" % (self.dirs["summary"][:root_len], new_root))
        for k, v in self.dirs.items():
            self.dirs[k] = os.path.join(new_root, v[root_len:])
        self.save_dirs()
        self.save_dirs("old_dirs", old_dirs)

    # def make_new_dir(self, dir_name, parent_dir_name = None, exist_ok=True, dir_tag = None):
    #     if parent_dir_name is None:
    #         parent_dir = self.job_dir
    #     else:
    #         if parent_dir_name  not in self.dirs.keys():
    #             self.log("Creating parent directory % s" % parent_dir_name)
    #             self.make_new_dir(parent_dir_name)
    #         parent_dir = self.dirs[parent_dir_name]
    #     if dir_tag is None:
    #         dir_tag = dir_name

    #     dir_path = os.path.join(parent_dir, dir_name)
    #     if os.path.exists(dir_path):
    #         self.log("Found dir %s with tag %s" % (dir_path, dir_tag), 2)
    #     else:
    #         os.makedirs(dir_path, exist_ok = exist_ok)
    #         self.log("Created dir %s with tag %s" % (dir_path, dir_tag))
    #     self.dirs[dir_tag] = dir_path
    #     n.save(os.path.join(self.job_dir, 'dirs.npy'), self.dirs)
    #     return dir_path

    def init_job_dir(self, root_dir, job_id, exist_ok=False):
        """Create a job directory and nested dirs

        Args:
            root_dir (str): Root directory to create job_dir in
            job_id (str): Unique name for job
            exist_ok (bool, optional): If False, throws error if job_dir exists. Defaults to False.
        """

        job_dir = os.path.join(root_dir, "s3d-%s" % job_id)
        self.job_dir = job_dir
        if os.path.isdir(job_dir):
            self.log("Job directory %s already exists" % job_dir, 0)
            assert exist_ok, "Set create=False to load existing job, or set overwrite=True to overwrite existing job"
        else:
            os.makedirs(job_dir, exist_ok=True)

        self.log("Loading job directory for %s in %s" % (job_id, root_dir), 0)
        if "dirs.npy" in os.listdir(job_dir):
            self.log("Loading dirs ")
            self.dirs = n.load(os.path.join(job_dir, "dirs.npy"), allow_pickle=True).item()
        else:
            self.dirs = {"job_dir": self.job_dir}

        if job_dir not in self.dirs.keys():
            self.dirs["job_dir"] = self.job_dir

        for dir_name in ["registered_fused_data", "summary", "iters"]:
            if dir_name not in self.dirs.keys() or not os.path.isdir(self.dirs[dir_name]):
                new_dir = os.path.join(job_dir, dir_name)
                if not os.path.isdir(new_dir):
                    os.makedirs(new_dir, exist_ok=True)
                    self.log("Created dir %s" % new_dir, 2)
                self.dirs[dir_name] = new_dir
            else:
                self.log("Found dir %s" % dir_name, 2)
        n.save(os.path.join(job_dir, "dirs.npy"), self.dirs)

    def run_init_pass(self):
        self.save_params(copy_dir_tag="summary")
        self.log("Launching initial pass", 0)
        init_pass.run_init_pass(self)

    def copy_init_pass_from_job(self, old_job):
        n.save(os.path.join(self.dirs["summary"], "summary.npy"), old_job.load_summary())
        self.summary = old_job.summary

    def copy_init_pass(self, summary_old_job):
        n.save(os.path.join(self.dirs["summary"], "summary.npy"), summary_old_job)
        self.summary = summary_old_job

    def load_summary(self):
        """
        Load the results of the init_pass

        Returns:
            dict: dictionary containing reference images, plane shifts, etc.
        """
        summary_path = os.path.join(self.dirs["summary"], "summary.npy")
        summary = n.load(summary_path, allow_pickle=True).item()
        self.summary = summary
        return summary

    def show_summary_plots(self):
        summary = self.load_summary()
        f1 = plt.figure(figsize=(8, 4), dpi=200)
        plt.plot(summary["plane_shifts"])
        plt.xlabel("Plane")
        plt.ylabel("# pixels of shift")
        plt.title("LBM shift between planes")
        plt.ylim(-100, 100)

        crosstalk_dir = os.path.join(self.dirs["summary"], "crosstalk_plots")
        gamma_fit_img = os.path.join(crosstalk_dir, "gamma_fit.png")
        plane_fits_img = os.path.join(crosstalk_dir, "plane_fits.png")

        if os.path.isfile(plane_fits_img):
            im = imread(plane_fits_img)
            f2, ax = plt.subplots(figsize=(im.shape[0] // 200, im.shape[1] // 200), dpi=400)
            ax.imshow(im)
            ax.set_axis_off()
        if os.path.isfile(gamma_fit_img):
            im = imread(gamma_fit_img)
            f3, ax = plt.subplots(figsize=(im.shape[0] // 200, im.shape[1] // 200), dpi=150)
            ax.imshow(im)
            ax.set_axis_off()

        if "fuse_shifts" in summary.keys() and "fuse_ccs" in summary.keys():
            if summary["fuse_shifts"] is not None:
                utils.plot_fuse_shifts(summary["fuse_shifts"], summary["fuse_ccs"])

    def register(self, tifs=None, start_batch_idx=0):
        """
        Register the dataset using the method specified in job.params.

        Args:
            tifs (list): List of tif files to register. If None, uses self.tifs.
            start_batch_idx (int): Starting batch index.
        """
        self.make_new_dir("registered_fused_data")
        params = self.params
        summary = self.load_summary()
        self.save_params(params=params, copy_dir_tag="registered_fused_data")

        if tifs is None:
            tifs = self.tifs

        do_3d_reg = params.get("3d_reg", False)
        do_gpu_reg = params.get("gpu_reg", False)

        self.log(f"Starting registration: 3D: {do_3d_reg}, GPU: {do_gpu_reg}", 1)

        if do_3d_reg:
            if do_gpu_reg:
                register_dataset_gpu_3d(
                    self, tifs, params, self.dirs, summary, self.log, start_batch_idx=start_batch_idx
                )
            else:
                pass
                # register_dataset_3d(self,tifs, params, self.dirs, summary, self.log, start_batch_idx=start_batch_idx)
        else:
            if do_gpu_reg:
                register_dataset_gpu(self, tifs, params, self.dirs, summary, self.log, start_batch_idx=start_batch_idx)
            else:
                register_dataset(self, tifs, params, self.dirs, summary, self.log, start_batch_idx=start_batch_idx)

    def register_gpu(self, tifs=None, max_gpu_batches=None):
        params = self.params
        summary = self.load_summary()
        save_dir = self.make_new_dir("registered_fused_data")
        if tifs is None:
            tifs = self.tifs
        register_dataset_gpu(self, tifs, params, self.dirs, summary, self.log, max_gpu_batches=max_gpu_batches)

    def register_gpu_3d(self, tifs=None, max_gpu_batches=None):
        params = self.params
        summary = self.load_summary()
        save_dir = self.make_new_dir("registered_fused_data")
        if tifs is None:
            tifs = self.tifs
        register_dataset_gpu_3d(self, tifs, params, self.dirs, summary, self.log, max_gpu_batches=max_gpu_batches)

    def calculate_corr_map(self, mov=None, save=True, iter_limit=None, output_dir_name=None, save_mov_sub=True):
        """
        Calculate the correlation map. Saves the correlation map results in
        parent_dir/corrmap, and saves the neuropil subtracted movie in
        parent_dir/mov_sub

        Args:
            mov (ndarray or dask array, optional): nz, nt, ny, nx. If none, the registered movie will be used. Defaults to None.
            save (bool, optional): Whether to create dirs and save results. Defaults to True.
            iter_limit (int, optional): Number of batches to run. Set to None for the whole recording. Defaults to None.
            output_dir_name (str, optional): Name of the parent directory to place results in. Defaults to None.
        """
        """
        if save:
            corr_map_dir = self.make_new_dir("corrmap", parent_dir_name=output_dir_name)
            mov_sub_dir = self.make_new_dir("mov_sub", parent_dir_name=output_dir_name)
        else:
            corr_map_dir = self.make_new_dir("corrmap", parent_dir_name=output_dir_name)
            mov_sub_dir = self.make_new_dir("mov_sub", parent_dir_name=output_dir_name)
        else:
            corr_map_dir = None
            mov_sub_dir = None

        if mov is None:
            mov = self.get_registered_movie("registered_fused_data", "fused")
            mov = self.get_registered_movie("registered_fused_data", "fused")

        self.save_params(copy_dir=corr_map_dir)
        self.corrmap = corrmap.calculate_corrmap(
            mov=mov,
            params=self.params,
            batch_dir=corr_map_dir,
            mov_sub_dir=mov_sub_dir,
            iter_limit=iter_limit,
            summary=self.load_summary(),
            log=self.log,
            save_mov_sub=save_mov_sub,
        )

        return self.corrmap


    def load_corr_map_results(self, parent_dir_name=None):
        files = ["max_img.npy", "mean_img.npy", "vmap.npy"]
        corrmap_dir_tag = "corrmap"
        files = ["max_img.npy", "mean_img.npy", "vmap.npy"]
        corrmap_dir_tag = "corrmap"
        if parent_dir_name is not None:
            corrmap_dir_tag = parent_dir_name + "-corrmap"
            corrmap_dir_tag = parent_dir_name + "-corrmap"
        results = {}
        for file in files:
            if file in os.listdir(self.dirs[corrmap_dir_tag]):
                results[file[:-4]] = n.load(os.path.join(self.dirs[corrmap_dir_tag], file))
        return results

    def setup_sweep(self, params_to_sweep, sweep_name, sweep_parent_dir="sweeps", all_combinations=True):
        """
        Setup the combinations of parameters and creates directories for a sweep

        Args:
            params_to_sweep (dict): Each key is a param name, values are lists
                                    of values for the param to sweep through
            sweep_name (str): name of the sweep
            sweep_parent_dir (str, optional): Parent directory name for the sweep to be stored in. Defaults to 'sweeps'.
            all_combinations (bool, optional): Whether to do all combinations of all values of parameters, or to start with a base set of params (self.params) and only vary a single parameter at a time.
              If set to True, you get a lot of combinations. Defaults to True.

        Returns:
            dict: sweep_summary: contains combinations of parameters for each run, and directories, etc.
        """
        self.log("Setting up sweep")
        # make a copy of the param file before the sweep
        init_params = copy.deepcopy(self.params)
        # make a directory for this sweep within the parent directory of all sweeps
        sweep_dir_name, sweep_dir = self.make_new_dir(
            sweep_name, parent_dir_name=sweep_parent_dir, return_dir_tag=True
        )

        n_per_param = []
        param_names = []
        param_vals_list = []
        # for each parameter that is sweeped, collect its possible values
        for k in params_to_sweep.keys():
            assert k in self.params.keys(), "%s not in params" % k
            param_names.append(k)
            n_per_param.append(len(params_to_sweep[k]))
            param_vals_list.append(params_to_sweep[k])
            assert (
                self.params[k] in params_to_sweep[k]
            ), "The 'base' value of the parameter %s should be included in the sweep (%s)" % (k, str(self.params[k]))
        if all_combinations:
            n_combs = n.product(n_per_param)
            combinations = list(itertools.product(*param_vals_list))
        else:
            n_combs = n.sum(n_per_param)
            base_vals = [init_params[param_name] for param_name in param_names]
            for i in range(n_combs):
                combinations.append(copy.copy(base_vals))
            cidx = 0
            for pidx in range(len(param_names)):
                for vidx in range(n_per_param[pidx]):
                    combinations[cidx][pidx] = param_vals_list[pidx][vidx]
                    cidx += 1
        self.log("Total of %d combinations" % n_combs, 1)
        # combinations is an array of size n_combs
        # each element is a list with the value of each param for the corresponding combination
        assert len(combinations) == n_combs

        comb_strs = []
        comb_params = []
        comb_dir_names = []
        comb_dirs = []
        for comb_idx, comb in enumerate(combinations):
            comb_param = copy.deepcopy(init_params)
            comb_str = "comb%05d-params" % comb_idx
            for param_idx, param in enumerate(param_names):
                param_value = comb[param_idx]
                if type(param_value) == str or type(param_value) == n.str_:
                    val_str = param_value
                else:
                    val_str = "%.03f" % param_value
                comb_str += "-%s_%s" % (param, val_str)
                comb_param[param] = param_value
            comb_dir_tag = "comb_%05d" % comb_idx
            self.log("Created directory for %s with params %s" % (comb_dir_tag, comb_str), 2)
            # create directories for each combination
            comb_dir_tag, comb_dir = self.make_new_dir(
                comb_dir_tag, parent_dir_name=sweep_dir_name, add_to_dirs=True, return_dir_tag=True
            )

            comb_params.append(comb_param)
            comb_dirs.append(comb_dir)
            comb_strs.append(comb_str)
            comb_dir_names.append(comb_dir_tag)
        sweep_summary = {
            "sweep_dir_path": sweep_dir,
            "sweep_dir_name": sweep_dir_name,
            "init_params": init_params,
            "comb_strs": comb_strs,
            "comb_dir_names": comb_dir_names,
            "comb_params": comb_params,
            "comb_dirs": comb_dirs,
            "param_names": param_names,
            "combinations": combinations,
            "all_combinations": all_combinations,
            "param_sweep_dict": params_to_sweep,
        }

        self.save_file(filename="sweep_summary", data=sweep_summary, path=sweep_dir)
        return sweep_summary

    def sweep_corrmap(
        self,
        params_to_sweep,
        sweep_name="corrmap",
        all_combinations=True,
        mov=None,
        iter_limit=None,
        save_mov_sub=False,
    ):
        sweep_summary = self.setup_sweep(params_to_sweep, sweep_name, all_combinations=all_combinations)
        sweep_summary["sweep_type"] = "corrmap"
        sweep_dir_path = sweep_summary["sweep_dir_path"]
        sweep_summary["results"] = []
        combinations = sweep_summary["combinations"]
        n_combs = len(combinations)
        for comb_idx in range(n_combs):
            comb_dir_name = sweep_summary["comb_dir_names"][comb_idx]
            comb_params = sweep_summary["comb_params"][comb_idx]
            self.log("Running combination %02d/%02d" % (comb_idx + 1, n_combs), 0)
            comb_dir_name = sweep_summary["comb_dir_names"][comb_idx]
            comb_params = sweep_summary["comb_params"][comb_idx]
            self.log("Running combination %02d/%02d" % (comb_idx + 1, n_combs), 0)
            self.params = comb_params
            corrmap = self.calculate_corr_map(
                output_dir_name=comb_dir_name, save_mov_sub=save_mov_sub, mov=mov, iter_limit=iter_limit
            )
            results = {"corrmap": corrmap, "output_dir": comb_dir_name}
            if comb_idx == 0:
                maps = self.load_corr_map_results(comb_dir_name)
                sweep_summary["mean_img"] = maps["mean_img"]
                sweep_summary["max_img"] = maps["max_img"]
                sweep_summary["mean_img"] = maps["mean_img"]
                sweep_summary["max_img"] = maps["max_img"]

            sweep_summary["results"].append(results)
            self.save_file("sweep_summary", sweep_summary, path=sweep_dir_path)
            sweep_summary["results"].append(results)
            self.save_file("sweep_summary", sweep_summary, path=sweep_dir_path)

        sweep_summary["complete"] = True
        self.save_file("sweep_summary", sweep_summary, path=sweep_dir_path)
        self.params = sweep_summary["init_params"]
        return sweep_summary

    def sweep_segmentation(
        self,
        params_to_sweep,
        sweep_name="seg",
        all_combinations=False,
        patches_to_segment=None,
        ts=None,
        input_dir_name=None,
        vmap=None,
    ):
        """
        Run segmentation with many different parameters

        Args:
            params_to_sweep (dict): Dictionary where keys are parameter names, and the values
                                    are the values each parameter will take during the sweep
            sweep_name (str, optional): Directory name under which sweep will be stored. Defaults to 'seg'.
            all_combinations (bool, optional): Whether to do all combinations of parameters, or to vary one at a time. Defaults to False.
            patches_to_segment (tuple, optional): Indices of patches of the movie to segment. Defaults to None.
            ts (tuple, optional): Indices of the start and end times of the movie to use. Defaults to None.
            input_dir_name (str, optional): Tag for directory containing corr_map and mov_sub directories. Typically this is the root directory, so leave as None.

        Returns:
            dict: sweep_summary containing results and sweep info
        """
        """
        sweep_summary = self.setup_sweep(params_to_sweep, sweep_name, all_combinations=all_combinations)
        sweep_summary["sweep_type"] = "segmentation"
        sweep_dir_path = sweep_summary["sweep_dir_path"]
        sweep_summary["results"] = []
        combinations = sweep_summary["combinations"]
        sweep_summary["sweep_type"] = "segmentation"
        sweep_dir_path = sweep_summary["sweep_dir_path"]
        sweep_summary["results"] = []
        combinations = sweep_summary["combinations"]
        n_combs = len(combinations)
        for comb_idx in range(n_combs):
            comb_dir_name = sweep_summary["comb_dir_names"][comb_idx]
            comb_params = sweep_summary["comb_params"][comb_idx]
            self.log("Running combination %02d/%02d" % (comb_idx + 1, n_combs), 0)
            comb_dir_name = sweep_summary["comb_dir_names"][comb_idx]
            comb_params = sweep_summary["comb_params"][comb_idx]
            self.log("Running combination %02d/%02d" % (comb_idx + 1, n_combs), 0)
            self.params = comb_params
            output_dir = self.segment_rois(
                output_dir_name=comb_dir_name,
                ts=ts,
                patches_to_segment=patches_to_segment,
                input_dir_name=input_dir_name,
                vmap=vmap,
            )
            results = {
                "stats": self.load_segmentation_results(output_dir, to_load=["stats"]),
                "roi_dir": output_dir,
            }
            if comb_idx == 0:
                results["info"] = self.load_segmentation_results(output_dir, to_load=["info"])
            sweep_summary["results"].append(results)
            self.save_file("sweep_summary", sweep_summary, path=sweep_dir_path)

        sweep_summary["complete"] = True
        self.save_file("sweep_summary", sweep_summary, path=sweep_dir_path)
        self.params = sweep_summary["init_params"]
        return sweep_summary

    def make_svd_dirs(self, n_blocks=None):
        self.make_new_dir("svd")
        self.make_new_dir("blocks", "svd", dir_tag="svd_blocks")
        block_dirs = []
        if n_blocks is not None:
            for i in range(n_blocks):
                block_dirs.append(self.make_new_dir("%03d" % i, "svd_blocks", dir_tag="svd_blocks_%03d" % i))
            return block_dirs

    def make_stack_dirs(self, n_stacks):
        stack_dirs = []
        self.make_new_dir("stacks", "svd", dir_tag="svd_stacks")
        for i in range(n_stacks):
            stack_dirs.append(self.make_new_dir("%03d" % i, "svd_stacks", dir_tag="svd_stacks_%03d" % i))
        return stack_dirs

    def segment_rois(self, input_dir_name=None, output_dir_name=None, patches_to_segment=None, ts=None, vmap=None):
        """
        Start from the correlation map in parent_dir and segment into ROIs

        Args:
            input_dir_name (str, optional): Tag for the directory containing the directories corr_map and mov_sub. Typically the root directory, so leave as None.
            output_dir_name (str, optional): Tag for the directory containing the results..
            patches_to_segment (tuple, optional): List of patches to segment. If none, segments the full movie. Defaults to None.
            ts (tuple, optional): Two integers, start and end indices of the movie to use. None means use the full movie. Defaults to None.
        """

        # load the results of the correlation map step
        mov_sub = self.get_subtracted_movie(parent_dir_name=input_dir_name)
        maps = self.load_corr_map_results(parent_dir_name=input_dir_name)
        if vmap is None:
            vmap = maps["vmap"]
        nt, nz, ny, nx = mov_sub.shape
        if ts is None:
            ts = (0, nt)
        if ts is None:
            ts = (0, nt)

        # segmentation_dir contains all of the sub-folders for each patch
        # output_dir contains the combined output for all patches
        segmentation_dir_tag, segmentation_dir_path = self.make_new_dir(
            "segmentation", output_dir_name, return_dir_tag=True
        )
        self.save_params(copy_dir_tag=segmentation_dir_tag)
        rois_dir_name, rois_dir_path = self.make_new_dir("rois", output_dir_name, return_dir_tag=True)

        self.log("Saving results to %s and %s " % (segmentation_dir_path, rois_dir_path))
        info = copy.deepcopy(maps)
        info["all_params"] = self.params
        print(info.keys())
        n.save(os.path.join(rois_dir_path, "info.npy"), info)

        # get the coordinates to split the movie into patches
        patch_size_xy = self.params["patch_size_xy"]
        patch_overlap_xy = self.params["patch_overlap_xy"]
        nt, nz, ny, nx = mov_sub.shape
        patches, grid_shape = svu.make_blocks((nz, ny, nx), (nz,) + patch_size_xy, (0,) + patch_overlap_xy)
        patches_vmap, __ = svu.make_blocks(
            (nz, ny, nx), (nz,) + patch_size_xy, (0,) + patch_overlap_xy, nonoverlapping_mask=True
        )
        n_patches = patches.shape[1]

        # optional argument to segment only some patches
        if patches_to_segment is None:
            patches_to_segment = n.arange(n_patches)

        # loop through all patches and segment them
        patch_counter = 1
        for patch_idx in patches_to_segment:
            self.log("Detecting from patch %d / %d" % (patch_counter, len(patches_to_segment)), 1)

            # set up the save directory for this patch
            patch_dir = self.make_new_dir("patch-%04d" % patch_idx, segmentation_dir_tag, add_to_dirs=False)
            stats_path = os.path.join(patch_dir, "stats.npy")
            info_path = os.path.join(patch_dir, "info.npy")

            zs, ys, xs = patches[:, patch_idx]
            vzs, vys, vxs = patches_vmap[:, patch_idx]

            # prepare the movie
            mov_patch = mov_sub[ts[0] : ts[1], zs[0] : zs[1], ys[0] : ys[1], xs[0] : xs[1]]
            if self.params["detection_timebin"] > 1:
                self.log("Binning movie with a factor of %.2f" % self.params["detection_timebin"], 2)
                mov_patch = ext.binned_mean(mov_patch, self.params["detection_timebin"])
            self.log(
                "Loading %.2f GB movie to memory, shape: %s " % (mov_patch.nbytes / 1024**3, str(mov_patch.shape)), 3
            )
            mov_patch = mov_patch.compute()
            self.log("Loaded", 3)

            # prepare the correlation map
            vmap_patch = n.zeros_like(mov_patch[0])
            dz = vzs[0] - zs[0]
            dy = vys[0] - ys[0]
            dx = vxs[0] - xs[0]
            vmap_patch[dz : dz + (vzs[1] - vzs[0]), dy : dy + (vys[1] - vys[0]), dx : dx + (vxs[1] - vxs[0])] = vmap[
                vzs[0] : vzs[1], vys[0] : vys[1], vxs[0] : vxs[1]
            ]

            mini_info = {"vmap": vmap_patch}

            stats = ext.detect_cells_mp(
                mov_patch,
                vmap_patch,
                **self.params,
                log=self.log,
                savepath=stats_path,
                patch_idx=patch_idx,
                offset=(zs[0], ys[0], xs[0])
            )
            n.save(info_path, mini_info)
            patch_counter += 1

        # combine all segmented patches
        rois_dir_path = self.combine_patches(
            patches_to_segment, rois_dir_path, parent_dir_name=segmentation_dir_tag, info_use_idx=None
        )
        rois_dir_path = self.combine_patches(
            patches_to_segment, rois_dir_path, parent_dir_name=segmentation_dir_tag, info_use_idx=None
        )

        return rois_dir_path

    def compute_npil_masks(self, stats_dir):
        info = n.load(os.path.join(stats_dir, "info.npy"), allow_pickle=True).item()
        stats = n.load(os.path.join(stats_dir, "stats.npy"), allow_pickle=True)
        nz, ny, nx = info["vmap"].shape
        n.save(os.path.join(stats_dir, "stats_small.npy"), stats)
        stats = ext.compute_npil_masks_mp(stats, (nz, ny, nx), n_proc=self.params["n_proc_corr"])
        n.save(os.path.join(stats_dir, "stats.npy"), stats)
        return stats_dir

    def load_segmentation_results(self, output_dir_path=None, output_dir_name="rois", to_load=None):
        """
        Load the results of cell segmentation from disk. Can provide the dir_name or absolute path
        to the directory containing stats.npy and info.npy (typically job_dir/rois)

        Args:
            output_dir_path (str, optional): Absolute path to the directory containing results.
            output_dir_name (str, optional): "Name" of the path, e.g. the key under which it is listed in self.dirs
            to_load (list, optional): Optional list of files to load. Typical options are info, stats and iscell.

        Returns:
            _type_: _description_
        """
        if to_load is None:
            to_load = ["info", "stats", "iscell"]
        to_return = {}
        for file in to_load:
            data = self.load_file(file, path=output_dir_path, dir_name=output_dir_name)
            if len(to_load) == 1:
                return data
            to_return[file] = data
        return to_return

    def export_results(self, export_path, result_dir_name="rois", results_to_export=None, export_frame_counts=True):
        """
        Save the relevant outputs of suite3d in a specified directory for further processing.
        Outputs will be saved in export_path/s3d-results-job_id

        Args:
            export_path (str): absolute path to the parent directory where results will be saved
            result_dir_name (str, optional): name of the directory where results are currently saved. Defaults to 'rois'.
            results_to_export (list, optional): list of files to export. Defaults to the important ones.
            export_frame_counts (bool, optional): Whether to export the number of frames in each file. Defaults to True.
        """
        full_export_path = os.path.join(export_path, "s3d-results-%s" % self.job_id)
        os.makedirs(full_export_path, exist_ok=True)
        self.log("Created dir %s to export results" % full_export_path)
        if results_to_export is None:
            results_to_export = ["stats_small.npy", "info.npy", "F.npy", "spks.npy", "Fneu.npy", "iscell.npy"]
        results = self.load_segmentation_results(output_dir_name=result_dir_name, to_load=results_to_export)

        # save the parameters that were used for the s3d run
        self.save_file(data=self.params, filename="s3d-params.npy", path=full_export_path)

        if export_frame_counts:
            # save the number of frames in each tiff file, and which directory they were in
            frames = self.load_frame_counts()
            self.save_file(data=frames, filename="frames.npy", path=full_export_path)
        for result in results.keys():
            data = results[result]
            # stats_small doesn't contain the neuropil coordinates,
            # which take up a lot of space and are kind of useless for further analysis
            if result == "stats_small.npy":
                result = "stats.npy"
            if result == "info.npy":
                if "all_params" not in data.keys():
                    # TODO remove this!!!!! just for backwards compatibility
                    data["all_params"] = self.params
            self.save_file(data=data, filename=result, path=full_export_path)
            self.log("Saved %s to %s" % (result, full_export_path), 2)

    def extract_and_deconvolve(
        self,
        patch_idx=0,
        mov=None,
        batchsize_frames=500,
        stats=None,
        offset=None,
        n_frames=None,
        stats_dir=None,
        iscell=None,
        ts=None,
        load_F_from_dir=False,
        parent_dir_name=None,
        save_dir=None,
        crop=True,
        mov_shape_tfirst=False,
    ):
        self.save_params()
        if stats_dir is None:
            stats_dir = self.get_patch_dir(patch_idx, parent_dir_name=parent_dir_name)
            stats, info = self.get_detected_cells(patch_idx, parent_dir_name=parent_dir_name)
            offset = (info["zs"], info["ys"], info["xs"])
        else:
            if stats is not None:
                if "stats.npy" not in os.listdir(stats_dir):
                    self.log("Saving provided stats.npy to %s" % stats_dir)
                    n.save(os.path.join(stats_dir, "stats.npy"), stats)
                else:
                    self.log(
                        "WARNING - overwriting with provided stats.npy in %s. Old one is in old_stats.npy" % stats_dir
                    )
                    old_stats = n.load(os.path.join(stats_dir, "stats.npy"), allow_pickle=True)
                    n.save(os.path.join(stats_dir, "old_stats.npy"), old_stats)
                    n.save(os.path.join(stats_dir, "stats.npy"), stats)
            else:
                stats = n.load(os.path.join(stats_dir, "stats.npy"), allow_pickle=True)

        # return stats
        if mov is None:
            if not mov_shape_tfirst:
                mov = self.get_registered_movie("registered_fused_data", "fused", edge_crop=False)
            else:
                mov = self.get_registered_movie("registered_fused_data", "fused", axis=0, edge_crop=False)
        if crop and self.params["svd_crop"] is not None:
            cz, cy, cx = self.params["svd_crop"]
            self.log("Cropping with bounds: %s" % (str(self.params["svd_crop"])))
            if mov_shape_tfirst:
                mov = mov[:, cz[0] : cz[1], cy[0] : cy[1], cx[0] : cx[1]]
            else:
                mov = mov[cz[0] : cz[1], :, cy[0] : cy[1], cx[0] : cx[1]]
        if ts is not None:
            if mov_shape_tfirst:
                mov = mov[ts[0] : ts[1]]
            else:
                mov = mov[:, ts[0] : ts[1]]
        self.log("Movie shape: %s" % (str(mov.shape)))
        if save_dir is None:
            save_dir = stats_dir
        if iscell is None:
            iscell = n.ones((len(stats), 2), int)
        if type(iscell) == str:
            if iscell[-4:] != ".npy":
                iscell += ".npy"
            iscell = n.load(os.path.join(stats_dir, iscell))
        if len(iscell.shape) < 2:
            iscell = iscell[:, n.newaxis]
        print(len(stats))
        assert iscell.shape[0] == len(stats)

        valid_stats = [stat for i, stat in enumerate(stats) if iscell[i, 0]]
        save_iscell = os.path.join(save_dir, "iscell_extracted.npy")
        self.log("Extracting %d valid cells, and saving cell flags to %s" % (len(valid_stats), save_iscell))
        stats = valid_stats
        # return stats
        n.save(save_iscell, iscell)
        # print(offset, batchsize_frames, n_frames)
        # return mov, stats
        if not load_F_from_dir:
            self.log("Extracting activity")
            F_roi, F_neu = ext.extract_activity(
                mov,
                stats,
                batchsize_frames=batchsize_frames,
                offset=offset,
                n_frames=n_frames,
                intermediate_save_dir=save_dir,
                mov_shape_tfirst=mov_shape_tfirst,
            )
            n.save(os.path.join(save_dir, "F.npy"), F_roi)
            n.save(os.path.join(save_dir, "Fneu.npy"), F_neu)
        else:
            F_roi = n.load(os.path.join(stats_dir, "F.npy"))
            F_neu = n.load(os.path.join(stats_dir, "Fneu.npy"))

        self.log("Deconvolving")
        F_sub = F_roi - F_neu * self.params.get("npil_coeff", 0.7)
        dcnv_baseline = self.params.get("dcnv_baseline", "maximin")
        dcnv_win_baseline = self.params.get("dcnv_win_baseline", 60)
        dcnv_sig_baseline = self.params.get("dcnv_sig_baseline", 10)
        dcnv_prctile_baseline = self.params.get("dcnv_prctile_baseline", 8)
        dcnv_batchsize = self.params.get("dcnv_batchsize", 3000)
        tau = self.params.get("tau", 1.3)
        F_sub = dcnv.preprocess(
            F_sub, dcnv_baseline, dcnv_win_baseline, dcnv_sig_baseline, self.params["fs"], dcnv_prctile_baseline
        )
        spks = dcnv.oasis(F_sub, batch_size=dcnv_batchsize, tau=tau, fs=self.params["fs"])

        self.log("Saving to %s" % save_dir)
        n.save(os.path.join(save_dir, "spks.npy"), spks)

        return self.get_traces(patch_dir=save_dir)

    def get_patch_dir(self, patch_idx=0, parent_dir_name="detection"):
        if type(patch_idx) == str:
            patch_str = patch_idx
        else:
            patch_str = "patch-%04d" % patch_idx
        patch_dir = self.make_new_dir(
            patch_str, parent_dir_name=parent_dir_name, dir_tag=parent_dir_name + "-" + patch_str
        )
        return patch_dir

    def load_patch_results(self, patch_idx=0, parent_dir_name="detection"):
        patch_dir = self.get_patch_dir(patch_idx, parent_dir_name)
        stats = n.load(os.path.join(patch_dir, "stats.npy"), allow_pickle=True)
        info = n.load(os.path.join(patch_dir, "info.npy"), allow_pickle=True).item()
        try:
            iscell = n.load(os.path.join(patch_dir, "iscell.npy"))
        except FileNotFoundError:
            iscell = n.ones((len(stats), 2), dtype=int)
            n.save(os.path.join(patch_dir, "iscell.npy"), iscell)
        return stats, info, iscell

    def combine_patches(
        self,
        patch_idxs,
        output_dir_path,
        info_use_idx=-1,
        save=True,
        deduplicate=True,
        extra_stats_keys=None,
        parent_dir_name="detection",
        max_roi_per_patch=None,
    ):

        if save:
            assert output_dir_path is not None
        stats = []
        iscells = []
        keep_stats_keys = [
            "idx",
            "threshold",
            "coords",
            "lam",
            "med",
            "peak_val",
            "npcoords",
            "patch_idx",
            "med_patch",
        ]
        if extra_stats_keys is not None:
            keep_stats_keys += extra_stats_keys

        for patch_idx in patch_idxs:
            stats_patch, info_patch, iscell = self.load_patch_results(patch_idx, parent_dir_name)
            if max_roi_per_patch is not None and len(stats_patch) > max_roi_per_patch:
                self.log(
                    "Clipping patch %d because it has %d ROIs, max is %d"
                    % (patch_idx, len(stats_patch), max_roi_per_patch),
                    2,
                )
                stats_patch = stats_patch[:max_roi_per_patch]
                iscell = iscell[:max_roi_per_patch]

            for stat in stats_patch:
                keep_stat = {}
                for key in keep_stats_keys:
                    if key in stat.keys():
                        keep_stat[key] = stat[key]
                stats.append(keep_stat)
            iscells.append(iscell)
            if info_use_idx is not None and patch_idx == patch_idxs[info_use_idx]:
                info = info_patch
        iscell = n.concatenate(iscells)

        if deduplicate:
            self.log("Deduplicating %d cells" % len(stats), 2)
            tic = time.time()
            stats, duplicate_cells = ext.prune_overlapping_cells(
                stats,
                self.params.get("detect_overlap_dist_thresh", 5),
                self.params.get("detect_overlap_lam_thresh", 0.5),
            )
            iscell = iscell[~duplicate_cells]
            self.log("Removed %d duplicate cells in %.2fs" % (duplicate_cells.sum(), time.time() - tic), 2)

        # stats = n.concatenate(stats)
        self.log("Combined %d patches, %d cells" % (len(patch_idxs), len(stats)))
        if not save:
            return stats, info, iscell
        else:
            self.log("Saving combined files to %s" % output_dir_path)
            n.save(os.path.join(output_dir_path, "stats.npy"), stats)
            self.log("Saved stats", 2)
            n.save(os.path.join(output_dir_path, "iscell.npy"), iscell)
            self.log("Saved iscell", 2)
            if info_use_idx is not None:
                n.save(os.path.join(output_dir_path, "info.npy"), info)
                self.log("Saved info (copied from patch) %d" % patch_idxs[info_use_idx], 2)
            return output_dir_path

    def get_detected_cells(self, patch=0, parent_dir_name="detection"):
        patch_dir = self.get_patch_dir(patch, parent_dir_name=parent_dir_name)
        stats = n.load(os.path.join(patch_dir, "stats.npy"), allow_pickle=True)
        info = n.load(os.path.join(patch_dir, "info.npy"), allow_pickle=True).item()
        return stats, info

    def get_traces(self, patch_idx=0, parent_dir_name="detection", patch_dir=None):
        if patch_dir is None:
            patch_dir = self.get_patch_dir(patch_idx, parent_dir_name=parent_dir_name)
        traces = {}
        for filename in ["F.npy", "Fneu.npy", "spks.npy"]:
            if filename in os.listdir(patch_dir):
                traces[filename[:-4]] = n.load(os.path.join(patch_dir, filename))
        return traces

    def get_registered_files(self, key="registered_fused_data", filename_filter="fused", sort=True):
        all_files = os.listdir(self.dirs[key])
        reg_files = [os.path.join(self.dirs[key], x) for x in all_files if x.startswith(filename_filter)]
        if sort:
            reg_files = sorted(reg_files)
        return reg_files

    def get_denoised_files(self):
        all_files = n.os.listdir(self.dirs["deepinterp"])
        reg_files = [os.path.join(self.dirs["deepinterp"], x) for x in all_files if x.startswith("dp")]
        return reg_files

    def get_iter_dirs(self, dir_tag="iters", sort=True):
        iters_dir = self.dirs[dir_tag]
        iter_dirs = [os.path.join(iters_dir, dir) for dir in os.listdir(iters_dir)]
        if sort:
            iter_dirs = sorted(iter_dirs)
        ret = []
        # print(iter_dirs)
        # return iter_dirs
        for dir in iter_dirs:
            if not os.path.isdir(dir):
                continue
            # print(os.listdir(dir))
            if "vmap.npy" in os.listdir(dir) or "vmap2.npy" in os.listdir(dir):
                ret.append(dir)
        return ret

    def load_iter_results(self, iter_idx, dir_tag="iters"):
        iter_dir = self.get_iter_dirs(dir_tag=dir_tag)[iter_idx]
        self.log("Loading from %s" % iter_dir)
        res = {}
        for filename in ["vmap", "max_img", "mean_img", "sum_img", "vmap2"]:
            if filename + ".npy" in os.listdir(iter_dir):
                res[filename] = n.load(os.path.join(iter_dir, filename + ".npy"), allow_pickle=True)
        return res

    def fuse_registered_movie(self, files=None, save=True, n_proc=8, delete_original=False, parent_dir=None):
        n_skip = self.params["n_skip"]
        if files is None:
            files = self.get_registered_files()
        __, xs = lbmio.load_and_stitch_full_tif_mp(self.tifs[0], channels=n.arange(1), get_roi_start_pix=True)
        centers = n.sort(xs)[1:]
        shift_xs = n.round(self.load_summary()["plane_shifts"][:, 1]).astype(int)
        if save:
            reg_fused_dir = self.make_new_dir("registered_fused_data", parent_dir_name=parent_dir)
        else:
            reg_fused_dir = ""
        if save:
            self.log("Saving to %s" % reg_fused_dir)
            self.save_params(copy_dir_tag="registered_fused_data")

        crop = self.params.get("fuse_crop", None)
        if crop is not None:
            self.log("Cropping: %s" % str(crop))
        # if you get an assertion error here with save=False in _get_more_data, assert left > 0
        # congratulations, you have run into a bug in Python itself!
        # https://bugs.python.org/issue34563, https://stackoverflow.com/questions/47692566/
        # the files are too big!
        if n_proc > 1:
            with Pool(n_proc) as p:
                fused_files = p.starmap(
                    fuse_and_save_reg_file,
                    [
                        (file, reg_fused_dir, centers, shift_xs, n_skip, crop, None, save, delete_original)
                        for file in files
                    ],
                )
        else:
            self.log("Single processor")
            fused_files = [
                fuse_and_save_reg_file(
                    file, reg_fused_dir, centers, shift_xs, n_skip, None, None, save, delete_original
                )
                for file in files
            ]
        if not save:
            # return fused_files
            fused_files = n.concatenate(fused_files, axis=1)
        return fused_files

    def svd_decompose_movie(self, svd_dir_tag, run_svd=True, end_batch=None, mov=None, mov_shape_tfirst=False):
        svd_dir = self.dirs[svd_dir_tag]
        self.save_params(copy_dir_tag=svd_dir_tag)

        if mov is None:
            if not mov_shape_tfirst:
                mov = self.get_registered_movie("registered_fused_data", "fused", edge_crop=False)
            else:
                mov = self.get_registered_movie("registered_fused_data", "fused", axis=0, edge_crop=False)
            self.log("Loaded mov of size %s" % str(mov.shape))
        if self.params.get("svd_crop", None) is not None:
            crop = self.params["svd_crop"]
            if not mov_shape_tfirst:
                mov = mov[crop[0][0] : crop[0][1], :, crop[1][0] : crop[1][1], crop[2][0] : crop[2][1]]
            else:
                mov = mov[:, crop[0][0] : crop[0][1], crop[1][0] : crop[1][1], crop[2][0] : crop[2][1]]
            self.log("Cropped to size %s" % str(mov.shape))
        if self.params.get("svd_time_crop", None) is not None:
            svd_time_crop = self.params.get("svd_time_crop", None)
            if not mov_shape_tfirst:
                mov = mov[:, svd_time_crop[0] : svd_time_crop[1]]
            else:
                mov = mov[svd_time_crop[0] : svd_time_crop[1]]
            self.log("Time-cropped to size %s" % str(mov.shape))

        if self.params.get("svd_pix_chunk") is None:
            self.params["svd_pix_chunk"] = n.product(self.params["svd_block_shape"]) // 2
        if self.params.get("svd_time_chunk") is None:
            self.params["svd_save_time_chunk"] = 4000
        if self.params.get("svd_save_time_chunk") is None:
            self.params["svd_save_comp_chunk"] = 400
        if self.params.get("svd_save_comp_chunk") is None:
            self.params["svd_comp_chunk"] = 100
        if self.params.get("n_svd_blocks_per_batch") is None:
            self.params["n_svd_blocks_per_batch"] = 16
        self.save_params(copy_dir_tag=svd_dir_tag)
        # return
        svd_info = svu.block_and_svd(
            mov,
            n_comp=self.params["n_svd_comp"],
            block_shape=self.params["svd_block_shape"],
            block_overlaps=self.params["svd_block_overlaps"],
            pix_chunk=self.params["svd_pix_chunk"],
            t_chunk=self.params["svd_time_chunk"],
            t_save_chunk=self.params["svd_save_time_chunk"],
            comp_chunk=self.params["svd_save_comp_chunk"],
            n_svd_blocks_per_batch=self.params["n_svd_blocks_per_batch"],
            log_cb=self.log,
            end_batch=end_batch,
            flip_shape=mov_shape_tfirst,
            svd_dir=svd_dir,
            run_svd=run_svd,
        )
        return svd_info

    # def get_subtracted_movie(self):
    #     mov_sub_paths = []
    #     for d in self.get_iter_dirs():
    #         if 'mov_sub.npy' in os.listdir(d):
    #             mov_sub_paths.append(os.path.join(d, 'mov_sub.npy'))

    #     mov_sub = utils.npy_to_dask(mov_sub_paths, axis=0)
    #     return mov_sub

    def get_registered_movie(
        self, key="registered_fused_data", filename_filter="fused", axis=1, edge_crop=False, edge_crop_npix=None
    ):
        paths = self.get_registered_files(key, filename_filter)
        mov_reg = utils.npy_to_dask(paths, axis=axis)
        if edge_crop:
            mov_reg = self.edge_crop_movie(mov_reg, edge_crop_npix=edge_crop_npix)

        self.mov_reg = mov_reg
        return mov_reg

    def get_subtracted_movie(self, key="mov_sub", parent_dir_name=None, filename_filter="mov_sub"):
        if parent_dir_name is not None:
            key = parent_dir_name + "-" + key
        paths = self.get_registered_files(key, filename_filter)
        self.mov_sub = utils.npy_to_dask(paths, axis=0)
        return self.mov_sub

    def edge_crop_movie(self, mov, edge_crop_npix=None):
        if edge_crop_npix is None:
            edge_crop_npix = self.params.get("edge_crop_npix", 0)
        if edge_crop_npix == 0:
            return mov
        self.log("Cropping the edges by %d pixels (accounting for plane shifts)" % edge_crop_npix)
        summary = self.summary
        if summary is None:
            summary = self.load_summary()
        nz, nt, ny, nx = mov.shape
        yt, yb, xl, xr = utils.get_shifted_plane_bounds(
            summary["plane_shifts"], ny, nx, summary["ypad"][0], summary["xpad"][0]
        )
        self.log(str(yt) + str(yb))
        self.log(str(xl) + str(xr))
        for i in range(nz):
            mov[i, :, : yt[i] + edge_crop_npix] = 0
            mov[i, :, yb[i] - edge_crop_npix :] = 0
            mov[i, :, :, : xl[i] + edge_crop_npix] = 0
            mov[i, :, :, xr[i] - edge_crop_npix :] = 0

        return mov

    def load_frame_counts(self):
        if "frames.npy" not in os.listdir(self.dirs["job_dir"]):
            self.save_frame_counts()
        return n.load(os.path.join(self.dirs["job_dir"], "frames.npy"), allow_pickle=True).item()

    def get_dir_frame_idxs(self, dir_idx):
        frames = self.load_frame_counts()
        idxs = n.where(frames["dir_ids"] == dir_idx)[0]
        st, en = idxs[0], idxs[-1]
        frame_start = frames["nframes"][:st].sum()
        frame_end = frames["nframes"][: en + 1].sum()
        return frame_start, frame_end

    def save_frame_counts(self):
        size_to_frames = {}
        nframes = []
        dir_ids = []
        for tif in self.tifs:
            dir_ids.append((tif.split(os.path.sep)[-2]))
            tifsize = int(os.path.getsize(tif))
            if tifsize in size_to_frames.keys():
                nframes.append(size_to_frames[tifsize])
            else:
                tf = tifffile.TiffFile(tif)
                nf = len(tf.pages) // self.params.get("n_ch_tif", 30)
                nframes.append(nf)
                size_to_frames[tifsize] = nf
                self.log(tif + " is %d frames and %d bytes" % (nf, tifsize))

        nframes = n.array(nframes)
        dir_ids = n.array(dir_ids)

        tosave = {"nframes": nframes, "dir_ids": dir_ids}
        self.frames = tosave
        n.save(os.path.join(self.dirs["job_dir"], "frames.npy"), tosave)

        return nframes, dir_ids

    def sweep_params(
        self,
        params_to_sweep,
        svd_info=None,
        mov=None,
        testing_dir_tag="sweep",
        n_test_iters=1,
        all_combinations=True,
        do_vmap=True,
        svs=None,
        us=None,
        test_parent_dir=None,
        delete_mov_sub=True,
    ):

        init_params = copy.deepcopy(self.params)
        testing_dir = self.make_new_dir(testing_dir_tag, parent_dir_name=test_parent_dir)
        sweep_summary_path = os.path.join(testing_dir, "sweep_summary.npy")
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
            combinations = n.stack([base_vals] * n_combs)
            cidx = 0
            for pidx in range(len(param_names)):
                for vidx in range(n_per_param[pidx]):
                    combinations[cidx][pidx] = param_vals_list[pidx][vidx]
                    cidx += 1
        assert len(combinations) == n_combs

        comb_strs = []
        comb_params = []
        comb_dir_tags = []
        comb_dirs = []
        for comb_idx, comb in enumerate(combinations):
            comb_param = copy.deepcopy(init_params)
            comb_str = "comb%05d-params" % comb_idx
            for param_idx, param in enumerate(param_names):
                param_value = comb[param_idx]
                if type(param_value) != str:
                    val_str = "%.03f" % param_value
                else:
                    val_str = param_value
                comb_str += "-%s_%s" % (param, val_str)
                comb_param[param] = param_value
            comb_dir_tag = testing_dir_tag + "-comb_%05d" % comb_idx
            comb_dir = self.make_new_dir(comb_dir_tag, parent_dir_name=testing_dir_tag)

            comb_params.append(comb_param)
            comb_dirs.append(comb_dir)
            comb_strs.append(comb_str)
            comb_dir_tags.append(comb_dir_tag)
        sweep_summary = {
            "init_params": init_params,
            "comb_strs": comb_strs,
            "comb_dir_tags": comb_dir_tags,
            "comb_params": comb_params,
            "comb_dirs": comb_dirs,
            "param_names": param_names,
            "combinations": combinations,
            "param_sweep_dict": params_to_sweep,
            "all_combinations": all_combinations,
            "complete": False,
        }
        n.save(sweep_summary_path, sweep_summary)
        self.log("Saving summary for %d combinations to %s" % (n_combs, sweep_summary_path))

        if do_vmap:
            vmaps = []
            for comb_idx in range(n_combs):
                comb_dir_tag = comb_dir_tags[comb_idx]
                comb_dir = comb_dirs[comb_idx]
                comb_str = comb_strs[comb_idx]
                self.log("Running combination %02d/%02d" % (comb_idx + 1, n_combs), 0, log_mem_usage=True)
                self.log("Summary dict size: %02d GB" % (sys.getsizeof(sweep_summary) / 1024**3))
                self.log("Combination params: %s" % comb_str, 2)
                self.log("Saving to tag %s at %s" % (comb_dir_tag, comb_dir), 2)
                self.params = comb_params[comb_idx]
                (vmap, mean_img, max_img), mov_sub_dir, iter_dir = self.calculate_corr_map(
                    mov=mov,
                    svd_info=svd_info,
                    parent_dir=comb_dir_tag,
                    iter_limit=n_test_iters,
                    update_main_params=False,
                    svs=svs,
                    us=us,
                )
                if delete_mov_sub:
                    self.log("Removing mov_sub from %s" % mov_sub_dir)
                    shutil.rmtree(mov_sub_dir)
                vmaps.append(vmap)
                sweep_summary["vmaps"] = vmaps
                sweep_summary["mean_img"] = mean_img
                sweep_summary["max_img"] = max_img
                n.save(sweep_summary_path, sweep_summary)
        sweep_summary["complete"] = True
        n.save(sweep_summary_path, sweep_summary)
        return sweep_summary

    def vis_vmap_sweep(self, summary):
        nz, ny, nx = summary["results"][0]["corrmap"].shape
        param_dict = summary["param_sweep_dict"]
        param_names = summary["param_names"]
        combinations = summary["combinations"]
        vmaps = [r["corrmap"] for r in summary["results"]]
        n_val_per_param = [len(param_dict[k]) for k in param_names]
        vmap_sweep = n.zeros(tuple(n_val_per_param) + (nz, ny, nx))
        print(n_val_per_param)
        print(vmap_sweep.shape)
        n_params = len(param_names)
        for cidx, combination in enumerate(combinations):
            param_idxs = [
                n.where(n.array(param_dict[param_names[pidx]]) == combination[pidx])[0][0] for pidx in range(n_params)
            ]
            vmap_sweep[tuple(param_idxs)] = vmaps[cidx]
        v = ui.napari.Viewer()
        v.add_image(summary["mean_img"], name="mean_img")
        v.add_image(summary["max_img"], name="max_img")
        v.add_image(vmap_sweep, name="Corrmap Sweep")
        v.dims.axis_labels = tuple(param_names + ["z", "y", "x"])
        return v

    def get_logged_mem_usage(self):
        mem_log_lines = []
        with open(self.logfile, "r") as logf:
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
                tstamp = datetime.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
                timestamps.append(tstamp)
            except:
                # print("Could not parse line %s" % line)
                continue

            tag = "Total Used: "
            num_len = 7
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx : num_idx + 7]
            num_float = float(num_str)
            used_mem.append(num_float)

            descriptors.append(line[25 : num_idx - len(tag)].strip())

            tag = "Swap Used: "
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx : num_idx + 7]
            num_float = float(num_str)
            used_swp.append(num_float)

            tag = "Virtual Used: "
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx : num_idx + 7]
            num_float = float(num_str)
            used_vrt.append(num_float)

            tag = "Virtual Available: "
            num_idx = line.find(tag) + len(tag)
            num_str = line[num_idx : num_idx + 7]
            num_float = float(num_str)
            avail_vrt.append(num_float)

        return timestamps, used_mem, used_swp, used_vrt, avail_vrt, descriptors

    def plot_memory_usage(self, show_descriptors_pctile=None):
        timestamps, used_mem, used_swp, used_vrt, avail_vrt, descriptors = self.get_logged_mem_usage()
        f, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

        ax = axs[0]
        ax.plot(timestamps, used_mem, label="Used")
        ax.plot(timestamps, used_swp, label="Swap")
        ax.plot(timestamps, used_vrt, label="Used (virtual)")
        ax.plot(timestamps, avail_vrt, label=("Available (virtual)"))
        ax.set_ylabel("Memory usage (GB)")
        ax.legend()

        ax = axs[1]
        deltas = n.diff(used_mem)
        ax.scatter(timestamps[1:], deltas)
        ax.set_ylabel("Change in memory usage (GB)")
        ax.set_xlabel("Timestamp")

        if show_descriptors_pctile is not None:
            top_deltas = n.where(deltas > n.percentile(deltas, show_descriptors_pctile))[0]
            for top_idx in top_deltas:
                ax.text(
                    timestamps[top_idx + 1],
                    deltas[top_idx],
                    descriptors[top_idx],
                    rotation=-45,
                    rotation_mode="anchor",
                )
                ax.scatter([timestamps[top_idx + 1]], [deltas[top_idx]], s=5, color="red")

        # plt.show()
        return f, axs

    # TODO add a non-rigid = False so can load rigid-only data
    def load_registration_results(self, offset_dir="registered_fused_data"):
        offset_files = self.get_registered_files(offset_dir, "offsets")
        n_offset_files = len(offset_files)
        summary = self.load_summary()
        nyb, nxb = summary["reference_params"]["block_size"]
        # nyb, nxb = summary['all_ops'][0]['nblocks'] #old method
        nz = len(self.params["planes"])

        first_file = n.load(offset_files[0], allow_pickle=True).item()
        keys = first_file.keys()
        results = {}
        for key in keys:
            results[key] = []

        for i in range(n_offset_files):
            offset = n.load(offset_files[i], allow_pickle=True).item()
            # print(i)
            for key in keys:
                results[key].append(offset[key])
            # print(offset.keys())
            # rigid_xs.append(offset['xmaxs_rr'])
            # rigid_ys.append(offset['ymaxs_rr'])
            # nonrigid_xs.append(offset['xmaxs_nr'].reshape(-1,nz, nyb, nxb))
            # nonrigid_ys.append(offset['ymaxs_nr'].reshape(-1,nz, nyb, nxb))

        return results

    def get_plane_shifts(self):
        summary = self.load_summary()
        return summary["plane_shifts"]

    def calculate_corr_map_old(
        self,
        mov=None,
        save=True,
        return_mov_filt=False,
        crop=None,
        svd_info=None,
        iter_limit=None,
        parent_dir=None,
        update_main_params=True,
        svs=None,
        us=None,
    ):
        self.save_params(copy_dir_tag=parent_dir, update_main_params=update_main_params)
        if self.summary is None:
            self.load_summary()
        mov_sub_dir_tag = "mov_sub"
        iter_dir_tag = "iters"
        if parent_dir is not None:
            mov_sub_dir_tag = parent_dir + "-" + mov_sub_dir_tag
            iter_dir_tag = parent_dir + "-iters"
            iter_dir = self.make_new_dir("iters", parent_dir_name=parent_dir, dir_tag=iter_dir_tag)
        mov_sub_dir = self.make_new_dir("mov_sub", parent_dir_name=parent_dir, dir_tag=mov_sub_dir_tag)
        n.save(os.path.join(mov_sub_dir, "params.npy"), self.params)
        self.log("Saving mov_sub to %s" % mov_sub_dir)
        if svd_info is not None:
            mov = svd_info
            self.log("Using SVD shortcut, loading entire V matrix to memory")
            self.log(
                "WARNING: if you encounter very large RAM usage during this run, use mov=svd_info instead of svd_info=svd_info. If it persists, reduce your batchsizes"
            )
            out = calculate_corrmap_from_svd(
                svd_info,
                params=self.params,
                log_cb=self.log,
                iter_limit=iter_limit,
                svs=svs,
                us=us,
                dirs=self.dirs,
                iter_dir_tag=iter_dir_tag,
                mov_sub_dir_tag=mov_sub_dir_tag,
                summary=self.summary,
            )
        else:
            if mov is None:
                mov = self.get_registered_movie("registered_fused_data", "fused", edge_crop=False)
            if crop is not None and svd_info is None:
                assert svd_info is None, "cant crop with svd - easy fix"
                self.params["detection_crop"] = crop
                self.save_params(copy_dir_tag="mov_sub", update_main_params=False)
                mov = mov[crop[0][0] : crop[0][1], :, crop[1][0] : crop[1][1], crop[2][0] : crop[2][1]]
                self.log("Cropped movie to shape: %s" % str(mov.shape))
            vmap, mean_img, max_img = calculate_corrmap(
                mov,
                self.params,
                self.dirs,
                self.log,
                return_mov_filt=return_mov_filt,
                save=save,
                iter_limit=iter_limit,
                iter_dir_tag=iter_dir_tag,
                mov_sub_dir_tag=mov_sub_dir_tag,
                summary=self.summary,
            )

        return (vmap, mean_img, max_img), mov_sub_dir, self.dirs[iter_dir_tag]

    def get_cwd(self):
        print(os.getcwd())
