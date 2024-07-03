


class IO:
    """
    A class that handles all operations related to data loading and preprocessing for suite3d.

    This class makes data-loading and preprocessing easy. You just have to set the parameters
    of the job class that it takes as input, then this class will handle all the rest. The goal
    is to make the interface of data-loading as straightforward as possible, with minimal need 
    to consider implementation details related to the particular requirements of the job.

    An instance of this class can be passed as an argument to other modules to make sure data
    loading is handled in a consistent and easy-to-use way wherever relevant.
    """
    def __init__(self, job):
        # By containing a reference to the job object, we can easily access all relevant
        # parameters related to data-loading without having to remember all the kwargs. 
        self.job = job 

    def _update_prms(self, **parameters):
        """
        A helper function that updates the job parameters with the provided parameters. 
        """
        use_params = self.job.params.copy() # get default parameters
        use_params.update(parameters) # update with any provided in kwargs
        return use_params # return the parameters intended for use right now

    def load_tifs(self, paths, planes, **parameters):
        """
        A central mechanism for loading tif files. This function is meant to be called
        every single time tifs are ever loaded, and it will handle all the necessary
        steps to ensure that the tifs are loaded correctly depending on the job's 
        parameters (lbm or not), any "local" parameters (like the ones typically passed
        as kwargs to ``lbmio.load_and_stitch_tifs()``), and anything else that is needed.
        """
        # example use of _update_prms to get the parameters to use for this call
        params = self._update_prms(**parameters)
        

    def load_accessory_data(self, *args, **kwargs):
        """
        Sometimes information related to IO needs to be loaded that is pulled from extra
        outputs of the ``lbmio.load_and_stitch_tifs()`` function. This function (and maybe
        a few others) is/are meant to be called in place of those for efficient processing
        and clear responsibility.

        For example, ``init_pass.run_init_pass()`` calls the following line:
        __, xs = lbmio.load_and_stitch_full_tif_mp(..., get_roi_start_pix=True, ...)

        Which is used to retrieve just "xs"=the starting offsets for each imaging ROI. 
        
        This can be replaced with a new "load accesory data" function that is called
        for this purpose -- depending on preference we can either have it run processing 
        again or potentially cache certain results that are known to be needed during an
        initial call to ``load_tifs()``. 
        """

    def all_supporting_functions(self, *args, **kwargs):
        """
        A collection of functions that are used to support the main functions of this class.
        These functions are meant to be called by the main functions, and should not be called
        directly by the user.

        Examples:
        load_and_stitch_full_tif_mp
        stitch_rois_fast
        etc...
        """
