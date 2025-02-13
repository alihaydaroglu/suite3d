import ipywidgets as ipyw
import numpy as n
from matplotlib import pyplot as plt
from suite3d.io.tiff_utils import show_tif
from IPython.display import display


# https://github.com/mohakpatel/ImageSliceViewer3D/blob/master/ImageSliceViewer3D.ipynb
class ImageSliceViewer3D:
    """
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(
        self,
        volume,
        figsize=(8, 8),
        cmap="Greys_r",
        vminmax=None,
        overlay=None,
        z0=None,
        **kwargs,
    ):
        self.volume = volume
        self.overlay = overlay
        self.figsize = figsize
        self.cmap = cmap
        self.kwargs = kwargs

        if vminmax is None:
            self.v = n.percentile(volume.flatten(), 0.5), n.percentile(
                volume.flatten(), 99.5
            )
        else:
            self.v = vminmax

        # self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)

        # Call to select slice plane
        ipyw.interact(
            self.view_selection,
            view=ipyw.RadioButtons(
                options=["x-y", "y-z", "z-x"],
                value="x-y",
                description="Slice plane selection:",
                disabled=False,
                style={"description_width": "initial"},
            ),
        )

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z": [1, 2, 0], "z-x": [2, 0, 1], "x-y": [0, 1, 2]}
        self.vol = n.transpose(self.volume, orient[view])
        if self.overlay is not None:
            self.overlay_vol = n.transpose(self.overlay, orient[view] + [3])
        print(self.vol.shape)
        maxZ = self.vol.shape[0] - 1

        # Call to view a slice within the selected slice plane
        ipyw.interact(
            self.plot_slice,
            z=ipyw.IntSlider(
                min=0,
                max=maxZ,
                value=int(maxZ // 2),
                step=1,
                continuous_update=False,
                description="Image Slice:",
            ),
            show_overlay=True,
        )

    def plot_slice(self, z, show_overlay=True):
        display(f"UPDATING to z: {z}")
        # Plot slice for the given plane and slice
        __, ax, __ = show_tif(
            self.vol[z, :, :],
            cmap=self.cmap,
            vminmax=self.v,
            figsize=self.figsize,
            **self.kwargs,
        )

        if show_overlay and self.overlay is not None:
            show_tif(self.overlay_vol[z], ax=ax)
        # plt.show()
        # self.fig = plt.figure(figsize=self.figsize)
        # plt.imshow(
        #     self.vol[:, :, z],
        #     cmap=plt.get_cmap(self.cmap),
        #     vmin=self.v[0],
        #     vmax=self.v[1],
        # )
