from dash import Dash, html
import imageio
from dash_slicer import VolumeSlicer
from manager import self_ip, app_port
import numpy as n

app = Dash(__name__, update_title=None)

# vol = imageio.volread("imageio:stent.npz")
vol = n.load("/mnt/md0/runs/s3d-AH012_2024-08-09_1-2-3-4-5-6-7-8-9-10/summary/ref_img_3d.npy")
slicer = VolumeSlicer(app, vol)
slicer.graph.config["scrollZoom"] = True

app.layout = html.Div([slicer.graph, slicer.slider, *slicer.stores])


if __name__ == "__main__":
    app.run(debug=True, dev_tools_props_check=False, host=self_ip, port=app_port)
