# %% imports and definitions
from pathlib import Path

import napari
import numpy as np
import xarray as xr
from bioio import BioImage
from scipy.ndimage import gaussian_filter
from skimage.io import imsave

IN_DPATH = Path(__file__).parents[1] / "data" / "separate"
OUT_DPATH = Path(__file__).parents[1] / "data" / "separate" / "output"
OUT_DPATH.mkdir(parents=True, exist_ok=True)
COLOR_MAP = {"STAR RED": "red", "STAR GREEN": "green", "STAR ORANGE": "orange"}
GS_SIGMA = {"STAR RED": 1, "STAR GREEN": 1, "STAR ORANGE": 1}

# %% load data and show napari window
for obf_file in IN_DPATH.glob("*.obf"):
    obf_data = BioImage(obf_file)
    layers_data = []
    for scene in obf_data.scenes:
        scene_parts = scene.split("/")
        spn_name = "/".join(scene_parts[:-1])
        chn_name = scene_parts[-1]
        if chn_name not in COLOR_MAP:
            continue
        obf_data.set_scene(scene)
        im_xr = obf_data.xarray_data.drop_attrs().astype(float)
        if chn_name in GS_SIGMA:
            im_xr = xr.apply_ufunc(
                gaussian_filter,
                im_xr,
                input_core_dims=[["X", "Y"]],
                output_core_dims=[["X", "Y"]],
                kwargs={"sigma": GS_SIGMA[chn_name]},
            )
        layers_data.append((im_xr, chn_name))
    if layers_data:
        viewer = napari.Viewer(title=obf_file.stem)
        for im_xr, chn_name in layers_data:
            viewer.add_image(
                im_xr.data,
                name=chn_name,
                colormap=COLOR_MAP[chn_name],
                blending="additive",
            )
        napari.run()
        # After viewer is closed, read contrast limits and save multi-page TIFF
        for layer in viewer.layers:
            clim = layer.contrast_limits
            data = layer.data.squeeze()  # Shape: (z, height, width)
            dat_clip = np.clip(data, clim[0], clim[1])
            dat_norm = (dat_clip - clim[0]) / (clim[1] - clim[0]) * 255
            fpath = OUT_DPATH / f"{obf_file.stem}-{layer.name}.tiff"
            imsave(fpath, dat_norm.astype(np.uint8))
            print(
                f"Saved {fpath.name} with {data.shape[0]} z-slices, contrast limits {clim}"
            )
