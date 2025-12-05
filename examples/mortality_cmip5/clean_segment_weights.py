import pandas as pd


IN_WEIGHTS_FILE = "gs://impactlab-data-scratch/brews/agglomerated-world-new_BCSD_grid_segment_weights_area_pop.csv"
OUT_ZARR = "gs://impactlab-data-scratch/brews/clean_segment_weights.zarr"


sw = pd.read_csv(
    IN_WEIGHTS_FILE,
)
sw["pix_cent_x"] = (sw["pix_cent_x"] + 180) % 360 - 180
sw = sw.to_xarray().rename_vars(
    {"pix_cent_x": "lon", "pix_cent_y": "lat", "hierid": "region", "popwt": "weight"}
)
sw.to_zarr(OUT_ZARR, mode="w")
print(f"Written to {OUT_ZARR}")
