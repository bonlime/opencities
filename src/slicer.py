"""Utils for slicing tiff image into windows"""
import os
import json
import time
import warnings
import subprocess
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
from skimage.io import imsave
import configargparse as argparse
from shapely.ops import cascaded_union
from scipy.ndimage.morphology import distance_transform_edt
import rio_tiler
from rio_tiler import main as rt_main
from rasterio.transform import from_bounds
from rasterio import features as rast_features
from pystac import Catalog, CatalogType, Item, Asset, LabelItem, Collection

## Fix issue with local imports
if __name__ == "__main__":
    from arg_parser import parse_args
else:
    from src.arg_parser import parse_args

warnings.filterwarnings("ignore", category=FutureWarning)


def geojson_to_squares(geojson_path, zoom_level=19, outfile=None, val_percent=0.15, random_val=True):
    """Turn Geojson with buildings annotations to non overlapping squares at `zoom_level`
    supermercado could only be run from comand line, so I'm using chaining of runs to simulate one long
    command
    Args: 
        geojson_path (str): path to geojson with building masks
        zoom_level (int): zoom level
        outfile (str): name of output file
        val_percent (float): percent of images for validation. split is by y coordinate.
    
    Returns:
        GeoPandas DF with tiles
    """

    outfile = outfile or f"z{zoom_level}tiles.geojson"
    kwargs = {"universal_newlines": True, "stdout": subprocess.PIPE}
    ps1 = subprocess.run(["cat", geojson_path], **kwargs)
    ps2 = subprocess.run(["supermercado", "burn", f"{zoom_level}"], input=ps1.stdout, **kwargs)
    ps3 = subprocess.run(["mercantile", "shapes"], input=ps2.stdout, **kwargs)
    ps4 = subprocess.run(["fio", "collect"], input=ps3.stdout, **kwargs)
    open(outfile, "w").write(ps4.stdout)
    tiles_df = gpd.read_file(outfile)
    tiles_df["xyz"] = tiles_df.id.apply(lambda x: list(eval(x)))
    # perform validation split
    assert 0 < val_percent < 1
    if random_val:
        tiles_df["dataset"] = np.random.rand(len(tiles_df)) > val_percent
        tiles_df["dataset"] = tiles_df["dataset"].map(lambda x: "train" if x else "val")
    else:
        split_y = np.percentile(
            tiles_df.xyz.apply(lambda x: x[1]), val_percent * 100) # lowest 15% by y coordinate
        tiles_df["dataset"] = tiles_df.xyz.apply(lambda x: "train" if x[1] > split_y else "val")

    return tiles_df


# preemptively fix and merge any invalid or overlapping geoms that would otherwise throw errors during the rasterize step.
# code from:
# https://gis.stackexchange.com/questions/271733/geopandas-dissolve-overlapping-polygons
# https://nbviewer.jupyter.org/gist/rutgerhofste/6e7c6569616c2550568b9ce9cb4716a3
def explode(gdf):
    """    
    Will explode the geodataframe's muti-part geometries into single 
    geometries. Each row containing a multi-part geometry will be split into
    multiple rows with single geometries, thereby increasing the vertical size
    of the geodataframe. The index of the input geodataframe is no longer
    unique and is replaced with a multi-index. 

    The output geodataframe has an index based on two columns (multi-index) 
    i.e. 'level_0' (index of input geodataframe) and 'level_1' which is a new
    zero-based index for each single part geometry per multi-part geometry
    
    Args:
        gdf (gpd.GeoDataFrame) : input geodataframe with multi-geometries
        
    Returns:
        gdf (gpd.GeoDataFrame) : exploded geodataframe with each single 
                                 geometry as a separate entry in the 
                                 geodataframe. The GeoDataFrame has a multi-
                                 index set to columns level_0 and level_1
        
    """
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: "geometry"})
    gdf_out = gdf2.merge(gdf.drop("geometry", axis=1), left_on="level_0", right_index=True)
    gdf_out = gdf_out.set_index(["level_0", "level_1"]).set_geometry("geometry")
    gdf_out.crs = gdf.crs
    return gdf_out


def cleanup_invalid_geoms(all_polys):
    """Fixed invalid geometries by merging them all together and then separating
    Args:
        all_polys (List or GeoSeries): list of separate polygons"""
    all_polys_merged = gpd.GeoDataFrame()
    all_polys_merged["geometry"] = gpd.GeoSeries(cascaded_union([p.buffer(0) for p in all_polys]))

    gdf_out = explode(all_polys_merged)
    gdf_out = gdf_out.reset_index()
    gdf_out.drop(columns=["level_0", "level_1"], inplace=True)
    all_polys = gdf_out["geometry"]
    return all_polys


def save_tile_img(tif_path, xyz, tile_size, save_path="", prefix=""):
    x, y, z = xyz
    tile, mask = rt_main.tile(tif_path, x, y, z, tilesize=tile_size)
    imsave(f"{save_path}/{prefix}{z}_{x}_{y}.png", np.moveaxis(tile, 0, 2), check_contrast=False)

def get_signed_distance_transform(mask, border=10):
    """turns mask in to SDT"""
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    if mask.max() == 1:
        mask *= 255 # need mask in [0, 255]
    pos = np.clip(distance_transform_edt(mask), 0, border) / border
    neg = np.clip(distance_transform_edt(255 - mask), 0, border) / border
    res =  pos - neg # in [-1, 1]
    res = ((res + 1) * 127.5).astype(np.uint8) # in [0, 255]
    return res

def save_tile_mask(labels_poly, tile_poly, xyz, tile_size, save_path="", prefix="", border_thickness=20):
    x, y, z = xyz
    tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size)
    # get poly inside the tile
    cropped_polys = [poly for poly in labels_poly if poly.intersects(tile_poly)]
    cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs=4326)
    # TODO: do manually because I don't like their definition of contact
    df = cropped_polys_gdf
    feature_list = list(zip(df["geometry"], [255] * len(df)))
    if len(feature_list) == 0:  # if no buildings return zero mask
        mask = np.zeros((tile_size, tile_size), np.uint8)
    else:
        mask = rast_features.rasterize(
            shapes=feature_list, out_shape=(args.tile_size, args.tile_size), transform=tfm
        )
    dist = get_signed_distance_transform(mask, border_thickness)
    im = np.stack([mask, dist, dist], axis=2)
    imsave(f"{save_path}/{prefix}{z}_{x}_{y}.png", im, check_contrast=False)


def pool_wrapper(idx_tile):
    idx, tile = idx_tile
    dataset = tile["dataset"]
    tile_poly = tile["geometry"]
    try:
        save_tile_img(
            tiff_path,
            tile["xyz"],
            args.tile_size,
            save_path=f"data/{args.suffix}-images-{args.tile_size}",
            prefix=f"{area_name}_{name}_{dataset}_",
        )

        save_tile_mask(
            all_polys,
            tile_poly,
            tile["xyz"],
            args.tile_size,
            save_path=f"data/{args.suffix}-masks-{args.tile_size}",
            prefix=f"{area_name}_{name}_{dataset}_",
        )

    except rio_tiler.errors.TileOutsideBounds:
        # don't need print
        pass  # some tiles are outside of image bounds we need to catch this errors


if __name__ == "__main__":

    start_time = time.time()
    args = parse_args()
    cols = {cols.id:cols for cols in Catalog.from_file(args.data_path + 'catalog.json').get_children()}

    ## Prepare data folders
    args.suffix = "tier_1" if "tier_1" in args.data_path else "tier_2" 
    Path("data").mkdir(exist_ok=True)
    Path(f"data/{args.suffix}-images-{args.tile_size}").mkdir(exist_ok=True)
    Path(f"data/{args.suffix}-masks-{args.tile_size}").mkdir(exist_ok=True)

    # Iterate over different areas
    print(
        f"Slicing images with zoom level={args.zoom_level}, tile size={args.tile_size}, border={args.border_thickness} and {args.val_percent} val split"
    )
    for area_name, area_data in cols.items():
        print(f"\nProcessing area: {area_name}")
        area_labels = [i for i in area_data.get_all_items() if "label" in i.id]
        area_labels_jsons = [i.make_asset_hrefs_absolute().assets["labels"].href for i in area_labels]
        area_images = [i for i in area_data.get_all_items() if "label" not in i.id]
        area_images_tiffs = [i.make_asset_hrefs_absolute().assets["image"].href for i in area_images]
        area_id_names = [i.id for i in area_images]
        # iterate over different tiffs inside one area
        for name, geojson_path, tiff_path in zip(area_id_names, area_labels_jsons, area_images_tiffs):
            print(f"\tProcessing id: {name}")
            labels_gdf = gpd.read_file(geojson_path)

            # some tiff jsons have type "MultiPolygon" which is not supported. just skip them
            tiff_json = json.load(open(tiff_path[:-3] + "json"))
            if tiff_json["geometry"]["type"] == "MultiPolygon":
                print("\t\tSkipped because MultiPolygon")
                continue

            # Get tiles
            tiles_gdf = geojson_to_squares(
                tiff_path[:-3] + "json",
                zoom_level=args.zoom_level,
                val_percent=args.val_percent,
                outfile="/tmp/tiles.geojson",
                random_val=args.random_val,
            )

            # get not overlappping polygons
            all_polys = cleanup_invalid_geoms(labels_gdf.geometry)
            # use multiprocessing to speedup chips generation
            pbar = tqdm(leave=False, total=len(tiles_gdf), ncols=0, ascii=True)
            with Pool() as pool:
                for _ in pool.imap_unordered(pool_wrapper, tiles_gdf.iterrows()):
                    pbar.update()
            pbar.close()
            # break
        # break # FOR DEBUG
    m = (time.time() - start_time) / 60
    print(f"Total time: {m:.1f}m")
