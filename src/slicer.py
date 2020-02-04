"""Utils for slicing tiff image into windows"""
# Many import are not used
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import solaris as sol
from pathlib import Path
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from pystac import (Catalog, CatalogType, Item, Asset, LabelItem, Collection)
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from rio_tiler import main as rt_main
import skimage
from tqdm import tqdm
import os
import subprocess
from multiprocessing import Pool
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def geojson_to_squares(geojson_path, zoom_level=19, outfile=None, val_percent=0.15):
    """Turn Geojson with buildings annotations to non overlapping squares at `zoom_level`
    supermercado could only be run from comand line, so I'm using chaining of runs to simulate one long
    command
    Args: 
        geojson_path (str): path to geojson with building masks
        zoom_level (int): zoom level
        outfile (str): name of output file
        val_percent (float): percent of images for validation. split is by y coordinate.
    """
    outfile = outfile or f"z{zoom_level}tiles.geojson"
    # if not os.path.exists(outfile):
    kwargs = {"universal_newlines":True, "stdout": subprocess.PIPE}
    ps1 = subprocess.run(["cat", geojson_path], **kwargs)
    ps2 = subprocess.run(["supermercado", "burn", f"{zoom_level}"], input=ps1.stdout, **kwargs)
    ps3 = subprocess.run(["mercantile", "shapes"], input=ps2.stdout, **kwargs)
    ps4 = subprocess.run(["fio", "collect"], input=ps3.stdout, **kwargs)
    open(outfile, "w").write(ps4.stdout)
    tiles_df = gpd.read_file(outfile)
    tiles_df["xyz"] = tiles_df.id.apply(lambda x: list(eval(x)))
    # perform validation split
    assert 0 < val_percent < 1
    split_y = np.percentile(tiles_df.xyz.apply(lambda x: x[1]), val_percent * 100) # lowest 15% by y coordinate
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
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf_out.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out

def cleanup_invalid_geoms(all_polys):
    """Fixed invalid geometries by merging them all together and then separating
    Args:
        all_polys (List or GeoSeries): list of separate polygons"""
    all_polys_merged = gpd.GeoDataFrame()
    all_polys_merged['geometry'] = gpd.GeoSeries(cascaded_union([p.buffer(0) for p in all_polys]))

    gdf_out = explode(all_polys_merged)
    gdf_out = gdf_out.reset_index()
    gdf_out.drop(columns=['level_0','level_1'], inplace=True)
    all_polys = gdf_out['geometry']
    return all_polys


def save_tile_img(tif_path, xyz, tile_size, save_path='', prefix='', display=False):
    x,y,z = xyz
    tile, mask = rt_main.tile(tif_path, x,y,z, tilesize=tile_size)
    skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}.png',np.moveaxis(tile,0,2), check_contrast=False)


def save_tile_mask(labels_poly, tile_poly, xyz, tile_size, save_path='', prefix='', display=False):
    x,y,z = xyz
    tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size) 
    # get poly inside the tile
    cropped_polys = [poly for poly in labels_poly if poly.intersects(tile_poly)]
    cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs=4326)

    fbc_mask = sol.vector.mask.df_to_px_mask(
        df=cropped_polys_gdf,
        channels=['footprint', 'boundary', 'contact'],
        affine_obj=tfm, shape=(tile_size,tile_size),
        boundary_width=5, boundary_type='inner', contact_spacing=5, meters=True
    )
    skimage.io.imsave(f'{save_path}/{prefix}{z}_{x}_{y}_mask.png',fbc_mask, check_contrast=False) 

def pool_wrapper(idx_tile):
    idx, tile = idx_tile
    dataset = tile['dataset']
    tile_poly = tile['geometry']
    save_tile_img(tiff_path, tile['xyz'], TILE_SIZE, save_path=IMGS_PATH, prefix=f'{area_name}_{name}_{dataset}_')
    save_tile_mask(all_polys, tile_poly, tile['xyz'], TILE_SIZE, save_path=MASKS_PATH,prefix=f'{area_name}_{name}_{dataset}_')

if __name__ == "__main__":
    DATA_ROOT_PATH = "/home/zakirov/datasets/opencities/train_tier_1/"
    train1_cat = Catalog.from_file(DATA_ROOT_PATH + 'catalog.json')
    cols = {cols.id:cols for cols in train1_cat.get_children()}
    ZOOM_LEVEL=19
    TILE_SIZE=512
    VAL_PERCENT=0.15
    # prepare data folders
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    IMGS_PATH = data_dir/f'images-{TILE_SIZE}'
    MASKS_PATH = data_dir/f'masks-{TILE_SIZE}'
    IMGS_PATH.mkdir(exist_ok=True)
    MASKS_PATH.mkdir(exist_ok=True)
    # iterate over different areas
    for area_name, area_data in cols.items():
        print(f"\nProcessing area: {area_name}")
        area_labels = [i for i in area_data.get_all_items() if 'label' in i.id]
        area_labels_jsons = [i.make_asset_hrefs_absolute().assets['labels'].href for i in area_labels]
        area_images = [i for i in area_data.get_all_items() if 'label' not in i.id]
        area_images_tiffs = [i.make_asset_hrefs_absolute().assets['image'].href for i in area_images]
        area_id_names = [i.id for i in area_images]
        # iterate over different tiffs inside one area
        for name, geojson_path, tiff_path in zip(area_id_names, area_labels_jsons, area_images_tiffs):
            print(f"\tProcessing id: {name}")
            labels_gdf = gpd.read_file(geojson_path)
            # get tiles
            tiles_gdf = geojson_to_squares(geojson_path, zoom_level=ZOOM_LEVEL, val_percent=VAL_PERCENT)
            # get not overlappping polygons
            all_polys = cleanup_invalid_geoms(labels_gdf.geometry)
            # use multiprocessing to speedup chips generation
            pbar = tqdm(leave=False, total=len(tiles_gdf))
            with Pool() as pool:
                for _ in pool.imap_unordered(pool_wrapper, tiles_gdf.iterrows()):
                    pbar.update()
            pbar.close()
            # break
        break # FOR DEBUG

