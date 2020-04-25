import numpy as np
import cv2
import math
import numpy as np
# TODO: Convert to numpy broadcasting method
# WARNING: Only work for zoom = 18
n = 2 ** 18


def deg2num(lat_deg, lon_deg):
    lat_rad = math.radians(lat_deg)
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile):
    # n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def point_in_map_to_lat_lon(xtile, ytile, x, y, tile_size=512):
    "Map (x, y) into (lat, lon)"
    xtile = xtile + (x / tile_size)
    ytile = ytile + (y / tile_size)
    return num2deg(xtile, ytile)


def deg2wh(lat_deg, lon_deg):
    """Map lat, lon, zoom to x, y inside a tile"""
    lat_rad = math.radians(lat_deg)
    xtile = int(((lon_deg + 180.0) / 360.0 * n) % 1 * 512)
    ytile = int(((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n) % 1 * 512)
    return (xtile, ytile)


def batch_num2deg(xtiles, ytiles):
    """Faster with numpy broadcasting"""
    lon_degs = xtiles / n * 360.0 - 180.
    lat_rads = np.arctan(
        np.sinh(
            np.pi * (1 - 2 * ytiles / n)
        )
    )
    lat_degs = np.rad2deg(lat_rads)
    return lat_degs, lon_degs


def batch_boxes2deg(boxes, xtile, ytile, tile_size=512):
    """Faster `point_in_map_to_lat_lon` with numpy broadcasting"""
    boxes = np.array(boxes)
    xs = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int)
    ys = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int)
    xtiles = (xs / tile_size) + xtile
    ytiles = (ys / tile_size) + ytile
    return xs, ys, batch_num2deg(xtiles, ytiles)
