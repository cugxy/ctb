import os, math
import gdal
import numpy as np


def tile_level(z):
    assert(z >= 0)
    x = 2 ** (z+1)
    y = 2 ** z
    xyz = [x, y, z]
    return xyz


def get_tif_tile_bounds(filename, zoom):
    '''
    根据 cesium 层级分块，读取 tif，利用 zoom 求得每一块的 gps 包围盒
    :param filename: tif 文件路径
    :param zoom: 层级
    :return: bounds [[min_lng, min_lat, max_lng, max_lat], ...]
    '''
    if not os.path.exists(filename):
        return None
    data_set = gdal.Open(filename)
    if data_set is None:
        return None
    x_size, y_size = data_set.RasterXSize, data_set.RasterYSize
    x0, dx, _, y0, _, dy = data_set.GetGeoTransform()
    lng_min = x0
    lat_max = y0
    lng_max = x0 + dx * x_size
    lat_min = y0 + dy * y_size

    lng_begine = -180.0
    lng_end = 180.0
    lat_begine = -90.0
    lat_end = 90.0
    xyz = tile_level(zoom)
    x = xyz[0]
    y = xyz[1]
    lng_span = (lng_end - lng_begine) / x
    lat_span = (lat_end - lat_begine) / y

    tile_x_min = math.floor((lng_min - lng_begine) / lng_span)
    tile_y_min = math.floor((lat_min - lat_begine) / lat_span)
    tile_x_max = math.ceil((lng_max - lng_begine) / lng_span)
    tile_y_max = math.ceil((lat_max - lat_begine) / lat_span)
    result = []
    for _x in range(tile_x_min, tile_x_max + 1):
        for _y in range(tile_y_min, tile_y_max + 1):
            bound = [_x * lng_span + lng_begine,
                     _y * lat_span + lat_begine,
                     _x * lng_span + lng_span + lng_begine,
                     _y * lat_span + lat_span + lat_begine]
            result.append(bound)
    return result

def get_z_in_bound(data_set, bound):
    if data_set is None:
        print('no data_set')
        return None
    band = data_set.GetRasterBand(1)
    if band is None:
        print('no band')
        return None
    x_size, y_size = data_set.RasterXSize, data_set.RasterYSize
    x0, dx, _, y0, _, dy = data_set.GetGeoTransform()
    x_min = round((bound[0] - x0) / dx)
    x_begine = x_min
    if x_begine > x_size:
        return None
    if x_begine < 0:
        x_begine = 0
    y_min = round((bound[3] - y0) / dy)
    y_begine = y_min
    if y_begine > y_size:
        return None
    if y_begine < 0:
        y_begine = 0
    x_max = round((bound[2] - x0) / dx)
    x_end = x_max
    if x_end < 0:
        return None
    if x_end > x_size:
        x_end = x_size
    y_max = round((bound[1] - y0) / dy)
    y_end = y_max
    if y_end < 0:
        return None
    if y_end > y_size:
        y_end = y_size
    z = band.ReadAsArray(x_begine, y_begine, x_end - x_begine, y_end - y_begine)
    return z


def get_intersect_block(filename_low, filename_height, zoom):
    '''
    在低精度 tif 中，找到高精度 tif 所影响的地形块，并修改受影响值 生成 terrain
    :param filename: tif 文件路径
    :param zoom: 层级
    '''
    if (not os.path.exists(filename_low)) or (not os.path.exists(filename_height)):
        return None
    data_set_low = gdal.Open(filename_low)
    if data_set_low is None:
        print('Read %s failed' %filename_low)
        return None
    data_set_height = gdal.Open(filename_height)
    if data_set_height is None:
        print('Read %s failed' % filename_height)
        return None

    band_low = data_set_low.GetRasterBand(1)
    if band_low is None:
        print('Read %s band failed' % filename_low)
        return None
    x_size_low, y_size_low = data_set_low.RasterXSize, data_set_low.RasterYSize
    x0_low, dx_low, _, y0_low, _, dy_low = data_set_low.GetGeoTransform()

    band_height = data_set_height.GetRasterBand(1)
    if band_height is None:
        print('Read %s band failed' % filename_height)
        return None
    x_size_height, y_size_height = data_set_height.RasterXSize, data_set_height.RasterYSize
    x0_height, dx_height, _, y0_height, _, dy_height = data_set_height.GetGeoTransform()

    # 获取低精度 tif 中 高精度 tif 所影响的 地形块
    bounds = get_tif_tile_bounds(filename_height, zoom)
    # [[min_lng, min_lat, max_lng, max_lat], ...]
    if bounds == []:
        return
    zoom_size_low = round((bounds[0][2]-bounds[0][0]) / dx_low)
    for bound in bounds:
        x_min_low = round((bound[0] - x0_low) / dx_low)
        x_begine_low = x_min_low
        if x_begine_low > x_size_low:
            continue
        if x_begine_low < 0:
            x_begine_low = 0
        y_min_low = round((bound[3] - y0_low) / dy_low)
        y_begine_low = y_min_low
        if y_begine_low > y_size_low:
            continue
        if y_begine_low < 0:
            y_begine_low = 0
        x_max_low = round((bound[2] - x0_low) / dx_low)
        x_end_low = x_max_low
        if x_end_low < 0:
            continue
        if x_end_low > x_size_low:
            x_end_low = x_size_low
        y_max_low = round((bound[1] - y0_low) / dy_low)
        y_end_low = y_max_low
        if y_end_low < 0:
            continue
        if y_end_low > y_size_low:
            y_end_low = y_size_low
        z_low = band_low.ReadAsArray(x_begine_low, y_begine_low, x_end_low - x_begine_low, y_end_low - y_begine_low)

        x_min_height = round((bound[0] - x0_height) / dx_height)
        x_begine_height = x_min_height
        if x_begine_height > x_size_height:
            continue
        if x_begine_height < 0:
            x_begine_height = 0
        y_min_height = round((bound[3] - y0_height) / dy_height)
        y_begine_height = y_min_height
        if y_begine_height > y_size_height:
            continue
        if y_begine_height < 0:
            y_begine_height = 0
        x_max_height = round((bound[2] - x0_height) / dx_height)
        x_end_height = x_max_height
        if x_end_height < 0:
            continue
        if x_end_height > x_size_height:
            x_end_height = x_size_height
        y_max_height = round((bound[1] - y0_height) / dy_height)
        y_end_height = y_max_height
        if y_end_height< 0:
            continue
        if y_end_height > y_size_height:
            y_end_height = y_size_height
        z_height = band_height.ReadAsArray(x_begine_height,
                                           y_begine_height,
                                           x_end_height - x_begine_height,
                                           y_end_height - y_begine_height,
                                           round(zoom_size_low * (x_end_height - x_begine_height) / (x_max_height - x_min_height)),
                                           round(zoom_size_low * (y_end_height - y_begine_height) / (y_max_height - y_min_height)))


        pass





    pass


if __name__ == '__main__':
    filename_low = r'E:\xy\doc\dem\cliped.tif'
    filename = r'E:\xy\doc\dem\dem.tif'
    zoom = 13
    r = get_intersect_block(filename_low, filename, zoom)
    pass