import os, math
import gdal
import numpy as np
import cv2
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from quantized_mesh_tile.terrain import TerrainTile
from quantized_mesh_tile.global_geodetic import GlobalGeodetic
from quantized_mesh_tile.topology import TerrainTopology


def tile_level(z):
    assert(z >= 0)
    x = 2 ** (z+1)
    y = 2 ** z
    xyz = [x, y, z]
    return xyz


def get_tif_tile_bounds(filename, output_dir, zoom):
    """
    根据 cesium 层级分块，读取 tif，利用 zoom 求得每一块的 gps 包围盒
    :param filename: tif 文件路径
    :param zoom: 层级
    :return: bounds [[min_lng, min_lat, max_lng, max_lat], ...]
    """
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
    result = {}

    for _x in range(tile_x_min, tile_x_max + 1):
        for _y in range(tile_y_min, tile_y_max + 1):
            fname = os.path.join(output_dir, str(zoom))
            fname = os.path.join(fname, str(_x))
            fname = os.path.join(fname, str(_y))
            bound = [_x * lng_span + lng_begine,
                     _y * lat_span + lat_begine,
                     _x * lng_span + lng_span + lng_begine,
                     _y * lat_span + lat_span + lat_begine]
            result[fname] = bound
    return result


def get_intersect_block(filename_low, filename_height, output_dir, zoom, end_zoom):
    """
    在低精度 tif 中，找到高精度 tif 所影响的地形块，并修改受影响值 生成 terrain
    :param filename: tif 文件路径
    :param zoom: 层级
    """
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

    for _zoom in range(zoom, end_zoom + 1):
        # 获取高精度 tif 所影响的 地形块
        bounds = get_tif_tile_bounds(filename_height, output_dir, _zoom)
        # [[min_lng, min_lat, max_lng, max_lat], ...]
        if bounds == {}:
            continue
        _, value = next(iter(bounds.items()))
        zoom_size_low = round((value[2]-value[0]) / dx_low * (2 ** (_zoom - zoom)))
        for fname in bounds:
            bound = bounds[fname]
            x_min_low = round((bound[0] - x0_low) / dx_low)
            y_min_low = round((bound[3] - y0_low) / dy_low)
            x_max_low = round((bound[2] - x0_low) / dx_low)
            y_max_low = round((bound[1] - y0_low) / dy_low)

            # 读取低精度 地形 要求低精度地形完整包含整块， 读取时 添加 buffer 防止平滑处理时边界错位
            buf_size = 8
            buf_size_low = round(buf_size / zoom_size_low * (x_max_low - x_min_low))
            x_begine_low = x_min_low - buf_size_low
            if x_begine_low > x_size_low or x_begine_low < 0:
                print('Low-precision terrain does not completely cover %s' % fname)
                continue
            y_begine_low = y_min_low - buf_size_low
            if y_begine_low > y_size_low or y_begine_low < 0:
                print('Low-precision terrain does not completely cover %s' % fname)
                continue
            x_end_low = x_max_low + buf_size_low
            if x_end_low < 0 or x_end_low > x_size_low:
                print('Low-precision terrain does not completely cover %s' % fname)
                continue
            y_end_low = y_max_low + buf_size_low
            if y_end_low < 0 or y_end_low > y_size_low:
                print('Low-precision terrain does not completely cover %s' % fname)
                continue
            z_low = band_low.ReadAsArray(x_begine_low,
                                         y_begine_low,
                                         x_end_low - x_begine_low,
                                         y_end_low - y_begine_low,
                                         zoom_size_low + buf_size * 2,
                                         zoom_size_low + buf_size * 2).astype('f4')
            # z_low 处理成 平滑处理
            # z_low = cv2.blur(z_low,(7, 7))
            z_low = z_low[buf_size:-buf_size, buf_size:-buf_size]

            # 读取 高精度 地形
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
            zoom_size_x = round(zoom_size_low * (x_end_height - x_begine_height) / (x_max_height - x_min_height))
            zoom_size_y = round(zoom_size_low * (y_end_height - y_begine_height) / (y_max_height - y_min_height))
            z_height = band_height.ReadAsArray(x_begine_height,
                                               y_begine_height,
                                               x_end_height - x_begine_height,
                                               y_end_height - y_begine_height,
                                               zoom_size_x,
                                               zoom_size_y)
            # 求 高精度地形相对于 低精度地形 offset
            lng_height = x0_height + x_begine_height * dx_height
            lat_height = y0_height + y_begine_height * dy_height
            lng_low = x0_low + (x_begine_low + buf_size_low) * dx_low
            lat_low = y0_low + (y_begine_low + buf_size_low) * dy_low
            x_offset = round((lng_height - lng_low) / dx_low)
            y_offset = round((lat_height - lat_low) / dy_low)

            # 替换
            for y in range(zoom_size_y):
                if y + y_offset >= len(z_low):
                    continue
                for x in range(zoom_size_x):
                    if x + x_offset >= len(z_low[y + y_offset]):
                        continue
                    z_height_v = z_height[y][x]
                    z_low_v = z_low[y + y_offset][x + x_offset]
                    if z_height_v != 0 and z_height_v != z_low_v:
                        z_low[y + y_offset][x + x_offset] = z_height_v
            points_xyz = []
            z_low = np.array(z_low)
            for x in range(z_low.shape[1]):
                for y in range(z_low.shape[0]):
                    point_xyz = [x, y, z_low[y, x]]
                    points_xyz.append(point_xyz)
            points_xyz = np.array(points_xyz)
            points_xy = points_xyz[:, 0:2]
            tri = Delaunay(points_xy)
            index = tri.simplices
            points_xyz = (points_xyz + (x_begine_low + buf_size_low, y_begine_low + buf_size_low, 0)) * (dx_low, dy_low, 1) + (x0_low, y0_low, 0)
            write_terrain(fname, points_xyz, index)


def write_terrain(fname, xyz, idx):
    """
    mash 三角网写入 terrain 文件，当该文件存在时，直接覆盖
    :param fname: terrain文件名
    :param xyz: 顶点
    :param idx: 索引
    :return: None
    """
    wkts = []
    for _ in range(idx.shape[0]):
        tri = xyz[idx[_]]
        triP = Polygon([(tri[0][0], tri[0][1], tri[0][2])
                           , (tri[1][0], tri[1][1], tri[1][2])
                           , (tri[2][0], tri[2][1], tri[2][2])])
        wkts.append(triP.wkt)
    topology = TerrainTopology(geometries=wkts)
    tile = TerrainTile(topology=topology)
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    if os.path.exists('%s.terrain' %fname):
        os.remove('%s.terrain' %fname)
    tile.toFile('%s.terrain' %fname, gzipped=False)


if __name__ == '__main__':
    filename_low = '/Users/cugxy/Documents/git/ctb-merge/data/dem/shaoguan.tif'
    filename = '/Users/cugxy/Documents/git/ctb-merge/data/dem/dem.tif'
    output_dir = '/Users/cugxy/Documents/git/ctb-merge/data/dem/result'
    zoom = 13
    end_zoom = 16
    r = get_intersect_block(filename_low, filename, output_dir, zoom, end_zoom)
    pass