import os, math
import gdal
import numpy as np
import cv2
from skimage import measure
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPoint
from quantized_mesh_tile.terrain import TerrainTile
from quantized_mesh_tile.global_geodetic import GlobalGeodetic
from quantized_mesh_tile.topology import TerrainTopology
from scipy import interpolate

import matplotlib.pyplot as plt
from IPython import embed

def tile_level(z):
    assert(z >= 0)
    x = 2 ** (z+1)
    y = 2 ** z
    xyz = [x, y, z]
    return xyz


def get_data_bound(z_value, alpha):
    '''
    获取 tif 有效值边界坐标
    :param data_set:
    :param alpha:
    :return:
    '''
    if z_value is None:
        return None
    z_value = np.insert(z_value, 0, 0, axis=0) # 添加 0 包围盒
    z_value = np.insert(z_value, 0, 0, axis=1)
    z_value = np.insert(z_value, z_value.shape[0], 0, axis=0)
    z_value = np.insert(z_value, z_value.shape[1], 0, axis=1)
    contours = measure.find_contours(z_value, alpha)
    yx = np.zeros(shape=(0, 2))
    for contour in iter(contours):
        contour = np.around(contour) # 取整
        yx = np.vstack((yx, contour))
    _tmp = yx + (1, 0)                  # 偏移
    _tmp_yx = np.vstack((yx, _tmp))
    _tmp = yx - (1, 0)
    _tmp_yx = np.vstack((_tmp_yx, _tmp))
    _tmp = yx + (0, 1)
    _tmp_yx = np.vstack((_tmp_yx, _tmp))
    _tmp = yx - (0, 1)
    _tmp_yx = np.vstack((_tmp_yx, _tmp))
    yx = np.vstack((yx, _tmp_yx))
    _tmp = yx.astype('i4')
    _tmp = _tmp.view('S8').flatten()  # 转换成 view 处理，即字符串
    keep = np.unique(_tmp, return_index=True)[1]  # 去重 得到索引
    yx = yx[keep].astype('i4')
    yx = np.array([_yx for _yx in yx if 0 <= _yx[0] < z_value.shape[0] and 0 <= _yx[1] < z_value.shape[1]]) # 保证范围内
    yx_r = np.zeros(shape=(0, 2))
    for _yx in yx:  # 保证有值
        if z_value[_yx[0], _yx[1]] != 0:
            yx_r = np.vstack((yx_r, _yx))
    return yx_r - 1   # 减去 0 包围盒


def merge_and_cut(filename_low, filename_height, output_dir, start_zoom, end_zoom):
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

    min_lng_height = x0_height
    min_lat_height = y0_height
    max_lng_height = x0_height + x_size_height * dx_height
    max_lat_height = y0_height + y_size_height * dy_height
    min_lng_low = x0_low
    min_lat_low = y0_low
    max_lng_low = x0_low + x_size_low * dx_low
    max_lat_low = y0_low + y_size_low * dy_low
    x_offset = round((min_lng_height - min_lng_low) / dx_low)
    y_offset = round((min_lat_height - min_lat_low) / dy_low)
    x_size = round((max_lng_height - min_lng_height) / dx_low)
    y_size = round((max_lat_height - min_lat_height) / dy_low)

    bounds = get_tif_tile_bounds(filename_height, output_dir, start_zoom)
    if bounds == {}:
        return
    _, value = next(iter(bounds.items()))
    _zoom_size_low = round((value[2] - value[0]) / dx_low)
    buf_size = _zoom_size_low + 128 # 设置 buf 大小大于一块地形的 size，保证按块取数据时，块是完整的

    z_low_with_buf = band_low.ReadAsArray(x_offset - buf_size, y_offset - buf_size, x_size + buf_size * 2, y_size + buf_size * 2).astype('f4')
    # z_height = band_height.ReadAsArray(0, 0, x_size_height, y_size_height).astype('f4')

    for zoom in range(start_zoom, end_zoom + 1):
        # merge
        # test-------------------------------------------------------------------------------------------------------------------------
        # zoom += 4
        # test-------------------------------------------------------------------------------------------------------------------------
        ratio = 2 ** (zoom - start_zoom)
        dsize_low = ((x_size + buf_size * 2) * ratio, (y_size + buf_size * 2) * ratio)
        _z_low_with_buf = cv2.resize(src=z_low_with_buf, dsize=dsize_low, interpolation=cv2.INTER_LINEAR)
        dsize_height = (x_size * ratio, y_size * ratio)
        # _z_height = cv2.resize(src=z_height, dsize=dsize_height, interpolation=cv2.INTER_NEAREST)
        z_height = band_height.ReadAsArray(0, 0, x_size_height, y_size_height, x_size * ratio, y_size * ratio).astype('f4')
        _z_height = z_height
        bound_yx = get_data_bound(_z_height, 0.8)

        # test 展示边界 ----------------------------------------------------------------------------------------------------------------------------------
        if 0:
            fig, ax = plt.subplots()
            ax.imshow(_z_height, interpolation='nearest', cmap=plt.cm.gray)
            ax.scatter(bound_yx[:,1], bound_yx[:,0], alpha=0.6)
            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()
        # test----------------------------------------------------------------------------------------------------------------------------------

        _yx_offset = buf_size * ratio
        # 如果没有边界 则，不融合，直接采用低精度数据 cut
        if bound_yx.shape[0] != 0:
            bound_yx = bound_yx.astype('i4')
            _x = np.zeros(shape=(0, 1)) # x y z 添加边界值
            _y = np.zeros(shape=(0, 1))
            _z = np.zeros(shape=(0, 1))
            for _bound_yx in bound_yx:
                _bound_y_height = _bound_yx[0]
                _bound_x_height = _bound_yx[1]
                _bound_z_height = _z_height[_bound_y_height, _bound_x_height]
                _bound_y_low = _bound_yx[0] + buf_size * ratio
                _bound_x_low = _bound_yx[1] + buf_size * ratio
                _bound_z_low = _z_low_with_buf[_bound_y_low, _bound_x_low]
                _z_d = _bound_z_height - _bound_z_low
                _x = np.vstack((_x, _bound_x_low))
                _y = np.vstack((_y, _bound_y_low))
                _z = np.vstack((_z, _z_d))
            # x y z 添加边界值缓冲边界 0 值
            _xy = np.hstack((_x, _y))
            _m_point = MultiPoint(_xy)
            _zero_bound = _m_point.buffer(50)
            _zero_bound_xy = _zero_bound.exterior.coords.xy
            _zero_bound_x = np.around(np.array(_zero_bound_xy[0])).reshape(len(_zero_bound_xy[0]), -1)
            _zero_bound_y = np.around(np.array(_zero_bound_xy[1])).reshape(len(_zero_bound_xy[1]), -1)
            _zero_bound_z = np.zeros(shape=(_zero_bound_x.shape))
            _x = np.vstack((_x, _zero_bound_x))
            _y = np.vstack((_y, _zero_bound_y))
            _z = np.vstack((_z, _zero_bound_z))
            # x y z 添加数据边界 0 值
            for _x_edge in range(0, (x_size + buf_size * 2) * ratio):
                _x = np.vstack((_x, [_x_edge]))
                _y = np.vstack((_y, [0]))
                _x = np.vstack((_x, [_x_edge]))
                _y = np.vstack((_y, [(y_size + buf_size * 2) * ratio]))
                _z = np.vstack((_z, [[0], [0]]))
            for _y_edge in range(0, (y_size + buf_size * 2) * ratio):
                _y = np.vstack((_y, [_y_edge]))
                _x = np.vstack((_x, [0]))
                _y = np.vstack((_x, [_y_edge]))
                _x = np.vstack((_x, [(x_size + buf_size * 2) * ratio]))
                _z = np.vstack((_z, [[0], [0]]))

            func = interpolate.interp2d(_x, _y, _z, kind='linear')
            _x_new = np.arange(0, (x_size + buf_size * 2) * ratio)
            _y_new = np.arange(0, (y_size + buf_size * 2) * ratio)
            _z_d_new = func(_x_new, _y_new)
            _z_low_with_buf = _z_low_with_buf + _z_d_new


            for y in range(y_size * ratio):
                for x in range(x_size * ratio):
                    _z_height_v = _z_height[y][x]
                    y_low = y + buf_size * ratio
                    x_low = x + buf_size * ratio
                    _z_low_v = _z_low_with_buf[y_low][x_low]
                    if _z_height_v != 0 and _z_height_v != _z_low_v:
                        _z_low_with_buf[y_low][x_low] = _z_height_v

        # cut
        z_merged = _z_low_with_buf
        blc_bounds = get_tif_tile_bounds(filename_height, output_dir, zoom)
        # [[min_lng, min_lat, max_lng, max_lat], ...]
        for fname in blc_bounds:
            blc_bound = blc_bounds[fname]
            blc_x_min_low = round((blc_bound[0] - min_lng_low) / dx_low)
            blc_y_min_low = round((blc_bound[3] - min_lat_low) / dy_low)
            blc_x_max_low = round((blc_bound[2] - min_lng_low) / dx_low)
            blc_y_max_low = round((blc_bound[1] - min_lat_low) / dy_low)

            blc_x_offset_low = (blc_x_min_low - x_offset + buf_size) * ratio
            blc_y_offset_low = (blc_y_min_low - y_offset + buf_size) * ratio
            blc_x_size_low = (blc_x_max_low - blc_x_min_low) * ratio
            blc_y_size_low = (blc_y_max_low - blc_y_min_low) * ratio
            blc_z_low = z_merged[blc_y_offset_low:blc_y_offset_low + blc_y_size_low, blc_x_offset_low:blc_x_offset_low + blc_x_size_low]
            points_xyz = []
            z_low = np.array(blc_z_low)
            for x in range(blc_z_low.shape[1]):
                for y in range(blc_z_low.shape[0]):
                    point_xyz = [x, y, blc_z_low[y, x]]
                    points_xyz.append(point_xyz)
            points_xyz = np.array(points_xyz)
            points_xy = points_xyz[:, 0:2]
            tri = Delaunay(points_xy)
            index = tri.simplices
            points_xyz = (points_xyz + (blc_x_min_low, blc_y_min_low, 0)) * (dx_low, dy_low, 1) + (x0_low, y0_low, 0)
            write_terrain(fname, points_xyz, index)


def get_tif_tile_bounds(filename, output_dir, zoom):
    '''
    根据 cesium 层级分块，读取 tif，利用 zoom 求得 tif 所涉及的每一块地形的 gps 包围盒
    :param filename: tif 文件路径
    :param zoom: 层级
    :return: bounds {fname: [min_lng, min_lat, max_lng, max_lat], ...}
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


def get_intersect_block_bak(filename_low, filename_height, output_dir, zoom, end_zoom):
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
            z_low = cv2.blur(z_low,(7, 7))
            # z_low = z_low[buf_size:-buf_size, buf_size:-buf_size]

            # 读取 高精度 地形
            x_min_height = round((bound[0] - x0_height) / dx_height)
            y_min_height = round((bound[3] - y0_height) / dy_height)
            x_max_height = round((bound[2] - x0_height) / dx_height)
            y_max_height = round((bound[1] - y0_height) / dy_height)

            buf_size_height = round(buf_size / zoom_size_low * (x_max_height - x_min_height))
            buf_size_height_left = 0
            x_begine_height = x_min_height - buf_size_height
            if x_begine_height > x_size_height:
                continue
            if x_begine_height < 0:
                buf_size_height_left = round(max((x_min_height - 0) / (x_max_height - x_min_height) * zoom_size_low, 0))
                x_begine_height = 0

            buf_size_height_botton = 0
            y_begine_height = y_min_height - buf_size_height
            if y_begine_height > y_size_height:
                continue
            if y_begine_height < 0:
                buf_size_height_botton = round(max((y_min_height - 0)/(x_max_height - x_min_height)*zoom_size_low, 0))
                y_begine_height = 0

            buf_size_height_right = 0
            x_end_height = x_max_height + buf_size_height
            if x_end_height < 0:
                continue
            if x_end_height > x_size_height:
                buf_size_height_right = round(max((x_size_height - x_max_height)/(x_max_height - x_min_height)*zoom_size_low, 0))
                x_end_height = x_size_height

            buf_size_height_top = 0
            y_end_height = y_max_height + buf_size_height
            if y_end_height < 0:
                continue
            if y_end_height > y_size_height:
                buf_size_height_top = round(max((y_size_height - y_max_height)/(x_max_height - x_min_height)*zoom_size_low, 0))
                y_end_height = y_size_height

            zoom_size_x = round(zoom_size_low * (x_end_height - x_begine_height) / (x_max_height - x_min_height))
            zoom_size_y = round(zoom_size_low * (y_end_height - y_begine_height) / (y_max_height - y_min_height))
            z_height = band_height.ReadAsArray(x_begine_height,
                                               y_begine_height,
                                               x_end_height - x_begine_height,
                                               y_end_height - y_begine_height,
                                               zoom_size_x + buf_size_height_left + buf_size_height_right,
                                               zoom_size_y + buf_size_height_top + buf_size_height_botton)
            # 求 高精度地形相对于 低精度地形 offset
            lng_height = x0_height + x_begine_height * dx_height
            lat_height = y0_height + y_begine_height * dy_height
            lng_low = x0_low + x_begine_low * dx_low
            lat_low = y0_low + y_begine_low * dy_low
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

            z_low = cv2.GaussianBlur(z_low, (5, 5), 0)
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
            # 去除 buf
            z_low = z_low[buf_size:-buf_size, buf_size:-buf_size]

            # 生成 terrain 文件
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
            points_xyz = (points_xyz + (x_min_low, y_min_low, 0)) * (dx_low, dy_low, 1) + (x0_low, y0_low, 0)
            write_terrain(fname, points_xyz, index)


def get_intersect_block(filename_low, filename_height, output_dir, zoom, end_zoom):
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

    for _zoom in range(zoom, end_zoom + 1):
        # 获取高精度 tif 所涉及的 地形块
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
                                         y_end_low - y_begine_low).astype('f4')
            z_low = cv2.resize(z_low, (zoom_size_low + buf_size * 2, zoom_size_low + buf_size * 2), interpolation=cv2.INTER_LINEAR)

            # # 读取 高精度 地形
            # x_min_height = round((bound[0] - x0_height) / dx_height)
            # y_min_height = round((bound[3] - y0_height) / dy_height)
            # x_max_height = round((bound[2] - x0_height) / dx_height)
            # y_max_height = round((bound[1] - y0_height) / dy_height)
            #
            # buf_size_height_left = 0
            # if x_min_height > x_size_height:
            #     print('Height-precision terrain does not completely cover %s' % fname)
            #     continue
            # if x_min_height < 0:
            #     buf_size_height_left = round(max((x_min_height - 0) / (x_max_height - x_min_height) * zoom_size_low, 0))
            #     x_min_height = 0
            #
            # buf_size_height_botton = 0
            # if y_min_height > y_size_height:
            #     print('Height-precision terrain does not completely cover %s' % fname)
            #     continue
            # if y_min_height < 0:
            #     buf_size_height_botton = round(
            #         max((y_min_height - 0) / (x_max_height - x_min_height) * zoom_size_low, 0))
            #     y_min_height = 0
            #
            # buf_size_height_right = 0
            # if x_max_height < 0:
            #     print('Height-precision terrain does not completely cover %s' % fname)
            #     continue
            # if x_max_height > x_size_height:
            #     buf_size_height_right = round(
            #         max((x_size_height - x_max_height) / (x_max_height - x_min_height) * zoom_size_low, 0))
            #     x_max_height = x_size_height
            #
            # buf_size_height_top = 0
            # if y_max_height < 0:
            #     print('Height-precision terrain does not completely cover %s' % fname)
            #     continue
            # if y_max_height > y_size_height:
            #     buf_size_height_top = round(
            #         max((y_size_height - y_max_height) / (x_max_height - x_min_height) * zoom_size_low, 0))
            #     y_max_height = y_size_height
            #
            # zoom_size_x = round(zoom_size_low * (x_max_height - x_min_height) / (x_max_height - x_min_height))
            # zoom_size_y = round(zoom_size_low * (y_max_height - y_min_height) / (y_max_height - y_min_height))
            # _z = band_low.ReadAsArray(x_min_height, y_min_height, x_max_height - x_min_height, y_max_height - y_min_height).astype('f4')
            # z_height = cv2.resize(_z, (zoom_size_x, zoom_size_y), interpolation=cv2.INTER_CUBIC)
            #
            # # 求 高精度地形相对于 低精度地形 offset
            # lng_height = x0_height + x_min_height * dx_height
            # lat_height = y0_height + y_min_height * dy_height
            # lng_low = x0_low + x_min_low * dx_low
            # lat_low = y0_low + y_min_low * dy_low
            # x_offset = round((lng_height - lng_low) / dx_low)
            # y_offset = round((lat_height - lat_low) / dy_low)

            # for y in range(zoom_size_y):
            #     if y + y_offset >= len(z_low):
            #         continue
            #     for x in range(zoom_size_x):
            #         if x + x_offset >= len(z_low[y + y_offset]):
            #             continue
            #         z_height_v = z_height[y][x]
            #         z_low_v = z_low[y + y_offset][x + x_offset]
            #         if z_height_v != 0 and z_height_v != z_low_v:
            #             z_low[y + y_offset][x + x_offset] = z_height_v

            z_low = z_low[buf_size:-buf_size, buf_size:-buf_size]

            # 生成 terrain 文件
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
            points_xyz = (points_xyz + (x_min_low, y_min_low, 0)) * (dx_low, dy_low, 1) + (x0_low, y0_low, 0)
            write_terrain(fname, points_xyz, index)


def write_terrain(fname, xyz, idx):
    '''
	mash 三角网写入 terrain 文件，当该文件存在时，直接覆盖
	:param fname: terrain文件名
	:param xyz: 顶点
	:param idx: 索引
	:return: None
	'''
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
    tile.toFile('%s.terrain' %fname, gzipped=True)


if __name__ == '__main__':
    filename_low = r'E:\xy\doc\dem\shaoguan.tif'
    filename = r'E:\xy\doc\dem\dem.tif'
    output_dir = r'E:\xy\doc\dem\result-merge'
    zoom = 13
    end_zoom = 16
    data_set = gdal.Open(filename)
    if 0:
        r = get_intersect_block(filename_low, filename, output_dir, zoom, end_zoom)
    if 1:
        merge_and_cut(filename_low, filename, output_dir, zoom, end_zoom)

    pass