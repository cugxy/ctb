
import os

from IPython import embed
import gdal
from gdalconst import *


def read_tif(filename):
    data_set = gdal.Open(filename)
    band = data_set.GetRasterBand(1)
    # 第 1， 2 参数表示左上角位置
    # 第 3， 4 参数表示读取范围大小
    # 第 5， 6 参数表示读取范围比例缩小后范围大小
    return band.ReadAsArray(30, 0, 20, 20, 10, 10)




if __name__ == '__main__':
    if 1:
        filename = r'E:\xy\doc\dem\dem.tif'
        data = read_tif(filename)
        print(data)