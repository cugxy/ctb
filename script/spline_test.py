# -*- coding: utf-8 -*-
"""
演示二维插值。
"""
import numpy as np
from IPython import embed
from scipy import interpolate
import pylab as pl
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if 0:
        x=np.linspace(0,10,11)
        y=np.sin(x)
        embed()
        xnew=np.linspace(0,10,101)

        pl.plot(x,y,'ro')
        list1=['linear','nearest']
        list2=[0,1,2,3]
        f = interpolate.interp1d(x, y, kind='linear')
        # f是一个函数，用这个函数就可以找插值点的函数值了：
        ynew = f(xnew)
        pl.plot(xnew, ynew, label='linear')
        embed()

        pl.legend(loc='lower right')
        pl.show()
    if 0:
        def func(x,y):
            return (x+y)*np.exp(-5*(x**2+y**2))
        x,y=np.mgrid[-1:1:8j,-1:1:8j]
        z=func(x,y)
        func=interpolate.interp2d(x,y,z,kind='linear')


        xnew=np.linspace(-1,1,100)
        ynew=np.linspace(-1,1,100)
        znew=func(xnew,ynew)#xnew, ynew是一维的，输出znew是二维的
        xnew,ynew=np.mgrid[-1:1:100j,-1:1:100j]#统一变成二维，便于下一步画图
        ax=plt.subplot(111,projection='3d')
        ax.plot_surface(xnew,ynew,znew)
        ax.scatter(x,y,z,c='r',marker='^')
        plt.show()
        pass
    if 1:
        x = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5]
        y = [0, 1, 2, 3, 4, 5, 0, 5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 5, 0, 1, 2, 3, 4, 5]
        z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0, 5, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fig, ax = plt.subplots()
        # ax.imshow(_z_height, interpolation='nearest', cmap=plt.cm.gray)
        # xnew, ynew = np.mgrid[0:(x_size + buf_size * 2) * ratio, 0:(y_size + buf_size * 2) * ratio]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='r')
        func = interpolate.interp2d(x, y, z, kind='linear')
        xnew = np.linspace(0, 5, 100)
        ynew = np.linspace(0, 5, 100)
        znew = func(xnew, ynew)
        xnew, ynew = np.mgrid[0:5:100j, 0:5:100j]  # 统一变成二维，便于下一步画图
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(xnew, ynew, znew)
        plt.show()



