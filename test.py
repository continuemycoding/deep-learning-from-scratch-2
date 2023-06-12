import matplotlib.pyplot as plt
import numpy as np


def mandelbrot(c, max_iter):
    z = c.copy()
    output = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)

    for n in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        newly_diverged = np.logical_and(mask, np.abs(z) > 2)
        output[newly_diverged] = n
        mask[newly_diverged] = False

    return output


width = 256
height = 256

# 缩放因子
zoom_factor = 1.05

# 最大迭代次数
max_iter = 100

# 初始复数网格
x = np.linspace(-2.0, 0.5, width)
y = np.linspace(-1.25, 1.25, height)
X, Y = np.meshgrid(x, y)
c = X + 1j * Y

# 放大到Mandelbrot集的边缘
while True:
    # 生成分形图像
    image = mandelbrot(c, max_iter)

    # 找到迭代次数最高的像素的位置
    max_iter_idx = np.unravel_index(np.argmax(image), image.shape)
    x_center = X[max_iter_idx]
    y_center = Y[max_iter_idx]

    # 使用matplotlib显示图像
    extent = (np.min(X), np.max(X), np.min(Y), np.max(Y))
    plt.imshow(image, extent=extent, cmap='hot')
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.draw()
    plt.pause(0.005)  # 暂停以便观察
    plt.clf()  # 清除当前图像，以便在下一个循环中绘制新的图像

    # 改变范围进行缩放
    x_range = (x[-1] - x[0]) / zoom_factor
    y_range = (y[-1] - y[0]) / zoom_factor
    x = np.linspace(x_center - x_range/2, x_center + x_range/2, width)
    y = np.linspace(y_center - y_range/2, y_center + y_range/2, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
