import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

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

# 上一次迭代的中心位置
prev_x_center = 0
prev_y_center = 0

threshold = max_iter * 0.1

while True:
    # 生成分形图像
    image = mandelbrot(c, max_iter)
    
    # 二值化图像：高迭代次数设置为1，其它设置为0
    binary_image = (image > threshold).astype(int)
    
    # 找连通区域
    labeled_image, num_features = ndi.label(binary_image)
    
    # 如果存在至少一个连通区域，选择最大的连通区域
    if num_features > 0:
        region_sizes = ndi.sum(binary_image, labeled_image, range(num_features + 1))
        largest_region_label = region_sizes[1:].argmax() + 1
        coordinates = np.argwhere(labeled_image == largest_region_label)
        
        # 计算连通区域的中心点坐标
        center_pixel = coordinates.mean(axis=0).astype(int)
        x_center = X[tuple(center_pixel)]
        y_center = Y[tuple(center_pixel)]
    else:
        x_center = prev_x_center
        y_center = prev_y_center
    
    prev_x_center = x_center
    prev_y_center = y_center
    
    extent = (np.min(X), np.max(X), np.min(Y), np.max(Y))
    plt.imshow(image, extent=extent, cmap='hot')
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.draw()
    plt.pause(0.005)
    plt.clf()  # 清除当前图像，以便在下一个循环中绘制新的图像
    
    # 改变范围进行缩放
    x_range = (x[-1] - x[0]) / zoom_factor
    y_range = (y[-1] - y[0]) / zoom_factor
    x = np.linspace(x_center - x_range/2, x_center + x_range/2, width)
    y = np.linspace(y_center - y_range/2, y_center + y_range/2, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
