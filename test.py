import numpy as np
import colorsys
import matplotlib.pyplot as plt

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


def color_map(c, max_iter):
    mfactor = 0.5
    v = c ** mfactor / max_iter ** mfactor
    hv = 0.67 - v
    hv[hv < 0] += 1
    rgb = np.array([colorsys.hsv_to_rgb(hv_val, 1, 1 - (v_val - 0.1) ** 2 / 0.9 ** 2) for hv_val, v_val in zip(np.ravel(hv), np.ravel(v))])
    rgb = np.minimum(255, (rgb * 255).astype(int)).reshape(*c.shape, 3)
    return rgb


width = 256
height = 256
zoom_factor = 1.05
max_iter = 500

x = np.linspace(-2.0, 0.5, width)
y = np.linspace(-1.25, 1.25, height)
X, Y = np.meshgrid(x, y)
c = X + 1j * Y

while True:
    image = mandelbrot(c, max_iter)
    rgb_image = color_map(image, max_iter)

    # Find new center
    max_iter_indices = np.unravel_index(np.argmax(image), image.shape)
    y_center, x_center = Y[max_iter_indices], X[max_iter_indices]

    extent = (np.min(X), np.max(X), np.min(Y), np.max(Y))
    plt.imshow(rgb_image, extent=extent)
    plt.title("Mandelbrot Set")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.draw()
    plt.pause(0.005)
    plt.clf()

    # Zoom in
    x_range = (x[-1] - x[0]) / zoom_factor
    y_range = (y[-1] - y[0]) / zoom_factor
    x = np.linspace(x_center - x_range / 2, x_center + x_range / 2, width)
    y = np.linspace(y_center - y_range / 2, y_center + y_range / 2, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
