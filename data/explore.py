# Header: Explore the data

import rasterio
from matplotlib import colors, pyplot as plt

from rasterio.windows import Window
from rasterio.enums import Resampling


ROOT_DIR = "C:/Users/somdi/Downloads/GHS_POP_P2030_GLOBE_R2022A_54009_100_V1_0"
POP_FILENAME = "GHS_POP_P2030_GLOBE_R2022A_54009_100_V1_0.tif"

ROW_SCALE = 1000  # 1 degree latitude = 1000 pixels
COL_SCALE = 1000  # 1 degree longitude = 1000 pixels
ROW_OFFSET = 90  # 90 degrees latitude
COL_OFFSET = 180  # 180 degrees longitude


def plot_data(data):
    """Plot np data as hot color map, in log scale"""

    # change 'unknown' values in data to 0
    data[data == 200] = 0

    # Plot np data as hot color map, in log scale, with color bar,
    # and set the color bar to log scale
    fig, ax = plt.subplots()
    im = ax.imshow(data, norm=colors.LogNorm(), cmap="hot_r")
    fig.colorbar(im, ax=ax, norm=colors.LogNorm())
    plt.show()


def plot_two_data(data1, data2):
    """Plot np data as hot color map, in linear scale.
    Plot two side by side plots."""

    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(data1, cmap="hot_r")
    fig.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(data2, cmap="hot_r")
    fig.colorbar(im2, ax=ax2)

    # resize the plot window
    fig.set_size_inches(16, 10)
    plt.show()


def read_resampled(resolution=0.1):
    """Read the whole data resampled to a resolution in degrees"""

    with rasterio.open(f"{ROOT_DIR}/{POP_FILENAME}") as src:
        return src.read(
            out_shape=(
                src.count,
                int(src.height / ROW_SCALE / resolution),
                int(src.width / COL_SCALE / resolution),
            ),
            resampling=Resampling.bilinear,
        )


def read_subset(latitude, longitude, size):
    """Read and plot a subset of data"""

    with rasterio.open(f"{ROOT_DIR}/{POP_FILENAME}") as src:
        return src.read(
            window=Window(
                (COL_OFFSET + longitude) * COL_SCALE - size / 2,
                (ROW_OFFSET - latitude) * ROW_SCALE - size / 2,
                size,
                size,
            )
        )


def data_summary():
    # Read the raster data
    with rasterio.open(f"{ROOT_DIR}/{POP_FILENAME}") as src:
        print("Tiff Boundary", src.bounds)
        print("Tiff CRS", src.crs)
        print("Tiff Resolution", src.res)
        print("Tiff Shape", src.shape)
        print("Tiff Count", src.count)


if __name__ == "__main__":
    data_summary()

    # plot the whole data resampled to 1 degree
    data = read_resampled(resolution=1)

    # read and plot a subset of data, it is too large to read whole at once
    # Supposed to be new york city, but the plot is about 33.7 S -83.2 W
    # data = read_subset(latitude=40.71, longitude=-74.01, size=10000) # Real NY coords
    # data = read_subset(latitude=47.71, longitude=-64.01, size=1000)  # adjusted NY coords

    plot_data(data[0])
