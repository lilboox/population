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

    # change negative values in data to 0
    data[data < 0] = 0

    # Plot np data as hot color map, in log scale, with color bar,
    # and set the color bar to log scale
    fig, ax = plt.subplots()
    im = ax.imshow(data, norm=colors.LogNorm(), cmap="hot_r")
    fig.colorbar(im, ax=ax, norm=colors.LogNorm())
    plt.show()


def plot_globe(src):
    """Read the whole data resampled to 1 degree"""

    data = src.read(
        out_shape=(
            src.count,
            int(src.height / ROW_SCALE),
            int(src.width / COL_SCALE),
        ),
        resampling=Resampling.bilinear,
    )

    plot_data(data[0])


def plot_subset(src, latitude, longitude, size):
    """Read and plot a subset of data"""

    # Read the raster data
    data = src.read(
        window=Window(
            (COL_OFFSET + longitude) * COL_SCALE - size / 2,
            (ROW_OFFSET - latitude) * ROW_SCALE - size / 2,
            size,
            size,
        )
    )

    plot_data(data[0])


def data_summary():
    # Read the raster data
    with rasterio.open(f"{ROOT_DIR}/{POP_FILENAME}") as src:
        print("Tiff Boundary", src.bounds)
        print("Tiff CRS", src.crs)
        print("Tiff Resolution", src.res)
        print("Tiff Shape", src.shape)
        print("Tiff Count", src.count)

        # plot the whole data resampled to 1 degree
        plot_globe(src)

        # read and plot a subset of data, it is too large to read whole at once
        # Supposed to be new york city, but the plot is about 33.7 S -83.2 W
        plot_subset(src, latitude=40.71, longitude=-74.01, size=10000)


if __name__ == "__main__":
    data_summary()
