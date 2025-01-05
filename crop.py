## Three channels (RGB) ##
from osgeo import gdal

input_file = "data/Sentinel-2/RGB.tif"
dataset = gdal.Open(input_file, gdal.GA_ReadOnly)


width = dataset.RasterXSize
height = dataset.RasterYSize

crop_width = 10
crop_height = 10

num_crops_width = width // crop_width
num_crops_height = height // crop_height

options = [
    'TILED=YES',
    'PHOTOMETRIC=RGB',
    'PROFILE=GeoTIFF',
]

for i in range(num_crops_width):
    for j in range(num_crops_height):
        left = i * crop_width
        upper = j * crop_height
        right = left + crop_width
        lower = upper + crop_height

        output_file = f"data/clip/rgb/output_{j+1}_{i+1}.tif"
        output_dataset = gdal.GetDriverByName('GTiff').Create(
            output_file,
            crop_width,
            crop_height,
            3,  
            gdal.GDT_Byte,
            options
        )

        for band_index in range(1, 4): 
            band = dataset.GetRasterBand(band_index)
            output_band = output_dataset.GetRasterBand(band_index)
            data = band.ReadAsArray(left, upper, crop_width, crop_height)
            output_band.WriteArray(data)

        output_dataset = None

dataset = None


# ## single channel ###
# from osgeo import gdal
# import numpy as np

# input_file = "data/Sentinel-2/SWIR.tif"
# dataset = gdal.Open(input_file, gdal.GA_ReadOnly)

# width = dataset.RasterXSize
# height = dataset.RasterYSize

# crop_width = 10
# crop_height = 10

# num_crops_width = width // crop_width
# num_crops_height = height // crop_height

# min_value = dataset.GetRasterBand(1).GetMinimum()
# max_value = dataset.GetRasterBand(1).GetMaximum()

# options = [
#     'TILED=YES',
#     'PROFILE=GeoTIFF',
# ]

# for i in range(num_crops_width):
#     for j in range(num_crops_height):
#         left = i * crop_width
#         upper = j * crop_height
#         right = left + crop_width
#         lower = upper + crop_height

#         data = dataset.ReadAsArray(left, upper, crop_width, crop_height)

#         scaled_data = np.interp(data, (min_value, max_value), (0, 65535)).astype(np.uint16)

#         output_file = f"data/clip/swir/output_{j+1}_{i+1}.tif"
#         output_dataset = gdal.GetDriverByName('GTiff').Create(
#             output_file,
#             crop_width,
#             crop_height,
#             1, 
#             gdal.GDT_UInt16, 
#             options
#         )
#         output_band = output_dataset.GetRasterBand(1)
#         output_band.WriteArray(scaled_data)

#         output_dataset = None

# dataset = None
