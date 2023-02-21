# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:00:32 2023

@author: qiangy
"""
import time
import math  
import sklearn.metrics  
import arcgisscripting
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.mask
import rasterio.plot as rio_pl
import matplotlib.image as mpimg
#import rioxarray as rxr

from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio import MemoryFile
from rasterio.profiles import DefaultGTiffProfile
from sklearn.metrics import mean_squared_error
from shapely.geometry import box
from shapely.geometry import Polygon, Point
#from osgeo import gdal
#from pykrige.ok import OrdinaryKriging

# import arcpy and environmental settings
import arcpy
from arcpy.sa import *
arcpy.env.overwriteOutput = True

def interpolation_method(year,method,extent):
    select_method = str(method)
    if   select_method == "idw":
        print("This is Inverse Distance Weighting interpolation method")
        with arcpy.EnvManager(extent=extent, mask = path+"GIS_data/ch.shp",outputCoordinateSystem = arcpy.SpatialReference(3086), cellSize = 30):
                 arcpy.ga.IDW(in_features = gis_path+"Dry"+year+'.shp', 
                 z_field = "ResultValu", 
#                This layer is not generated  
                 out_ga_layer = gis_path + "dryIDW.lyrx",
                 out_raster   = gis_path + "dryIDW.tif"
                )

        with arcpy.EnvManager(extent=extent, mask = path+"GIS_data/ch.shp",outputCoordinateSystem = arcpy.SpatialReference(3086), cellSize = 30):
                 arcpy.ga.IDW(in_features = gis_path+"Wet"+year+'.shp', 
                 z_field = "ResultValu", 
#                This layer is not generated  
                 out_ga_layer = gis_path + "wetIDW.lyrx",
                 out_raster   = gis_path + "wetIDW.tif"
                )
                
        fig, axes = plt.subplots(1,2, figsize=(18, 8))

        fig.suptitle("{} (Parameter: {}).".format(Area, Para),fontsize=20)

        gdfDryShp.plot(ax = axes[0], marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = axes[0], color='none', edgecolor='black')
        cx.add_basemap(ax = axes[0],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)

#       Raster must added after basemap
        with rasterio.open(gis_path + "dryIDW.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[0], cmap="RdBu_r")
            axes[0].set_title('Dry Season')
#           Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[0],shrink=0.8)

            gdfWetShp.plot(ax = axes[1], marker = 'o', color = 'green', markersize = 6)
            extentShp.plot(ax = axes[1], color='none', edgecolor='black')
            cx.add_basemap(ax = axes[1],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)

#       Raster must added after basemap
        with rasterio.open(gis_path + "wetIDW.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[1], cmap="RdBu_r")
            axes[1].set_title('Wet Season')
#           Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[1],shrink=0.8)       
                
    elif select_method == "ok":
        print("This is Ordinary Kriging interpolation method")
#       Calculate dry season
        search_radius = 20000
        with arcpy.EnvManager(extent = extent, mask = path+"GIS_data/ch.shp",outputCoordinateSystem = arcpy.SpatialReference(3086), cellSize = 30):
            out_surface_raster = arcpy.sa.Kriging(in_point_features = gis_path+"Dry"+year+'.shp',
                                          z_field = "ResultValu",
                                          kriging_model = KrigingModelOrdinary("Spherical # # # #"),
                                         search_radius = RadiusVariable(20, search_radius))                           
#                                          "Spherical 0.000486 # # #",
#                                          0.000764192085967863,
#                                          "VARIABLE 12",
        out_surface_raster.save(gis_path+"dryKriging.tif")
#       Calculate wet season
        with arcpy.EnvManager(extent = extent, mask = path+"GIS_data/ch.shp",outputCoordinateSystem = arcpy.SpatialReference(3086), cellSize = 30):
            out_surface_raster = arcpy.sa.Kriging(gis_path+"Wet"+year+'.shp',
                                          "ResultValu",
                                         kriging_model = KrigingModelOrdinary("Spherical # # # #"),
                                         search_radius = RadiusVariable(20, search_radius))
#                                           "Spherical 0.000486 # # #",
#                                           0.000764192085967863,
#                                          "VARIABLE 12",
                                          
        out_surface_raster.save(gis_path+"wetKriging.tif")
        
#       Draw the map
        fig, axes = plt.subplots(1,2, figsize=(18, 8))
        fig.suptitle("{} (Parameter: {}).".format(Area, Para),fontsize=20)
        gdfDryShp.plot(ax = axes[0], marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = axes[0], color='none', edgecolor='black')
        cx.add_basemap(ax = axes[0],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)
#       Raster must added after basemap
        with rasterio.open(gis_path + "dryKriging.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[0], cmap="RdBu_r")
            axes[0].set_title('Dry Season')
            # Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[0],shrink=0.8)
        gdfWetShp.plot(ax = axes[1], marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = axes[1], color='none', edgecolor='black')
        cx.add_basemap(ax = axes[1],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)
        # Raster must added after basemap
        with rasterio.open(gis_path + "wetKriging.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[1], cmap="RdBu_r")
            axes[1].set_title('Wet Season')
            # Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[1],shrink=0.8)

    elif select_method == "ebk":
        print("This is Empirical Bayesian Kriging interpolation method")
        start_time = time.time()

        with arcpy.EnvManager(extent = extent, mask = path+"GIS_data/ch.shp",outputCoordinateSystem = arcpy.SpatialReference(3086), 
                              cellSize = 30, parallelProcessingFactor = "80%"):
            arcpy.ga.EmpiricalBayesianKriging(in_features = gis_path+"Dry"+year+'.shp', 
                                      z_field = "ResultValu", 
                                    # This layer is not generated  
                                      out_ga_layer = gis_path+"EBK_Dry_Layer",
                                      out_raster   = gis_path+"dryEBK.tif",
                                     # transformation_type = 'EMPIRICAL',
                                    search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(10000,0.5))
    
            arcpy.ga.EmpiricalBayesianKriging(in_features = gis_path+"Wet"+year+'.shp', 
                                      z_field = "ResultValu", 
                                     # This layer is not generated  
                                      out_ga_layer = gis_path+"EBK_Wet_Layer",
                                      out_raster = gis_path+"wetEBK.tif",
                                     # transformation_type = 'EMPIRICAL',
                                    search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(10000,0.5))
    
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        
        fig, axes = plt.subplots(1,2, figsize=(18, 8))

        fig.suptitle("{} (Parameter: {}).".format(Area, Para),fontsize=20)

        gdfDryShp.plot(ax = axes[0], marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = axes[0], color='none', edgecolor='black')
        cx.add_basemap(ax = axes[0],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)

        # Raster must added after basemap
        with rasterio.open(gis_path + "dryEBK.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[0], cmap="RdBu_r")
            axes[0].set_title('Dry Season')
#           Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[0],shrink=0.8)


        gdfWetShp.plot(ax = axes[1], marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = axes[1], color='none', edgecolor='black')
        cx.add_basemap(ax = axes[1],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)

#       Raster must added after basemap
        with rasterio.open(gis_path + "wetEBK.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[1], cmap="RdBu_r")
            axes[1].set_title('Wet Season')
#       Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[1],shrink=0.8)

    elif select_method == "rk":
        print("This is Regression Kriging interpolation method")
        ma_table = pd.read_csv(gis_path + "MA_table.csv")
        ra_fname = 'basy_{}.tif'.format(ma_table[ma_table['LONG_NAME']== Area]['MA_AreaID'].iloc[0].astype(str))
        fig, ax = plt.subplots(1, figsize=(18, 8))

        fig.suptitle("{} (covariate: {}).".format(Area, 'basymetry'),fontsize=20)

        gdfDryShp.plot(ax = ax, marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = ax, color='none', edgecolor='black')
        cx.add_basemap(ax = ax,source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)

#       Raster must added after basemap
        with rasterio.open(gis_path + "covariates/basymetry/basy_18.tif", "r+") as covar:
            band = covar.read(1)
            #band = numpy.ma.masked_array(band, mask=(band < -1000))
            #band = numpy.ma.masked_array(band, mask=(band > 1000))
            band = np.ma.masked_where((band < -100) | (band > 100), band)
            retted = rio_pl.show(band, transform=covar.transform, ax = ax, cmap="RdBu_r")
            ax.set_title('Basymetry')
            # Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=ax,shrink=0.8)
            
        start_time = time.time()

        with arcpy.EnvManager(extent = extent, mask = path+"GIS_data/ch.shp",outputCoordinateSystem = arcpy.SpatialReference(3086), 
                              cellSize = 30, parallelProcessingFactor = "80%"):
            out_surface_raster = arcpy.EBKRegressionPrediction_ga(in_features = gis_path+"Dry"+year+'.shp', 
                                                                   dependent_field = "ResultValu", 
                                                                  out_ga_layer = gis_path + "dryRK_GA",
                                                                    out_raster = gis_path + "dryRK.tif",
                                                                  in_explanatory_rasters = gis_path + "covariates/{}".format('NCEI_DEM_30m.tif'),
                                                                   transformation_type = 'EMPIRICAL',
                                                                  search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(10000,0.5)) 

            out_surface_raster = arcpy.EBKRegressionPrediction_ga(in_features = gis_path+"Wet"+year+'.shp', 
                                                                   dependent_field = "ResultValu",
                                                                  out_ga_layer = gis_path + "wetRK_GA",
                                                                  out_raster = gis_path + "wetRK.tif",
                                                                  in_explanatory_rasters = gis_path + "covariates/{}".format('NCEI_DEM_30m.tif'),
                                                                   transformation_type = 'EMPIRICAL',
                                                                  search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(10000,0.5))

        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        
        fig, axes = plt.subplots(1,2, figsize=(18, 8))

        fig.suptitle("{} (Parameter: {}).".format(Area, Para),fontsize=20)

        gdfDryShp.plot(ax = axes[0], marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = axes[0], color='none', edgecolor='black')
        cx.add_basemap(ax = axes[0],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)

        # Raster must added after basemap
        with rasterio.open(gis_path + "dryRK.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[0], cmap="RdBu_r")
            axes[0].set_title('Dry Season')
            # Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[0],shrink=0.8)


        gdfWetShp.plot(ax = axes[1], marker = 'o', color = 'green', markersize = 6)
        extentShp.plot(ax = axes[1], color='none', edgecolor='black')
        cx.add_basemap(ax = axes[1],source=cx.providers.Stamen.TonerLite,crs=gdfDryShp.crs)

        # Raster must added after basemap
        with rasterio.open(gis_path + "wetRK.tif", "r+") as dryOK:
            band = dryOK.read(1)
            band = numpy.ma.masked_array(band, mask=(band < 0))
            retted = rio_pl.show(band, transform=dryOK.transform, ax = axes[1], cmap="RdBu_r")
            axes[1].set_title('Wet Season')
            # Add legend
            im = retted.get_images()[1]
            fig.colorbar(im, ax=axes[1],shrink=0.8)
