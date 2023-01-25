# SEACAR Water Quality Pilot Project

# Task 1a: Exploratory Analysis

The exploratory analyses include:
1.	Data slicing and outlier removal
2.	Line chart of weekly/monthly/yearly count of data records for each WQ parameter in each managed area.
3.	Line chart with error bars showing monthly and yearly mean and standard deviation of each WQ parameter in each managed area
4.	Line charts of monthly mean at different stations for each WQ parameter in each managed area.
5.	Box plots of WQ parameters aggregated by month of the year for each managed area.
6.	Plots in (4) and (5) of four managed areas with sufficient data are organized by rows (WQ parameter) and columns (managed areas) in order to determine seasonality (https://usf.box.com/s/x05pmyx686hwsp56km0ayelwf1uxxnu2)
7.	Histogram showing sampling frequencies at each observation locations for each WQ parameters
8.	Pie chart showing the ratios of random and fixed sampling locations for each WQ parameters
9.	Bar chart of ratios of fixed sampling locations for each WQ parameters

Maps are created to show the spatial distribution of the WQ samples.
1.	Bubble map of sampling points of WQ parameter in each managed area. The Github repository only shows dissolved oxygen and total nitrogen. Maps of other parameters can be easily generated by changing program parameters.
2.	Bubble maps of sampling points of WQ parameter in each month of a year in each managed area. The Github repository only shows dissolved oxygen and total nitrogen in 2019. Maps of other parameters and in other years can be easily generated by changing program parameters.
3.	Maps showing locations of different continuous stations are created for each managed area.


File descriptions

-	[SEACAR_WQ_Exploratory_Analysis_Dis.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/SEACAR_WQ_Exploratory_Analysis_Dis.ipynb): Temporal analysis of discrete data
-	[Sample_Location_Analysis_Dis.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Sample_Location_Analysis_Dis.ipynb): Spatial locations of discrete data
-	[SEACAR_WQ_Exploratory_Analysis_Con.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/SEACAR_WQ_Exploratory_Analysis_Con.ipynb): Temporal analysis of continuous data: all stations
-	[Sample_Location_Analysis_Con.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Sample_Location_Analysis_Con.ipynb): Spatial locations of discrete data: all stations
-	[SEACAR_WQ_Exploratory_Analysis_Con_Stations.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/SEACAR_WQ_Exploratory_Analysis_Con_Stations.ipynb): Temporal analysis of continuous data: separate by station
-	[Sample_Location_Analysis_Con_Stations.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Sample_Location_Analysis_Con_Stations.ipynb): Spatial locations of discrete data: separate by station


# Task 1b: Exploratory Analysis
- [Interpolation_ArcGIS.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Interpolation_ArcGIS.ipynb): Spatial interpolation using testing dataset.
