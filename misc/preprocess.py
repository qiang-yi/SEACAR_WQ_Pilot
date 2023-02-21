# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:35:25 2023

@author: qiangy
"""

import numpy as np
import pandas as pd

def preprocessDis(df,col_ls,para_ls):
    
    # ------- Select columns and parameters ------
    df = df[col_ls]
    df = df.loc[df["ParameterName"].isin(para_ls)]
    df["timestamp"]=  pd.to_datetime(df['SampleDate'])
    #---------- remove outliers -----------------
    # Remove total nitrogen outliers (>100)
    df.drop(df[(df['ParameterName'] == 'Total Nitrogen') & 
                     (df['ResultValue'] > 10)].index,inplace=True)

    # Remove a single measurement in 1996-07-22 (RowID: 1582917)
    df.drop(df[df['RowID'] == 1582917].index, inplace=True)

    # Remove turbidity outliers (>25)
    df.drop(df[(df['ParameterName'] == 'Turbidity') & 
                     (df['ResultValue'] > 25)].index, inplace=True)

    # Remove Secchi Depth before 1995 (117 records)
    df.drop(df[(df['ParameterName'] == 'Secchi Depth') & 
                     (df['Year'] < 1995)].index, inplace=True)
    
    return df

def preprocessCon(df,col_ls,para_ls):
    
    # ------- Select columns and parameters ------
    df = df[col_ls]
    df = df.loc[df["ParameterName"].isin(para_ls)]
    df["timestamp"]=  pd.to_datetime(df['SampleDate'])
    # ------- Select data during daytime ------
    df["Hour"]     = df.apply(lambda x:x["timestamp"].strftime("%H"), axis=1)
    df["Hour"]     = df["Hour"].astype(int)
    df             = df[(df["Hour"]>=8) & (df["Hour"]<=18)]
    #---------- remove outliers -----------------
    # Remove total nitrogen outliers (>100)
    df.drop(df[(df['ParameterName'] == 'Total Nitrogen') & 
                     (df['ResultValue'] > 10)].index,inplace=True)

    # Remove a single measurement in 1996-07-22 (RowID: 1582917)
    df.drop(df[df['RowID'] == 1582917].index, inplace=True)

    # Remove turbidity outliers (>25)
    df.drop(df[(df['ParameterName'] == 'Turbidity') & 
                     (df['ResultValue'] > 25)].index, inplace=True)

    # Remove Secchi Depth before 1995 (117 records)
    df.drop(df[(df['ParameterName'] == 'Secchi Depth') & 
                     (df['Year'] < 1995)].index, inplace=True)
    
    return df

def preprocess(dis, con1, con2):
    # read csv files
    dfDis_orig = pd.read_csv(dis)
    dfCon1_orig = pd.read_csv(con1)
    dfCon2_orig = pd.read_csv(con2)
    
    # preset function parameters
    col_ls = ['RowID','ParameterName','ParameterUnits','ProgramLocationID','ActivityType','ManagedAreaName',
                   'SampleDate','Year','Month','ResultValue','ValueQualifier','Latitude_DD','Longitude_DD']
    para_ls = ["Salinity","Total Nitrogen","Dissolved Oxygen","Turbidity","Secchi Depth"]
        
    # preprocess and concatenate dataframes
    dfCon1 = preprocessCon(dfCon1_orig, col_ls, para_ls)
    dfCon2 = preprocessCon(dfCon2_orig, col_ls, para_ls)
    dfDis  = preprocessDis(dfDis_orig, col_ls, para_ls)
    dfCon  = pd.concat([dfCon1,dfCon2],ignore_index=True)
    return dfDis, dfCon