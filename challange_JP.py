# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:42:36 2022

@author: JOERG4
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#%%
def find_all_files_of_type(indirectory:str,filoeextension:str='train.gz'):
    """Scans for multiple .gz files in given directory and return a list of 
    strings!!
    """
    
    filelist = [indirectory +'\\'+ file.name for file in os.scandir(indirectory) 
                 if file.name.endswith((filoeextension))]
    
    if len(filelist) == 0:
        print('No gz files found')
    
    return filelist
    

def load_single_gz_file_in_to_pandas(inpath:str):
    """Load a sing .gz file into pandas df"""
    
    if not inpath:
        print('Path can not be NONE')
    if inpath:
        indata = pd.read_csv(inpath, 
                             compression='gzip', 
                             header=0, sep=',', 
                             quotechar='"')
        return indata


def group_columns_by_given(datafr:pd.DataFrame,columtogroupby:str='hour'):
    """Group columns by given grouper"""
    datafr = datafr.groupby(by=columtogroupby).sum()
    return datafr.copy(deep=True)


def qc_dataframe(datframetoanalyse:pd.DataFrame):
    """bit of std vis"""
    print(datframetoanalyse.shape)
    print(datframetoanalyse.head)
    datframetoanalyse.hist(bins=20)
    for col in datframetoanalyse.columns.tolist():
        print('NaNs in '+col+' : ',datframetoanalyse[col].isna().sum())
    
    
def specific_filter(dataframe:pd.DataFrame,columnsnamestofilter:list=['hour','click']):
    """Task specific filter"""
    newdf = dataframe[columnsnamestofilter]
    return newdf


def rolling_z_score(dataframe:pd.DataFrame,windowsize:int=3,columntonanalyze:str='click'):
    """Z score = (x -mean) / std. deviation"""
    dataframe['mean'] = dataframe[columntonanalyze].rolling(windowsize).mean()
    dataframe['std'] = dataframe[columntonanalyze].rolling(windowsize).std()
    dataframe['z_Score'] = (dataframe[columntonanalyze] - dataframe['mean']) / dataframe['std']
    return dataframe


def hourly_z_score(dataframe:pd.DataFrame,columntonanalyze:str='click'):
    """Z score = (x -mean) / std. deviation
    calculated for every hour of the day from the cumulative of the ten days
    """
    dataframe['hourly_z_Score'] = (dataframe[columntonanalyze] - dataframe['mean_hourly']) / dataframe['std_hourly']
    return dataframe.copy(deep=True)


def outlier_col(dataframerolme,z_score_cut:float=1.5,columnameforoutlier:str='Outlier',columnameforzsc:str='z_Score'):
    """Apply cut off to zscore and append outlier column True = Outlier, False=Inlier"""
    dataframerolme[columnameforoutlier] = True
    dataframerolme[columnameforoutlier] = dataframerolme[columnameforoutlier][(dataframerolme[columnameforzsc] > z_score_cut) | (dataframerolme[columnameforzsc] < -(z_score_cut)) ]
    dataframerolme[columnameforoutlier] = dataframerolme[columnameforoutlier].fillna(-9999)
    dataframerolme[columnameforoutlier] = dataframerolme[columnameforoutlier].replace(-9999,False)
    return dataframerolme.copy(deep=True)


def convert_todatetime(dataf:pd.DataFrame):
    """No time for transforming hours to datetime and not realy needed beside of plotting things
    """
    #TODO: BUG? --> int type not handled? conversion to str results in same
    dataf['hour'] =   pd.to_datetime(dataf.index,errors='coerce', format='%Y%m%d')
    return dataf    
    

def transpose_concat_daily(dataf:pd.DataFrame,columntonanalyze:str='click'):
    """Helpfer function for concat and transpose to 
    check for outlier on hourly bases. This is for cumulating for each hour of the day all ten days.
    """
    twentyfourhourlist = []
    #Split data in 10 day-lists
    for ix in range(0,len(dataf),24):
        twentyfourhourlist.append(dataf[columntonanalyze].tolist()[ix:ix+24])
        
    twentyfourhourframea = pd.DataFrame(twentyfourhourlist)
    
    dummyyy_mean= pd.DataFrame({'hourly_mean':twentyfourhourframea.mean()}).T
    dummyyyy_std = pd.DataFrame({'hourly_std':twentyfourhourframea.std()}).T

    twentyfourhourframe = pd.concat([dummyyy_mean,dummyyyy_std], 
                                    axis=0, 
                                    ignore_index=False).T
    #return mean and std for every hour of the day --> 24 hours
    return twentyfourhourframe.copy(deep=True)


def concat_hourly_std_mean_with_mainDF(twentyfourhourframe:pd.DataFrame,mainframe:pd.DataFrame,):
    """Helperfunction for concat the df from the 
    24 hour cumulated over ten days with the main dataframe
    """
    container = twentyfourhourframe.copy(deep=True)
    for ixb in range(0,9):
        container = pd.concat([container,twentyfourhourframe],ignore_index=True)
    mainframe['mean_hourly'] = container['hourly_mean'].tolist()
    mainframe['std_hourly'] = container['hourly_std'].tolist()
    return mainframe.copy(deep=True)


def detrend_Data(twframe:pd.DataFrame,columntonanalyze:str='click'):
    """Simplest way of detrending data is to use the delta between two timesteps
    and this is done with this function
    """
    twframe['delta_click'] = np.abs(twframe[columntonanalyze].rolling(window=2).apply(np.diff))
    return twframe

def data_loading_filter(path:str=r"O:\EP\AT\Projects\GPIP\Peisker\13_kaggle\avazu-ctr-prediction\Data"):
    """Helperfunction in case looping over several files is needd in this case I hard code the biggest data file
    by idx 0.
    """
    listofgz = find_all_files_of_type(path)
    dummydf = load_single_gz_file_in_to_pandas(listofgz[0])
    newdf = specific_filter(dummydf)
    return newdf


#%%
if __name__ == "__main__":
    #Loading the thrain.gz from given directory
    newdf = data_loading_filter(r"O:\EP\AT\Projects\GPIP\Peisker\13_kaggle\avazu-ctr-prediction\Data")

    #%%
    grouped_df_per_hours = group_columns_by_given(newdf.copy())

    detrended_grouped_df_per_hours = detrend_Data(grouped_df_per_hours.copy())

    # Looping over both option with trend and very simple detrending 
    for elem in ['click','delta_click']:
        
        preprocessing_for_24_hour_analysis = transpose_concat_daily(detrended_grouped_df_per_hours.copy(),
                                                                    columntonanalyze=elem)
        
        dataframerollingmean = rolling_z_score(detrended_grouped_df_per_hours.copy(),
                                               windowsize=6,
                                               columntonanalyze=elem)

        colmdf_with_bool_outlier = outlier_col(dataframerollingmean.copy(),)

        container_for_hourly_meanstd = concat_hourly_std_mean_with_mainDF(preprocessing_for_24_hour_analysis.copy(),
                                                                          colmdf_with_bool_outlier.copy())

        newcont_hourly_z_score = hourly_z_score(container_for_hourly_meanstd.copy(),columntonanalyze=elem)

        final_container=outlier_col(newcont_hourly_z_score.copy(),
                                    z_score_cut=1.5,
                                    columnameforoutlier='outlier_hourly',
                                    columnameforzsc='hourly_z_Score')

        
        if elem =='click':
            addon = ' with Trend'
        elif elem =='delta_click':
            addon = ' after simple detrend'
            
        # Plot for showing the datapoint over time and fit regression first order for 24 hours mean
        plt.figure(figsize=(15,8))
        plt.title('Regplot of hourly Mean value over 24h from all 10 days'+addon)
        sns.regplot(x=preprocessing_for_24_hour_analysis.index,y='hourly_mean',data=preprocessing_for_24_hour_analysis,order=2)
        plt.legend(loc='upper left',
                   fontsize='x-small')   
        
        # Plot for showing the datapoint over time and fit regression for all hours
        plt.figure(figsize=(15,8))
        plt.title('Regplot of Mean value all Datapoints'+addon)
        sns.regplot(x=final_container.index,y='mean',data=final_container,order=1)
        plt.legend(loc='upper left',
                   fontsize='x-small')   
        plt.figure(figsize=(15,8))
        
        # Plot for showing respective click values over time and color by outlier vs inlier 
        plt.title('Scatterplot of '+elem+' vs INTHours and Detected Ouliers'+addon)
        sns.scatterplot(x=final_container.index,
                        y=elem,
                        data=final_container,
                        style='Outlier',
                        hue='Outlier',
                        size='z_Score')
        
        # Plot for showing respective click mean with window size of 6 hours over time and color by outlier vs inlier 
        plt.figure(figsize=(15,8))
        plt.title('Scatterplot of Mean(ws=6hr) '+elem+' vs INTHours and Detected Ouliers'+addon)
        sns.scatterplot(x=final_container.index,
                        y='mean',
                        data=final_container,
                        style='Outlier',
                        hue='Outlier',
                        size='z_Score')
        plt.legend(loc='upper left')

        # Plot for showing respective click over values over time but with outlier detect on day level
        # day level: 24 hours cummulated for all 10 daysover time and color by outlier vs inlier 
        plt.figure(figsize=(15,8))
        plt.title('Scatterplot of '+elem+' vs INTHours and Ouliers detected for each of hour of the day'+addon)
        sns.scatterplot(x=final_container.index,
                        y=elem,
                        data=final_container,
                        style='outlier_hourly',
                        hue='outlier_hourly',
                        size='hourly_z_Score')
        plt.legend(loc='upper left',
                   fontsize='x-small')
