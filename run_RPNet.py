# Script for RPNet prediction
# from SAC or MSEED files
# This script produces SKHASH input and control files
#
# - Jongwon Han (@KIGAM)
# - jongwony@korea.ac.kr
# - 2024.03.14
###############################################

import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import parmap
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import subprocess
import shutil
from obspy import Stream, Trace
from obspy import UTCDateTime
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
import matplotlib
import fnmatch
from predict import *
from rpnet2skhash import *
import time

#########################################################################################################
""" PARAMETER SETTING """
# Origin location should be in lat/lon/dep header names
model='./RPNet_v1.h5' # Pretrained model
# model='/home/jwhan/PycharmProjects/workspace02/proj_KFpol/benchmark/models/pd_model_scsn.h5' # APP SCSN model
wf_dir='./example/SAC' # directory should be in ~/waveformID/station.* order
event_catalog='./example/Kumamoto_catalog.csv' # CSV file of event catalog
phase_metadata='./example/Kumamoto_phase.csv' # CSV file of phases metadata
sta_metadata='./example/station.csv' # CSV file of station metadata (net/sta/chan/lat/lon/elv)
out_dir='./example/rpnet_hinet' # output directory
ctrl0='./control_file0.txt' # default and other params for SKHASH
ftime='jst' # header of origin time in event catalog
fwfid='data_id' # header of waveform ID in event/phase catalog
fptime='ptime' # header of P arrival time in phase catalog
cores=100 # multiprocessing cores
batch_size=2**13 # batch size for dataset
iteration=0 # Iterative prediction (Mean/STD), If 0 it will produce deterministic prediction value
gpu_num=0 # GPU number / If use cpu make it -1 / If dataset is small, CPU is much faster
std_threshold=0.2 # std threshold for iterative prediction when making SKHASH (if iteration is not 0)
rm_unknwon=True # remove unknown result when making SKHASH
#########################################################################################################

# set gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

stime=time.time()

# make output directory / if exist remove it
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
shutil.copy2(__file__,out_dir)

# load raw data
cat_df=pd.read_csv(event_catalog)
pha_df=pd.read_csv(phase_metadata)
sta_df=pd.read_csv(sta_metadata).sort_values(['sta']).reset_index(drop=True)
sta_df['sta0']=sta_df['sta']

# Filter the data with lat/lon boundary
# print('# Get area bound')
# cat_df=cat_df[(cat_df.lat>=35.4518)&(cat_df.lat<36.6906)&(cat_df.lon>=139.2384)&(cat_df.lon<140.4924)].reset_index(drop=True)
# pha_df=pha_df[pha_df[fwfid].isin(cat_df[fwfid].to_list())].reset_index(drop=True)
# print(cat_df)

# Add station metadata to phase df
print('# Arrange metadata')
pha_df=pha_df[pha_df['sta'].isin(sta_df['sta'].to_list())].reset_index(drop=True)
pha_df['lat']=[sta_df[sta_df.sta==i]['lat'].iloc[0] for i in pha_df['sta'].to_list()]
pha_df['lon']=[sta_df[sta_df.sta==i]['lon'].iloc[0] for i in pha_df['sta'].to_list()]
pha_df['elv']=[sta_df[sta_df.sta==i]['elv'].iloc[0] for i in pha_df['sta'].to_list()]
sta_df['net']='HI'

# make UTCDateTime objects
cat_df[ftime]=[UTCDateTime(i) for i in cat_df[ftime].to_list()]
pha_df[fptime]=[UTCDateTime(i) for i in pha_df[fptime].to_list()]

# Change to TauP P arrival times (OPTION; considering pick uncertainty)
print('# change to TauP arrival')
pha_df['ptime0']=pha_df[fptime]
results=parmap.map(change2taup,[[idx,val,cat_df[cat_df[fwfid]==val[fwfid]].iloc[0],ftime] for idx,val in pha_df.iterrows()]
                   , pm_pbar=True, pm_processes=cores,pm_chunksize=1)
pha_df[fptime]=results

pha_df['dt']=[UTCDateTime(val.ptime0)-UTCDateTime(val[fptime]) for idx,val in pha_df.iterrows()]
plt.figure()
plt.hist(pha_df['dt'],range=(0,4),bins=400)
plt.xlabel('Time shift (s)')
plt.ylabel('Counts')
# plt.show()

# Make input data matrix
print('# Make input matrix from waveform data')
results=parmap.map(wf2matrix,[[idx,val,fwfid,fptime,wf_dir] for idx, val in pha_df.iterrows()], pm_pbar=True, pm_processes=cores,pm_chunksize=1)
results=[i for i in results if i is not None]
a,b=zip(*results)
pha_df=pha_df.iloc[list(a)].reset_index(drop=True)
in_mat=np.vstack(b)

# RPNet main prediction
print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
print('# Predict polarity (RPNet)')
r_df=pred_rpnet(model,in_mat,pha_df,batch_size=batch_size,iteration=iteration,gpu_num=gpu_num,time_shift=0.0,mid_point=250)
print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
print(r_df)
r_df.to_csv(out_dir+'/pol_result.csv',index=False)

# only for Hinet
sta_df['net']='HI'
sta_df['chan']=sta_df['chan'].replace('U','HHZ')
sta_df['sta']=['S'+str(i+1).rjust(3,'0') for i in range(len(sta_df))]
cat_df=cat_df
pha_df=pha_df[pha_df['data_id'].isin(cat_df['data_id'].to_list())].reset_index(drop=True)
r_df=pd.read_csv(out_dir+'/pol_result.csv')
r_df=r_df[r_df['data_id'].isin(cat_df['data_id'].to_list())].reset_index(drop=True)

# Make SKHASH input setting
if iteration!=0:
    r_df.loc[r_df['std'] > std_threshold, 'predict'] = 'K'
if rm_unknwon:
    r_df=r_df[r_df['predict']!='K'].reset_index(drop=True)
r_df=r_df.drop_duplicates(['sta',fwfid]).reset_index(drop=True)
prep_skhash(cat_df=cat_df,pol_df=r_df,sta_df=sta_df,ftime=ftime,fwfid=fwfid,ctrl0=ctrl0,out_dir=out_dir)
print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
print('@ ALL DONE!')
