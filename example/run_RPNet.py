"""
# RPNet (v.0.0.2)
https://github.com/jongwon-han/RPNet

RPNet: Robust P-wave first-motion polarity determination using deep learning (Han et al., 2025; SRL)
doi: https://doi.org/10.1785/0220240384

Example script to run the sample Hi-net dataset

- Jongwon Han (@KIGAM)
- jwhan@kigam.re.kr
- Last update: 2025. 3. 18.
"""

#########################################################################################################

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
import time
from rpnet import *
from hyperparams import *

#########################################################################################################

# set gpu number
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

stime=time.time()

# make output directory / if exist remove it
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
shutil.copy2(__file__,out_dir) # copy main script

# load raw data (catalog, phase, station files)
cat_df=pd.read_csv(event_catalog)
pha_df=pd.read_csv(phase_metadata)
sta_df=pd.read_csv(sta_metadata).sort_values(['sta']).reset_index(drop=True)
sta_df['sta0']=sta_df['sta']

# add empty pick stations
if add_sta:
    z_files=sorted(glob.glob(wf_dir+'/*/*Z'))
    def get_add(z):
        id=z.split('/')[-2]
        sta=z.split('/')[-1].split('.')[0]
        if len(pha_df[(pha_df[fwfid]==id)&(pha_df['sta']==sta)])!=0:
            return pd.DataFrame()
        if not id in cat_df[fwfid].to_list():
            return pd.DataFrame()
        return pd.DataFrame({'pick':[id],'sta':[sta],'time':[np.nan],'pha':['P']})
    print('# get list of additional stations')
    results=parmap.map(get_add,z_files, pm_pbar=True, pm_processes=cores,pm_chunksize=1)
    pha_df=pd.concat([pha_df]+results)

# Add station metadata to phase df
print('\n# Arrange metadata')
pha_df=pha_df[pha_df['sta'].isin(sta_df['sta'].to_list())].reset_index(drop=True)
pha_df['lat']=[sta_df[sta_df.sta==i]['lat'].iloc[0] for i in pha_df['sta'].to_list()]
pha_df['lon']=[sta_df[sta_df.sta==i]['lon'].iloc[0] for i in pha_df['sta'].to_list()]
pha_df['elv']=[sta_df[sta_df.sta==i]['elv'].iloc[0] for i in pha_df['sta'].to_list()]
# pha_df['net']=[sta_df[sta_df.sta==i]['net'].iloc[0] for i in pha_df['sta'].to_list()]
# pha_df['chan']=[sta_df[sta_df.sta==i]['chan'].iloc[0] for i in pha_df['sta'].to_list()]
sta_df['net']='HI' # Renaming, just for consistency

# make UTCDateTime objects
cat_df[ftime]=[UTCDateTime(i) for i in cat_df[ftime].to_list()]
pha_df[fptime]=[UTCDateTime(i) for i in pha_df[fptime].to_list()]
print('- Done')

# Change to TauP P arrival times (OPTION; considering pick uncertainty)
if change2taup:
    print('\n\n# change to TauP arrival')
    pha_df['ptime0']=pha_df[fptime]
    results=parmap.map(change2taup,[[idx,val,cat_df[cat_df[fwfid]==val[fwfid]].iloc[0],ftime] for idx,val in pha_df.iterrows()]
                       , pm_pbar=True, pm_processes=cores,pm_chunksize=1)
    pha_df[fptime]=results
    print('- Done')
# Make input data matrix
print('\n\n# Make input matrix from waveform data')
results=parmap.map(wf2matrix,[[idx,val,fwfid,fptime,wf_dir,out_dir] for idx, val in pha_df.iterrows()], pm_pbar=True, pm_processes=cores,pm_chunksize=1)
results=[i for i in results if i is not None]
a,b=zip(*results)
pha_df=pha_df.iloc[list(a)].reset_index(drop=True)
in_mat=np.vstack(b)
print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
print('- Done')

# RPNet main prediction
print('\n\n# Predict polarity (RPNet)')
r_df=pred_rpnet(model,in_mat,pha_df,batch_size=batch_size,iteration=iteration,gpu_num=gpu_num,time_shift=0.0,mid_point=250)
print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
r_df.to_csv(out_dir+'/pol_result.csv',index=False)
print('- Done')
# Renaming station code (only for Hi-net)
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
# make threshold for mean
if iteration!=0 and mean_threshold!=0:
    r_df=r_df[r_df['prob']>=mean_thresuld].reset_index(drop=True)
print('\n\n# Final result:')
print(r_df.to_string())

r_df=r_df.drop_duplicates(['sta',fwfid]).reset_index(drop=True)
prep_skhash(cat_df=cat_df,pol_df=r_df,sta_df=sta_df,ftime=ftime,fwfid=fwfid,ctrl0=ctrl0,out_dir=out_dir)
print('% calculation time (min): ','%.2f'%((time.time()-stime)/60))
print('\n\n@ ALL DONE!')
