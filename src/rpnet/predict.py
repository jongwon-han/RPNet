"""
# RPNet (v.0.0.2)
https://github.com/jongwon-han/RPNet

RPNet: Robust P-wave first-motion polarity determination using deep learning (Han et al., 2025; SRL)
doi: https://doi.org/10.1785/0220240384

Main function modules

- Jongwon Han (@KIGAM)
- jwhan@kigam.re.kr
- Last update: 2025. 3. 18.
"""

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
from obspy import UTCDateTime,read
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
import matplotlib
import fnmatch
from obspy.taup import TauPyModel
import traceback

np.random.seed(0)

# TauP model
model = TauPyModel(model="ak135")

def change2taup(vals):
    idx, val, cat, ftime=vals
    p_est = model.get_travel_times_geo(cat.dep, cat.lat, cat.lon, val.lat,val.lon, phase_list=['P', 'Pn', 'p'])[0].time
    p_est = UTCDateTime(cat[ftime])+p_est
    return p_est


def wf2matrix(vals):
    try:
        idx,val,fwfid,fptime,wf_dir,out_dir=vals
        st=read(wf_dir+'/'+val[fwfid]+'/'+val.sta+'.*')
        st=st.select(channel='*Z')
        if not st[0].stats.sampling_rate==100:
            st.interpolate(100)
        if not len(st)==0:
            st.merge()
        st.filter('highpass',freq=1)
        st.trim(val[fptime]-2.5,val[fptime]+2.5).normalize()

        # if len(st[0].data)<500:
        #     return

        if not os.path.exists(out_dir+'/MSEED/'+val[fwfid]):
            try:
                os.makedirs(out_dir+'/MSEED/'+val[fwfid])
            except:
                pass
        st.write(out_dir+'/MSEED/'+val[fwfid]+'/'+val.sta+'.mseed',format='MSEED')
        return idx,st[0].data[0:500][np.newaxis,:]
    except:
        print(traceback.format_exc())
        return

# custom slicing
def pad_slice(a, start, end):
    r, c = a.shape
    s, e = max(start, 0), min(end, c)
    p = ((0, 0), (max(-start, 0), max(end - c, 0)))
    return np.pad(a[:, s:e], p, 'constant')

# in: (none,400,1)
# out: (none,3)
def prep_input(data_mat,ref_mat):
    rlst=[pad_slice(np.expand_dims(data_mat[i,:],axis=0),ref_mat[i]-200,ref_mat[i]+200)
                     for i in range(data_mat.shape[0])]
    rmat=np.vstack(rlst)
    # normalize
    row_maxes = np.max(np.abs(rmat), axis=1).reshape(-1, 1)
    rmat = rmat / row_maxes
    rmat=np.expand_dims(rmat,axis=-1)
    return rmat

def pred_model(model,in_mat,batch_size=None,itr_pred=0):
    # model.summary()
    if itr_pred==0:
        if batch_size==None:
            p_mat=model(in_mat)
        else:
            p_mat = model.predict(in_mat, batch_size=batch_size,verbose=1)
        # keepe Unknown
        p_mat = p_mat[:, :3]
        i_mat = np.argmax(p_mat, axis=1)
        v_mat = np.max(p_mat, axis=1)
        r_df = pd.DataFrame({'predict': i_mat, 'prob': np.round(v_mat,4)})
        r_df['predict'] = r_df['predict'].replace({0: 'U', 1: 'D', 2: 'K'})
    else:
        print('# iterate prediction')
        itrs=[]
        for itr in tqdm.tqdm(range(itr_pred)):
            if batch_size == None:
                p_mat = model(in_mat)
            else:
                p_mat = model.predict(in_mat, batch_size=batch_size,verbose=0)
                itrs.append(p_mat[np.newaxis,:,:])
        p_mat0=np.vstack(itrs)
        p_mat=np.nanmean(p_mat0,axis=0)
        std_mat=np.nanstd(p_mat0,axis=0)

        # keep Unknown
        p_mat = p_mat[:, :3]
        i_mat = np.argmax(p_mat, axis=1)
        v_mat = np.max(p_mat, axis=1)
        s_mat = std_mat[np.arange(std_mat.shape[0]),i_mat]
        r_df = pd.DataFrame({'predict': i_mat, 'prob': np.round(v_mat,4),'std':np.round(s_mat,4)})
        r_df['predict'] = r_df['predict'].replace({0: 'U', 1: 'D', 2: 'K'})
    return r_df

def pred_rpnet(model,in_mat,metadata,batch_size=2**13,iteration=100,gpu_num=-1,time_shift=0.5,mid_point=250):

    if os.path.exists(model):
        model = tf.keras.models.load_model(model,custom_objects=SeqSelfAttention.get_custom_objects())
    else:
        print('[QUIT!] Can not find model file: ',model)
        quit()

    data_mat = in_mat
    df = metadata

    # if True:
    #     print('- augment with flipped data')
    #     data_mat2 = data_mat.copy() * -1
    #     df2 = df.copy()
    #     df2['pol'] = df2['pol'].replace({'U': 'D', 'D': 'U'})
    #     data_mat = np.vstack([data_mat, data_mat2])
    #     xmat2 = []
    #     df = pd.concat([df, df2], axis=0).reset_index(drop=True)
    #     df2 = []

    r_df0=df.copy()

    # mid point of matrix (if you want shift matrix, change time_shift)
    if not time_shift == 0:
        rand_mid = [int(mid_point + np.random.randint(int(-100 * time_shift), int(time_shift * 100))) for i in
                    range(data_mat.shape[0])]
    else:
        rand_mid = [int(mid_point) for i in range(data_mat.shape[0])]

    # Predict RPNet
    in_mat = prep_input(in_mat, rand_mid)
    r_df = pred_model(model, in_mat, batch_size=batch_size, itr_pred=iteration)
    r_df = pd.concat([r_df0, r_df], axis=1)

    # r_df2=r_df.copy()
    # r_df2=r_df2[r_df2['pol']!='K'].reset_index(drop=True)
    # matches = r_df2['pol'] == r_df2['predict']
    # r_df['bool'] = matches
    # match_percentage = matches.mean() * 100
    # print(f"Matching Percentage: {match_percentage:.2f}%")
    #
    # print('@ number of U: ', len(r_df[r_df.pol == 'U']))
    # print('@ number of D: ', len(r_df[r_df.pol == 'D']))

    # print('@ check NaN results')
    # print(r_df[pd.isnull(r_df['prob'])])

    return r_df
