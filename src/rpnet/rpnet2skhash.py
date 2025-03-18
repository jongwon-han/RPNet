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

np.random.seed(0)

def prep_skhash(cat_df,pol_df,sta_df,out_dir,ftime,fwfid,ctrl0,hash_version='hash2'):

    if os.path.exists(out_dir+'/'+hash_version):
        shutil.rmtree(out_dir+'/'+hash_version)
    os.makedirs(out_dir+'/'+hash_version+'/IN')
    os.makedirs(out_dir+'/'+hash_version+'/OUT')

    cat_df=cat_df.sort_values([fwfid]).reset_index(drop=True)
    sta_df=sta_df.sort_values(['sta']).reset_index(drop=True)

    pol_df.to_csv(out_dir+'/'+hash_version+'/uniq_pol.csv',index=False)

    # First, make station list
    with open(out_dir+'/'+hash_version+'/IN/station.txt','a') as f1:
        for idx,val in sta_df.iterrows():
            # f1.write(str(val.sta.strip()+' '+val.chan.rjust(3,' ')).ljust(42,' '))
            f1.write(str(val.sta).ljust(5,' '))
            f1.write(str(val.chan).rjust(3,' '))
            f1.write(str(' ').rjust(34,' '))
            f1.write(str('%.5f'%val.lat).rjust(8,' '))
            f1.write(str('%.5f'%val.lon).rjust(11,' '))
            f1.write(str(int(val.elv)).rjust(6,' '))
            f1.write(' 1900/01/01 3000/01/01 ')
            f1.write(val.net)
            if not idx==len(sta_df)-1:
                f1.write('\n')

    # Next, make phase file
    with open(out_dir + '/' + hash_version + '/IN/phase.txt', 'a') as f2:
        for idx,val in cat_df.iterrows():
            ot=UTCDateTime(val[ftime])
            ot0=UTCDateTime(year=ot.year,month=ot.month,day=ot.day,hour=ot.hour,minute=ot.minute)
            # line0=f'{otime.year}{otime.month:02d}{otime.day:02d}{otime.hour:02d}{otime.minute:02d}' \
            #         f'{otime.second:02d}.{int(otime.microsecond/10000):02d}{val.lat}{val.lon}{val.dep:5.2f}' \
            #         f'                                                {0:.2f}  {0:.2f}                                        ' \
            #         f'0.0'+str(val[fwfid]).rjust(18,' ')

            if val.lat>0:
                dm_lat='%02d'%int(val.lat)+'N'+'%5.2f'%((val.lat-int(val.lat))*60)
            else:
                dm_lat='%02d'%int(abs(val.lat))+'S'+'%5.2f'%((abs(val.lat)-int(abs(val.lat)))*60)
            if val.lon>0:
                dm_lon='%03d'%int(val.lon)+'E'+'%5.2f'%((val.lon-int(val.lon))*60)
            else:
                dm_lon='%03d'%int(abs(val.lon))+'W'+'%5.2f'%((abs(val.lon)-int(abs(val.lon)))*60)

            line0='%04d'%ot.year+'%02d'%ot.month+'%02d'%ot.day+'%02d'%ot.hour+'%02d'%ot.minute+'%5.2f'%(ot-ot0)\
                  +dm_lat+dm_lon+'%5.2f'%val.dep+str(' ').rjust(88-39,' ')+' 0.00 0.00'+str(' ').rjust(139-99,' ')\
                  +' %4.2f'%val.mag+val[fwfid].rjust(165-143,' ')

            line1=f'                                                                  {val[fwfid]}'

            s_df=pol_df[pol_df[fwfid]==val[fwfid]].drop_duplicates(['sta']).sort_values(['sta']).reset_index(drop=True)
            f2.write(line0+'\n')
            for idx2,val2 in s_df.iterrows():
                sta=sta_df[sta_df.sta0==val2.sta]['sta'].iloc[0]
                net=sta_df[sta_df.sta0==val2.sta]['net'].iloc[0]
                chan=sta_df[sta_df.sta0==val2.sta]['chan'].iloc[0]
                f2.write(sta.ljust(4,' '))
                f2.write(net.rjust(3,' '))
                f2.write(chan.rjust(5,' '))
                f2.write(' I ')
                f2.write(val2.predict)
                f2.write('\n')
            f2.write(line1)
            if not idx==len(cat_df)-1:
                f2.write('\n')

    # Last, make control file
    with open(out_dir + '/' + hash_version + '/control_file.txt', 'a') as f3:
        f3.write('## Control file for SKHASH driver2 (from RPNet result)\n\n')
        f3.write('$input_format  # format of input files\n')
        f3.write(hash_version+'\n\n')
        f3.write('$stfile        # station list filepath\n')
        f3.write(out_dir+ '/' + hash_version+'/IN/station.txt\n\n')
        f3.write('$fpfile        # P-polarity input filepath\n')
        f3.write(out_dir+ '/' + hash_version+'/IN/phase.txt\n\n')
        f3.write('$outfile1      # focal mechanisms output filepath\n')
        f3.write(out_dir+ '/' + hash_version+'/OUT/out.txt\n\n')
        f3.write('$outfile2      # acceptable plane output filepath\n')
        f3.write(out_dir+ '/' + hash_version+'/OUT/out2.txt\n\n')
        f3.write('$outfolder_plots        # figure directory\n')
        f3.write(out_dir+'/'+hash_version+'/OUT/figure\n\n')
        with open(ctrl0,'r') as f4:
            for l in f4:
                f3.write(l)

    return
