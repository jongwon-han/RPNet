"""
# RPNet (v.0.1.0)
https://github.com/jongwon-han/RPNet

RPNet: Robust P-wave first-motion polarity determination using deep learning (Han et al., 2025; SRL)
doi: https://doi.org/10.1785/0220240384

Prepare SKHASH input files from RPNet results

- Jongwon Han (@KIGAM)
- jwhan@kigam.re.kr
- Last update: 2025. 5. 15.
"""

###############################################

import pandas as pd
import numpy as np
import tensorflow as tf
import parmap
import matplotlib.pyplot as plt
import tqdm
import matplotlib.ticker as ticker
import os
import subprocess
import shutil
from obspy import Stream, Trace
from obspy import UTCDateTime, read
import plotly.figure_factory as ff
import matplotlib
import fnmatch

np.random.seed(0)

def prep_skhash(cat_df,pol_df,amp,sta_df,out_dir,ftime,fwfid,ctrl0,hash_version='hash2'):

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
            f1.write(' 1900/01/01 3000/01/01 ') # If you need, please specify the time range of the station
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

    # Next, make amplitude file for hash3
    if hash_version=='hash3':
        with open(out_dir + '/' + hash_version + '/IN/amp.txt', 'w', encoding='UTF-8') as f:
            for i, name in enumerate(amp):
                if i != len(amp) - 1:
                    f.write(name + '\n')
                else:
                    f.write(name)
        pass

    # Last, make control file
    if hash_version=='hash2':
        with open(out_dir + '/' + hash_version + '/control_file.txt', 'a') as f3:
            f3.write('## Control file for SKHASH driver2 (from RPNet result)\n\n')
            f3.write('$input_format  # format of input files\n')
            f3.write(hash_version+'\n\n')
            f3.write('$stfile        # station list filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/IN/station.txt\n\n')
            f3.write('$fpfile        # P-polarity input filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/IN/phase.txt\n\n')
            f3.write('$outfile1      # focal mechanisms output filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/OUT/out.csv\n\n')
            f3.write('$outfile2      # acceptable plane output filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/OUT/out2.csv\n\n')
            f3.write('$outfolder_plots        # figure directory\n')
            f3.write(out_dir+'/'+hash_version+'/OUT/figure\n\n')
            with open(ctrl0,'r') as f4:
                for l in f4:
                    f3.write(l)
    elif hash_version=='hash3':
        with open(out_dir + '/' + hash_version + '/control_file.txt', 'a') as f3:
            f3.write('## Control file for SKHASH driver3 (from RPNet result)\n\n')
            f3.write('$input_format  # format of input files\n')
            f3.write(hash_version+'\n\n')
            f3.write('$stfile        # station list filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/IN/station.txt\n\n')
            f3.write('$fpfile        # P-polarity input filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/IN/phase.txt\n\n')
            f3.write('$ampfile       # amplitude input filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/IN/amp.txt\n\n')
            f3.write('$outfile1      # focal mechanisms output filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/OUT/out.csv\n\n')
            f3.write('$outfile2      # acceptable plane output filepath\n')
            f3.write(out_dir+ '/' + hash_version+'/OUT/out2.csv\n\n')
            f3.write('$outfolder_plots        # figure directory\n')
            f3.write(out_dir+'/'+hash_version+'/OUT/figure\n\n')
            with open(ctrl0,'r') as f4:
                for l in f4:
                    f3.write(l) 

    return



#################################################################################################
#################################################################################################
# Amplitude Ratio modules


def preprocess(p_time, s_time, stream_path, sp_win, low_freq=1.0, high_freq=20.0, taper_pct=0.01):
    """
    Loading and preprocessing the stream data.
    """
    st = read(stream_path)
    st.trim(starttime=p_time + sp_win[0] - 5, endtime=s_time + sp_win[-1] + 10)
    st.filter('bandpass', freqmin=low_freq, freqmax=high_freq)
    if st[0].stats.sampling_rate != 100:
        st.resample(100)
    st.normalize(global_max=True)
    st.detrend('demean')
    st.detrend('linear')
    st.taper(taper_pct)
    st.sort()
    for tr in st:
        tr.data *= 1e3
    return st

def calc_amplitude(p_time, s_time, station, st,sp_win):
    """
    Calcualte the amplitude ratio of each window in the 3-channel stream,
    then calculate the vector sum of N (nosie), P, S components and S/P ratio, and return as a formatted string.
    """
    # If less than 3 channels, return None
    if len(st) != 3:
        return "None"
    try:
        n_vals, p_vals, s_vals = [], [], []
        # Calculate the amplitude of each window
        for trace in st:
            n_win = trace.slice(p_time + sp_win[0], p_time + sp_win[1]).copy()
            p_win = trace.slice(p_time + sp_win[2], p_time + sp_win[3]).copy()
            s_win = trace.slice(s_time + sp_win[4], s_time + sp_win[5]).copy()
            n_vals.append(max(n_win) - min(n_win))
            p_vals.append(max(p_win) - min(p_win))
            s_vals.append(max(s_win) - min(s_win))
        # Calculate the vector sum of N, P, S components and S/P ratio
        N = np.sqrt(sum(val ** 2 for val in n_vals))
        P = np.sqrt(sum(val ** 2 for val in p_vals))
        S = np.sqrt(sum(val ** 2 for val in s_vals))
        sp_ratio = S / P if P != 0 else float('inf')
        # Example: sta, chan, net, 0.0, 0.0, N, N, P, S
        return f"{station.sta:4s} {station.chan:3s} {station.net:2s} {0.0:4.1f} {0.0:4.1f} {N:18.3f} {N:10.3f} {P:10.3f} {S:10.3f}"
    
    except Exception:
        return "None"

def prepare_amplitudes(params):
    """
    Import the tuple (station_df, pick_id, event_info, data_dir,sp_freq,sp_win) and return the list of amplitude results for each station.
    """
    station_df, pick_id, event_info, data_dir,sp_freq,sp_win = params
    # Remove rows without necessary columns and remove duplicates
    station_df = station_df.drop_duplicates(subset=['sta0'])
    station_df = station_df[station_df['ptime'].notnull() & station_df['stime'].notnull()].reset_index(drop=True)
    
    amp_list = []
    for _, row in station_df.iterrows():
        p_time = UTCDateTime(row.ptime)
        s_time = UTCDateTime(row.stime)
        stream_path = f"{data_dir}/{pick_id}/*{row.sta0}.*"
        st = preprocess(p_time, s_time, stream_path, sp_win, sp_freq[0],sp_freq[1])
        amp_str = calc_amplitude(p_time, s_time, row, st,sp_win)
        if amp_str != "None":
            amp_list.append(amp_str)
    
    amp_list.sort()
    header = f"{event_info} {len(amp_list)}"
    amp_list.insert(0, header)
    
    return amp_list

#################################################################################################
#################################################################################################