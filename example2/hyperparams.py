"""
# RPNet (v.0.1.0)
https://github.com/jongwon-han/RPNet

RPNet: Robust P-wave first-motion polarity determination using deep learning (Han et al., 2025; SRL)
doi: https://doi.org/10.1785/0220240384

Example2 script to run the sample dataset (hash driver3; hyperparameters)

- Jongwon Han (@KIGAM)
- jwhan@kigam.re.kr
- Last update: 2025. 5. 15.
"""


#########################################################################################################
""" PARAMETER SETTING """

# Pre-trained RPNet model. Please specify exact file path.
model='../model/RPNet_v1.h5' # Pretrained model

wf_dir='./waveform' # directory should be in $wf_dir/fwfid/station.* order

event_catalog='./Buan_catalog.csv' # CSV file of event catalog (Origin location should be in lat/lon/dep header names)

phase_metadata='./Buan_phase.csv' # CSV file of phases metadata

sta_metadata='./Buan_station.csv' # CSV file of station metadata (net/sta/chan/lat/lon/elv)

out_dir='./output01' # output directory

ctrl0='./control_file0.txt' # default and other params for SKHASH

ftime='utc' # header of origin time in event catalog

fwfid='pick' # header of waveform ID in event/phase catalog

fptime='ptime' # header of P arrival time in phase catalog

fstime='stime' # header of P arrival time in phase catalog

cores=5 # multiprocessing cores

batch_size=2**13 # batch size for dataset

iteration=100 # Iterative prediction (Mean/STD), If 0 it will produce deterministic prediction value

gpu_num="" # GPU number / If use cpu make it empty "" / If dataset is small, CPU is much faster

std_threshold=0.02 # std threshold for iterative prediction when making SKHASH (if iteration is not 0)

mean_threshold=0.95 # mean threshold for iterative prediction when making SKHASH (if iteration is not 0; if not use, set 0)

rm_unknwon=True # remove unknown result when making SKHASH

change2taup=True # change reference P time to estimated arrival time using TauP

add_sta=True # When you want to add stations that exist as waveform files but are not included in the phase list / If True, change2taup also must be True.

keep_initial_phase=True # During TauP estimation, keep initial phase (P/S). If True, only empty phase will be estimated.

taup_model='iasp91' # TauP model (iasp91, ak135, prem, ...) / If you want to use custom model, please specify the path (e.g., '/home/jwhan/srkim_iasp.npz')

hash_version='hash3' # hash2: only P polarity, hash3: P polarity ans S/P ratio (refer to SKHASH for more details)

sp_freq=(1,20) # Bandpass filter frequency range for S/P ratio (e.g., (1, 20)) [Hz]

sp_win=[-2.5,-0.5,-0.5,+1.0,-0.5,+2.5] # S/P ratio window (before/after) for noise, P and S (e.g., [-2.5,-0.5,-0.5,+1.0,-0.5,+2.5]) [s]. 
# Noise and P-window are before/after the P arrival time, and S-window is before/after the S arrival time.
#########################################################################################################
