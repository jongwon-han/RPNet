"""
# RPNet (v.0.0.1)
https://github.com/jongwon-han/RPNet

RPNet: Robust P-wave first-motion polarity determination using deep learning (Han et al., 2025; SRL)
doi: https://doi.org/10.1785/0220240384

Example script to run the sample Hi-net dataset (hyperparameters)

- Jongwon Han (@KIGAM)
- jwhan@kigam.re.kr
- Last update: 2025. 2. 24.
"""


#########################################################################################################
""" PARAMETER SETTING """

# Pre-trained RPNet model. Please specify exact file path.
model='../model/RPNet_v1.h5' # Pretrained model

wf_dir='./waveform' # directory should be in ~/waveformID/station.* order

event_catalog='./Kumamoto_catalog.csv' # CSV file of event catalog (Origin location should be in lat/lon/dep header names)

phase_metadata='./Kumamoto_phase.csv' # CSV file of phases metadata

sta_metadata='./hinet_station.csv' # CSV file of station metadata (net/sta/chan/lat/lon/elv)

out_dir='./output01' # output directory

ctrl0='./control_file0.txt' # default and other params for SKHASH

ftime='jst' # header of origin time in event catalog

fwfid='data_id' # header of waveform ID in event/phase catalog

fptime='ptime' # header of P arrival time in phase catalog

cores=5 # multiprocessing cores

batch_size=2**13 # batch size for dataset

iteration=100 # Iterative prediction (Mean/STD), If 0 it will produce deterministic prediction value

gpu_num=-1 # GPU number / If use cpu make it -1 / If dataset is small, CPU is much faster

std_threshold=0.2 # std threshold for iterative prediction when making SKHASH (if iteration is not 0)

mean_threshold=0 # mean threshold for iterative prediction when making SKHASH (if iteration is not 0; if not use, set 0)

rm_unknwon=True # remove unknown result when making SKHASH

change2taup=False # change reference P time to estimated arrival time using TauP

add_sta=False # When you want to add stations that exist as waveform files but are not included in the phase list
#########################################################################################################
