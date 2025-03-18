## RPNet (Han et al., 2025; SRL)

This is the repository for the RPNet package, a deep learning model for automatic P-wave first motion determination.

The RPNet predicts P-wave polarity from SAC or MSEED files using the pre-trained model and automatically generates input files for SKHASH (Skoumal et al., 2024), a Python software based on HASH (Hardebeck and Shearer, 2002, 2003).

[SKHASH (Skoumal et al., 2024)](https://code.usgs.gov/esc/SKHASH)


---
### Installation
The various dependencies required to run the **RPNet** can be easily installed using **pip**.<br>
It is recommended to run the program in a separate virtual environment using Anaconda with python version 3.9.<br>

**Note**: If you want to use a GPU, you must install **CUDA** libaray. The RPNet was developed using CUDA version 11.1.74.

**In terminal:**<br>

```sh
# create conda environment
conda create -n rpnet python=3.9

# activate
conda activate rpnet

# install RPNet using pip
pip install rpnet

# download (clone) example files (github)
git clone https://github.com/jongwon-han/RPNet
```

Next, open the Jupyter notebook **run_RPNet.ipynb** in the example directory to refer to the example files and tutorial.

RPNet supports multiprocessing-based preprocessing using the **parmap** module.<br>

---
### Dependencies
- 'pandas==1.4.4'
- 'h5py==3.1.0'
- 'numpy==1.19.5',
- 'parmap==1.7.0',
- 'tensorflow==2.7.0',
- 'tensorflow-gpu==2.7.0',
- 'keras-self-attention==0.50.0',
- 'matplotlib==3.6.3',
- 'tqdm==4.66.2',
- 'obspy==1.3.1',
- 'scikit-learn==1.6.1',
- 'plotly==5.19.0',
- 'protobuf==3.20.0',
- 'notebook==7.3.2'

---
### Upcoming Features
I am planning to add the following features in future updates:<br>
🚀  Training (scratch) & Re-training (fine-tuning/transfer learning) modules<br>
🚀  Automatic estimation of S/P ratio for the focal mechanism calculation (SKHASH)<br>

I'm working on this and will update it soon!

---
### Reference

Han, J., S, Kim, & D.-H. Sheen (in review), RPNet: Robust P-wave first-motion polarity determination using deep learning. Seismological Research Letters; doi: https://doi.org/10.1785/0220240384

Hardebeck, J. L., & Shearer, P. M. (2002). A new method for determining first-motion focal mechanisms. Bulletin of the Seismological Society of America, 92(6), 2264-2276.

Hardebeck, J. L., & Shearer, P. M. (2003). Using S/P amplitude ratios to constrain the focal mechanisms of small earthquakes. Bulletin of the Seismological Society of America, 93(6), 2434-2444.

Skoumal, R. J., Hardebeck, J. L., & Shearer, P. M. (2024). SKHASH: A Python Package for Computing Earthquake Focal Mechanisms. Seismological Research Letters, 95(4), 2519-2526.

