# RPNet (Han et al., under review)

This is the repository for the RPNet package, a deep learning model for automatic P-wave first motion determination (Han et al., under review).

A detailed README.md and user manual file will be updated soon. 

The `run_RPNet.py` script predicts P-wave polarity from SAC or MSEED files using the RPNet model and automatically generates input files for SKHASH (Skoumal et al., 2024), a Python software based on HASH (Hardebeck and Shearer, 2002, 2003).

The pretrained model files can be downloaded from the following link:
[Pretrained Model Download](https://drive.google.com/drive/folders/1VlhPiLEx6XKBkmLdkc9RJ6fFTcSD0-0B?usp=sharing)


SKHASH (Skoumal et al., 2024):
https://code.usgs.gov/esc/SKHASH




**Reference**

Hardebeck, J. L., & Shearer, P. M. (2002). A new method for determining first-motion focal mechanisms. Bulletin of the Seismological Society of America, 92(6), 2264-2276.

Hardebeck, J. L., & Shearer, P. M. (2003). Using S/P amplitude ratios to constrain the focal mechanisms of small earthquakes. Bulletin of the Seismological Society of America, 93(6), 2434-2444.

Skoumal, R. J., Hardebeck, J. L., & Shearer, P. M. (2024). SKHASH: A Python Package for Computing Earthquake Focal Mechanisms. Seismological Research Letters, 95(4), 2519-2526.

