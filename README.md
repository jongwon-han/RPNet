# RPNet (Han et al., 2025; SRL)

This is the repository for the RPNet package, a deep learning model for automatic P-wave first motion determination (Han et al., 2025; SRL).

The code is currently being packaged with detailed user manual, and the official release is planned before the end of March. 
Please take this into consideration when using it.

The `run_RPNet.py` script predicts P-wave polarity from SAC or MSEED files using the RPNet model and automatically generates input files for SKHASH (Skoumal et al., 2024), a Python software based on HASH (Hardebeck and Shearer, 2002, 2003).

The pretrained model files can be downloaded from the following link:
[Pretrained Model Download](https://drive.google.com/drive/folders/1VlhPiLEx6XKBkmLdkc9RJ6fFTcSD0-0B?usp=sharing)


SKHASH (Skoumal et al., 2024):
https://code.usgs.gov/esc/SKHASH


---

**Reference**

Han, J., S, Kim, & D.-H. Sheen (in review), RPNet: Robust P-wave first-motion polarity determination using deep learning. Seismological Research Letters; doi: https://doi.org/10.1785/0220240384

Hardebeck, J. L., & Shearer, P. M. (2002). A new method for determining first-motion focal mechanisms. Bulletin of the Seismological Society of America, 92(6), 2264-2276.

Hardebeck, J. L., & Shearer, P. M. (2003). Using S/P amplitude ratios to constrain the focal mechanisms of small earthquakes. Bulletin of the Seismological Society of America, 93(6), 2434-2444.

Skoumal, R. J., Hardebeck, J. L., & Shearer, P. M. (2024). SKHASH: A Python Package for Computing Earthquake Focal Mechanisms. Seismological Research Letters, 95(4), 2519-2526.

