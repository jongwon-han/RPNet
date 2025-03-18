import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rpnet",
    version="0.0.2",
    author="Jongwon_Han",
    author_email="jwhan@kigam.re.kr",
    description="Robust P-wave first motion determination using deep learning (Han et al., 2025; SRL)",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/jongwon-han/RPNet",
    project_urls={
        "Bug Tracker": "https://github.com/jongwon-han/RPNet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        'pandas==1.4.4',
        'h5py==3.1.0',
        'numpy==1.19.5',
        'parmap==1.7.0',
        'tensorflow==2.7.0',
        'tensorflow-gpu==2.7.0',
        'keras-self-attention==0.50.0',
        'matplotlib==3.6.3',
        'tqdm==4.66.2',
        'obspy==1.3.1',
        'scikit-learn==1.6.1',
        'plotly==5.19.0',
        'protobuf==3.20.0',
        'notebook==7.3.2'
    ],
)
