#!/bin/bash
set -uexo pipefail

# For Ubuntu, it's far easier to use conda since it ships with the correct CUDA version.
# Before running this script, be sure to do:
#   conda create -n py3-bullet python=3.7
#   conda activate py3-bullet
# Then run this script:: ./install_python_ubuntu.sh
# We need to upgrade tensorflow-addons so that it doesn't throw the error:
#   https://github.com/tensorflow/addons/issues/1132
# To use --disp, run `sudo Xorg :1` in a persistent GNU screen session.

echo "Installing Python libraries..."
conda install ipython
conda install tensorflow-gpu==2.2

pip install pybullet==3.0.4
pip install packaging==19.2
pip install matplotlib==3.1.1
pip install opencv-python==4.1.2.30
pip install meshcat==0.0.18
pip install transformations==2020.1.1
pip install scikit-image==0.17.2
pip install gputil==1.4.0
pip install circle-fit==0.1.3

pip install tensorflow-addons==0.11.1
pip install tensorflow_hub==0.8.0

pip install -e .
