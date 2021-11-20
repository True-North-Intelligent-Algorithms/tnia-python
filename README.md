# tnia-python
A collection of useful python utilities from True North Intelligent Algorithms

# Install on Windows

Note: due to issue with 'pip install pyopencl' pyopencl needs to be installed with conda 

run

conda install -c conda-forge pyopencl=2020.3.1  
pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python  

(to include CLIJ Deconvolution you need to explicitly clone that repository and navigate to the locaton of setup.py, as the setup.py is not at the top level)  
   
git clone https://github.com/clij/clij2-fft.git  
cd clij2-fft/python  
pip install .  

# Install on Linux

Note: due to issue with 'pip install pyopencl' on windows it wasn't included in setup.py requirements. pyopencl needs to be installed separate

pip install pyopencl  
pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python  

(to include CLIJ Deconvolution you need to explicitly clone that repository and navigate to the locaton of setup.py, as the setup.py is not at the top level)  
git clone https://github.com/clij/clij2-fft.git  
cd clij2-fft/python  
pip install .  


