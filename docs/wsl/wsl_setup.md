## Windows Subystem for Linux 2

Note:  We don't install the GUI related components (Napari and pyqt) because those can be accessed straight from Windows.  The use case is to use WSL2 Ubuntu for training models.  So only need the processing code. 

Instruction to install WSL2 are [here](https://learn.microsoft.com/en-us/windows/wsl/install)

Note that we want WSL2 which should be installed by default (but double check to make sure). 

Once WSL2 Ubuntu is installed open it and type ```nivdia-smi```.  It should automatically have installed NVIDIA drivers and ```nvidia-smi``` should output a short report, but if no ```nvidia-smi``` install graphsics drivers.  

Then install mini-conda 

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

You will also need to install gcc and g++ compiler and this can be done by installing ```build-essential```

```
sudo apt update
sudo apt install build-essential
```

Then install the deep learning dependencies you want to work with. 

### Running in Visual Studio code

The nice thing is that you can access Ubuntu through the VS Code GUI.  Search for the 'WSL' plugin, install it then run ' `Ctrl + Shift + P` (command palette) and find ```WSL: Connect to WSL (in new windows)```
