# Getting Started
Docker environment setup is not provided separately.
Here, we provide instructions for setting up a anconda environment.


## Anaconda

### 0. Preliminary
We assume that both the conda environment and Python, as well as the PyTorch-related libraries, are already installed.
* We recommend PyTorch 1.13 and above including PyTorch 2.0.
* We recommend Python 3.8 and above.

```bash
# torch install example
# please refer to https://pytorch.org/get-started/previous-versions/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


### 1. Package Installation

We install packages using the following command:
```bash
pip3 install -r requirements.txt
```