# Getting Started
여기서는 도커 환경 구성에 대한 방법을 제공하지 않습니다.
여기서는 Anaconda 환경을 구성하는 방법을 제공합니다.


## Anaconda
### 0. Preliminary
Python, conda 환경, PyTorch 관련 라이브러리가 모두 설치가 되어있음을 가정합니다.
* PyTorch 2.0을 포함한 PyTorch 1.13 이상 버전을 권장.
* Python 3.8 이상 버전을 권장.

```bash
# torch install example
# please refer to https://pytorch.org/get-started/previous-versions/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


### 1. Package Installation
다음과 같은 명령어로 pacakge를 설치합니다.
```bash
pip3 install -r requirements.txt
```