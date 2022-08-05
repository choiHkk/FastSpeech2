## Introduction
1. nvidia-tacotron2 오픈 소스와 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. 기존 오픈소스는 MFA기반 preprocessing을 진행한 상태에서 학습을 진행하지만 본 레포지토리에서는 alignment learning 기반 학습을 진행하고 preprocessing으로 인해 발생할 수 있는 디스크 용량 문제를 방지하기 위해 data_utils.py로부터 학습 데이터가 feeding됩니다.
3. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
4. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
5. 별도의 pre-processing 과정은 필요하지 않지만, pitch와 energy의 max, min, mean, str 값을 다시 추출해야 한다면 pre-processing을 진행하셔야 합니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/FastSpeech2/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/FastSpeech2/data/dataset`

## Docker build
1. `cd /path/to/the/FastSpeech2`
2. `docker build --tag FastSpeech2:latest .`

## Training
1. `nvidia-docker run -it --name 'FastSpeech2' -v /path/to/FastSpeech2:/home/work/FastSpeech2 --ipc=host --privileged FastSpeech2:latest`
2. `cd /home/work/FastSpeech2`
3. (OPTIONAL) `python preprocess.py ./config/kss/preprocess.yaml`
4. `python train.py -p ./config/kss/preprocess.yaml -m ./config/kss/model.yaml -t ./config/kss/train.yaml`
12. arguments
  * -p : preprocess config path
  * -m : model config path
  * -t : train config path
13. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses
![FastSpeech2-tensorboard-losses](https://user-images.githubusercontent.com/69423543/179240643-d0be3733-c19a-4f33-ae4a-1fa255ddd191.png)

## Tensorboard alignment
![FastSpeech2-tensorboard-alignment](https://user-images.githubusercontent.com/69423543/179240657-8090b2f0-1e16-43c6-9167-7e88141770e3.png)

## Tensorboard mel-spectrograms
![FastSpeech2-tensorboard-mels](https://user-images.githubusercontent.com/69423543/179240889-0d39f2a7-309a-4741-81fd-aa0f91203cc5.png)


## Reference
1. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
2. [One TTS Alignment To Rule Them All](https://arxiv.org/pdf/2108.10447.pdf)
3. [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS)
