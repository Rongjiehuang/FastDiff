# FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis

![alt-text-1](asserts/Demo.gif "denoising")

### Rongjie Huang, Max W. Y. Lam, Jun Wang, Dan Su, Dong Yu, Yi Ren, Zhou Zhao

PyTorch Implementation of (IJCAI'22) [FastDiff: a conditional diffusion probabilistic model capable of generating high fidelity speech efficiently.](https://arxiv.org/abs/2204.09934).

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2204.09934)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/FastDiff?style=social)](https://github.com/Rongjiehuang/Multi-Singer)
<a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>

We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://fastdiff.github.io/) for audio samples.

# News
- April.22, 2021: We submitted **FastDiff** to Arxiv. The expected release time of the full version codes is at the IJCAI-2022 conference (before July. 2022). Please star us and stay tuned! 

# Quick Started
We provide an example of how you can generate high-fidelity samples using FastDiff.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below intructions.

## Pretrained Model

You can also use pretrained models we provide.
Download pretrained models
Details of each folder are as in follows:

| Dataset  | Config                                         | Fine-Tuned | Model          | 
|----------|------------------------------------------------|------------|----------------|
| LJSpeech | modules/FastDiff/config/FastDiff.yaml          | No         | Coming  Soon   |
| VCTK     | modules/FastDiff/config/FastDiff_libritts.yaml | No         | Coming  Soon   |
| LibriTTS | modules/FastDiff/config/FastDiff_vctk.yaml     |  No        | Coming  Soon   |

## Dependencies
See requirements in `requirement.txt`:
- [pytorch](https://github.com/pytorch/pytorch)==1.8.1
- [librosa](https://github.com/librosa/librosa)==0.7.1
- [tacotron2](https://github.com/NVIDIA/tacotron2) (source included in this repo)

## Inference from wav file
1. Make `test_files` directory and copy wav files into the directory.
2. Run the following command.
```
python tasks/run.py --config path/to/config  --exp_name [your experiment name] --infer --hparams='test_input_dir=wavs,use_wav=true'
```

Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.


## Inference for end-to-end speech synthesis
1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
```
python tasks/run.py --config  --exp_name [your experiment name] --infer --hparams='test_input_dir=mels,use_wav=false'
```
Generated wav files are saved in `generated_files_from_mel` by default.<br>
You can change the path by adding `--output_dir` option.


# Train your own model

### I. Data Preparation and Configuraion ## 
(Work in Progress)


### II. Training the Refinement Network
```
python tasks/run.py --config path/to/config  --exp_name [your experiment name] --reset
```

### III. Training the Noise Predictor Network (Optional)
Given a well-trained score network, we start training the scheduling network (noise predictor in paper), following [BDDM](https://github.com/tencent-ailab/bddm).

### IV. Noise Scheduling (Optional)
- Given a well-trained FastDiff (Refinement and Scheduling Network), we can then perform noise scheduling to select the best noise schedule suitable to your needs (trade-off between performance v.s. step number), following [BDDM](https://github.com/tencent-ailab/bddm). 
- Put the derived **noise_schedule** in config file

Or you can use our pre-derived **noise_schedule** in config file

## III. Inference

```
python tasks/run.py --config path/to/config  --exp_name [your experiment name] --infer
```


## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[Tacotron2](https://github.com/NVIDIA/tacotron2), and
[DiffWave-Vocoder](https://github.com/philsyn/DiffWave-Vocoder)
as described in our code.

## Citations ##

```

```

## Disclaimer ##
This is not an officially supported Tencent product.
