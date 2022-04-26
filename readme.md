# FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis

<div align=center> <img src="assets/Demo.gif" alt="drawing" style="width:400px; "/> </div>


#### Rongjie Huang, Max W. Y. Lam, Jun Wang, Dan Su, Dong Yu, Yi Ren, Zhou Zhao

PyTorch Implementation of [FastDiff (IJCAI'22)](https://arxiv.org/abs/2204.09934): a conditional diffusion probabilistic model capable of generating high fidelity speech efficiently.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2204.09934)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/FastDiff?style=social)](https://github.com/Rongjiehuang/FastDiff)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Rongjiehuang/FastDiff)
![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)
![Colab](https://colab.research.google.com/assets/colab-badge.svg)

We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://fastdiff.github.io/) for audio samples.

# News
- April.22, 2021: **FastDiff** accepted by IJCAI 2022. The expected release time of the full version codes (Pre-trained models, Colab, Hugging Face) is at the IJCAI-2022 conference (before July. 2022). Please star us and stay tuned! 

# Status
- [x] Release init codes
- [ ] Release pre-trained models
- [ ] Colab and hugging Face
- [ ] Release baseline codes (including HIFI-GAN, Parallel WaveGAN, Diffwave, WaveGrad, etc)
- [ ] programmatic inference API
- [ ] PyPI package

# Quick Started
We provide an example of how you can generate high-fidelity samples using FastDiff.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below intructions.

## Pretrained Model

You can also use pretrained models we provide.
Details of each folder are as in follows:

| Dataset  | Config                                         | Model            | 
|----------|------------------------------------------------|------------------|
| LJSpeech | `modules/FastDiff/config/FastDiff.yaml`          | [Coming  Soon]() |
| VCTK     | `modules/FastDiff/config/FastDiff_libritts.yaml` | [Coming  Soon]() |
| LibriTTS | `modules/FastDiff/config/FastDiff_vctk.yaml`     | [Coming  Soon]() |

Put the checkpoints to `checkpoints/your_experiment_name/model_ckpt_steps_*.ckpt`

## Dependencies
See requirements in `requirement.txt`:
- [pytorch](https://github.com/pytorch/pytorch)
- [librosa](https://github.com/librosa/librosa)
- [tacotron2](https://github.com/NVIDIA/tacotron2) (source included in this repo)

## Multi-GPU
By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.

## Inference from wav file
1. Make `wavs` directory and copy wav files into the directory.
2. Run the following command.
```bash
python tasks/run.py --config path/to/config  --exp_name your_experiment_name --infer --hparams='test_input_dir=wavs'
```

Generated wav files are saved in `checkpoints/your_experiment_name/` by default.<br>

## Inference for end-to-end speech synthesis
1. Make `mels` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
```bash
python tasks/run.py --config  --exp_name your_experiment_name --infer --hparams='test_mel_dir=mels,use_wav=False'
```
Generated wav files are saved in `checkpoints/your_experiment_name/` by default.<br>


# Train your own model

### Data Preparation and Configuraion ##
1. Set `raw_data_dir`, `processed_data_dir`, `binary_data_dir` in the config file
2. Download dataset to `raw_data_dir`
3. Preprocess Dataset 
```bash
# Preprocess step: text and unify the file structure.
python data_gen/tts/runs/preprocess.py --config path/to/config
# Align step: MFA alignment (Optional).
python data_gen/tts/runs/train_mfa_align.py --config path/to/config
# Binarization step: Binarize data for fast IO. You only need to rerun this line when running different task if you have `preprocess`ed and `align`ed the dataset before.
python data_gen/tts/runs/binarize.py --config path/to/config
```

### Training the Refinement Network
```bash
python tasks/run.py --config path/to/config  --exp_name your_experiment_name --reset
```

### Training the Noise Predictor Network (Optional)
Given a well-trained refinement network, we start training the scheduling network (noise predictor in paper), following [BDDM](https://github.com/tencent-ailab/bddm).

### Noise Scheduling (Optional)
- Given a well-trained FastDiff (refinement and scheduling network), we can then perform noise scheduling to select the best noise schedule suitable to your needs (trade-off between performance v.s. step number), following [BDDM](https://github.com/tencent-ailab/bddm). 
- Put the derived `noise_schedule` in config file

Or you can use our pre-derived `noise_schedule` in config file

### Inference

```bash
python tasks/run.py --config path/to/config  --exp_name your_experiment_name --infer
```


## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[Tacotron2](https://github.com/NVIDIA/tacotron2), and
[DiffWave-Vocoder](https://github.com/philsyn/DiffWave-Vocoder)
as described in our code.

## Citations ##
If you find this code useful in your research, please consider citing:
```
@article{huang2022fastdiff,
  title={FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis},
  author={Huang, Rongjie and Lam, Max WY and Wang, Jun and Su, Dan and Yu, Dong and Ren, Yi and Zhao, Zhou},
  journal={arXiv preprint arXiv:2204.09934},
  year={2022}
}
```

## Disclaimer ##
This is not an officially supported Tencent product.

