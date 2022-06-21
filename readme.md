# FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis

<div align=center> <img src="assets/Demo.gif" alt="drawing" style="width:250px; "/> </div>


#### Rongjie Huang, Max W. Y. Lam, Jun Wang, Dan Su, Dong Yu, Yi Ren, Zhou Zhao

PyTorch Implementation of [FastDiff (IJCAI'22)](https://arxiv.org/abs/2204.09934): a conditional diffusion probabilistic model capable of generating high fidelity speech efficiently.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2204.09934)
[![GitHub Stars](https://img.shields.io/github/stars/Rongjiehuang/FastDiff?style=social)](https://github.com/Rongjiehuang/FastDiff)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Rongjiehuang/FastDiff)

We provide our implementation and pretrained models as open source in this repository.

Visit our [demo page](https://fastdiff.github.io/) for audio samples.

## News
- April.22, 2021: **FastDiff** accepted by IJCAI 2022. The expected release time of the full version codes (including pre-trained models, more datasets, and more neural vocoders) is at the IJCAI-2022 conference (before July. 2022). Please star us and stay tuned!
- June.21, 2022: The LJSpeech checkpoint and demo code are provided.


# Quick Started
We provide an example of how you can generate high-fidelity samples using FastDiff.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below intructions.

## Support Datasets and Pretrained Models

You can also use pretrained models we provide.
Details of each folder are as in follows:

| Dataset                                   | Config                                         | Pretrained Model | 
|-------------------------------------------|------------------------------------------------|------------------|
| LJSpeech                                  | `modules/FastDiff/config/FastDiff.yaml`          | [OneDrive](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/Epia7La6O7FHsKPTHZXZpoMBF7PoDcjWeKgC-7jtpVkCOQ?e=BF0nOF)     |
| LibriTTS                                  | `modules/FastDiff/config/FastDiff_libritts.yaml` | [Coming  Soon]() |
| VCTK                                      | `modules/FastDiff/config/FastDiff_vctk.yaml`     | [Coming  Soon]() |

More supported datasets are coming soon.

Put the checkpoints in `checkpoints/your_experiment_name/model_ckpt_steps_*.ckpt`

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
2. Set `N` for reverse sampling, which is a trade off between quality and speed. 
3. Run the following command.
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config path/to/config  --exp_name your_experiment_name --infer --hparams='test_input_dir=wavs,N=$N'
```

Generated wav files are saved in `checkpoints/your_experiment_name/` by default.<br>

## Inference for end-to-end speech synthesis
1. Make `mels` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Set `N` for reverse sampling, which is a trade off between quality and speed. 
3. Run the following command.
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config  --exp_name your_experiment_name --infer --hparams='test_mel_dir=mels,use_wav=False,N=$N'
```
Generated wav files are saved in `checkpoints/your_experiment_name/` by default.<br>

Note: If you find the output wav noisy, it's likely because of the mel-preprocessing mismatch between the acoustic and vocoder models. Please tackle this mismatch or use the provided validated acoustic model,  and now we have: [PortaSpeech](https://huggingface.co/spaces/NATSpeech/PortaSpeech/tree/main)

# Train your own model

### Data Preparation and Configuraion ##
1. Set `raw_data_dir`, `processed_data_dir`, `binary_data_dir` in the config file
2. Download dataset to `raw_data_dir`. Note: the dataset structure needs to follow `egs/datasets/audio/*/pre_align.py`, or you could rewrite `pre_align.py` according to your dataset.
3. Preprocess Dataset 
```bash
# Preprocess step: unify the file structure.
python data_gen/tts/bin/pre_align.py --config path/to/config
# Binarization step: Binarize data for fast IO.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config path/to/config
```

### Training the Refinement Network
```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config path/to/config  --exp_name your_experiment_name --reset
```

### Training the Noise Predictor Network
Coming Soon.

### Noise Scheduling
Coming Soon, and you can use our pre-derived noise schedule in this time.

### Inference

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config path/to/config  --exp_name your_experiment_name --infer
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

