# TTS 语音合成

## 安装强制对齐工具
[查看安装步骤](install_align.md)
## 下载对应的数据集
[查看数据集下载文档](prepare_datasets.md)

## 选择配置文件
export配置文件为环境变量，例如：
```bash
export CONFIG=egs/datasets/audio/lj/fs2.yaml 
```

可选的配置文件：

```yaml
- LJSpeech
  - egs/datasets/audio/lj/fs2.yaml # FastSpeech 2
  - egs/datasets/audio/lj/fs2_adv.yaml # FastSpeech 2 + Adversarial
  - egs/datasets/audio/lj/fs2_adv_ms.yaml # FastSpeech 2 + multi-scale Adversarial
  - egs/datasets/audio/lj/fs2s.yaml # FastSpeech 2s
  - egs/datasets/audio/lj/transformer_tts.yaml # Transformer TTS
  - egs/datasets/audio/lj/pwg.yaml  # Parallel WaveGAN
  - ...
- VCTK
  - egs/datasets/audio/vctk/fs2.yaml
  - egs/datasets/audio/vctk/pwg.yaml
  - ...
- libritts
  - egs/datasets/audio/libritts/fs2.yaml
  - egs/datasets/audio/libritts/pwg.yaml
  - ...
- bilizzard
  - ...
- thchs30
  - ...
- didispeech 
  - ...
- aishell3
  - ...
```
## 对齐数据集并制作二进制数据文件
- MFA对齐（推荐）
```bash
python data_gen/tts/bin/pre_align.py --config $CONFIG
python data_gen/tts/bin/train_mfa_align.py --config $CONFIG
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config $CONFIG
```
> 对于TransformerTTS，可以跳过`train_mfa_align`。

## 训练
选取对应模型的config作为$CONFIG，例如`FastSpeech 2`可选取`egs/datasets/audio/lj/fs2.yaml`: `export CONFIG=egs/datasets/audio/lj/fs2.yaml`

### TransformerTTS
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG --exp_name 1004_transtts_1 --reset
```

### FastSpeech 2
Implementation of "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"

#### Train
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG --exp_name 1004_fs2_1 --reset
```
#### Infer (GPU)
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG --exp_name 1004_fs2_1 --infer
```
#### Infer (CPU)
```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python tasks/run.py --config $CONFIG --exp_name 1004_fs2_1 --infer
```

### FastSpeech 2s
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG --exp_name 1004_fs2s_1 --reset
```

### FastSpeech 3 (FastSpeech 2 + Multi-Scale Adversarial)
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG --exp_name 1004_fs3_1 --reset
```
