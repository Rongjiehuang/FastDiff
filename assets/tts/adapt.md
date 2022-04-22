# 语音克隆 [TODO]

## 训练基础模型
按照 [语音合成文档](tts.md) 训练任一基础模型，如`checkpoints/1004_vctk_1`，以下以FastSpeech和数据集vctk为例。

## 新建配置文件
在用户配置文件夹下新建配置文件`configs_usr/vctk/fs2_adapt.yaml`：
```yaml
base_config:
  - configs/tts/vctk/fs2.yaml
  - configs/tts/fs2_adapt.yaml
adapt_train_items: []
adapt_valid_items: []
adapt_test_items: []
adapt_data_dir: data/processed/vctk_adapt_1004 # 克隆训练用的数据保存的位置
# 在语音克隆中，若use_spk_id=True，为了得到稳定的韵律，时长预测器通常直接使用老的speaker embedding。spk_id_dur指定了时长预测器使用的老的spk id。可以使用 python scripts/view_spk_map.py --config $CONFIG 来查看某一个数据集的spk_map，$CONFIG不是adapt的config，而是base model的config
spk_id_dur: -1
```

## 制作Adapt数据集
- Adapt数据集分`adapt_train`、`adapt_valid`和`adapt_test`，其中`adapt_train`用于adapt阶段训练，`adapt_valid`用于判断训练停止，`adapt_test`用于infer时提取音高、时长等信息。
- 以下以`configs_usr/vctk/fs2_adapt.yaml`配置文件为例。
```bash
export CONFIG_ADAPT=configs_usr/ljspeech/fs2_adapt.yaml
```
- 用以下其中一个方式指定adapt所用数据集：
    - 用config中指定的`raw_data_dir`里的数据作为adapt所需数据集：修改配置文件中的`adapt_*_items`，用`p225_*`类似的形式来指定需要选取的adapt数据的item_name的pattern，*表示通配符。
    - 其他数据数据作为adapt所需数据集：将`adapt_train`、`adapt_valid`和`adapt_test`分别放在`${adapt_data_dir}/train`、`${adapt_data_dir}/valid`和`${adapt_data_dir}/test`下，每条样本需要2个文件：`xxx.wav`波形文件和`xxx.text`中文歌词文件（单句一行）。
- 运行
```bash
python data_gen/tts/gen_adapt_data.py --config $CONFIG_ADAPT
```

## Adapt训练 
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_ADAPT --reset --exp_name 1004_fs2_vctk_adapt_1 --hparams="load_ckpt=checkpoints/1004_vctk_1"
```

### Adapt推理
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_ADAPT --reset --exp_name 1004_fs2_vctk_adapt_1 --infer --hparams="gen_dir_name=adapt"
```