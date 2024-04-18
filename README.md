# **环境配置**

创建环境

```bash
conda create -n dtp python==3.10
```

激活环境

```bash
conda activate dtp
```

安装pytorch

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# Alternatively: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```
安装相关环境

```bash
pip install tensorboard
```

```bash
pip install -U openmim
```

```bash
mim install mmcv-full
```

```

克隆仓库

```bash
git clone https://github.com/DaoCaoRenH/samwater.git
```

```bash
cd water_seg
```

```bash
pip install -v -e .
# Alternatively: python setup.py develop
```

Your directory structure should resemble:

```plaintext
.
├── checkpoints
│   ├── night
│   ├── night+day
|   └── simmim_pretrain__swin_base__img192_window6__800ep.pth
├── custom
├── custom-tools
│   ├── dist_test.sh
│   ├── dist_train.sh
│   ├── test.py
│   └── train.py
├── data
│   ├── cityscapes
│   │   ├── gtFine
│   │   └── leftImg8bit
│   └── nightcity-fine
│       ├── train
│       └── val
├── mmseg
├── readme.md
├── requirements.txt
├── setup.cfg
└── setup.py
```

## Testing

Execute tests using:

```bash
python custom-tools/test.py checkpoints/night/cfg.py checkpoints/night/night.pth --eval mIoU --aug-test
```

## Training
1. Download pre-training weight from [Google Drive](https://drive.google.com/file/d/15zENvGjHlM71uKQ3d2FbljWPubtrPtjl/view).
2. Convert it to MMSeg format using:
    ```shell
    python custom-tools/swin2mmseg.py </path/to/pretrain> checkpoints/simmim_pretrain__swin_base__img192_window6__800ep.pth
    ```
3. Start training with:
    ```shell
    python custom-tools/train.py </path/to/your/config>
    # </path/to/your/config>：our config: checkpoints/night/cfg.py or checkpoints/night+day/cfg.py
    ```

## Results

The table below summarizes our findings:

| logs                                            | train dataset                  | validation dataset | mIoU |
|-------------------------------------------------|--------------------------------|--------------------|------|
| checkpoints/night/eval_multi_scale_20230801_162237.json | nightcity-fine                 | nightcity-fine     | 64.2 |
| checkpoints/night+day/eval_multi_scale_20230809_170141.json | nightcity-fine + cityscapes    | nightcity-fine     | 64.9 |

# Acknowledgements
This dataset is refined based on the dataset of [NightCity](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html) by Xin Tan *et al.* and [NightLab](https://github.com/xdeng7/NightLab) by Xueqing Deng *et al.*.

This project is based on the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation.git).

Pretraining checkpoint comes from the [SimMIM](https://github.com/microsoft/SimMIM).

The annotation process was completed using [LabelMe](https://github.com/wkentaro/labelme.git).

# **准备工作**

在configs/data-sam-vit-t.yaml中修改测试数据集路径

第34行root_path_1: data/val/img, data/val/img为测试图片所在文件夹的路径, 可以根据实际数据集路径修改。该文件夹下只能存放图片。

第35行root_path_2: data/val/label, data/val/label为测试图片对应标签所在文件夹的路径, 可以根据实际数据集路径修改。该文件夹下只能存放图片。

出现报错时，大概率是因为填写的测试图片路径存在问题，或者测试图片文件夹下存在其他文件

测试命令：在当前目录下执行 --config后是配置文件路径 --model后是权重文件路径

```bash
python test.py --config configs/data-sam-vit-t.yaml --model model.pth
```



# **分别在Ubuntu和Windows环境下测试**

1.测试环境：Ubuntu20.04

![testubuntu.png](./testubuntu.png)

2.测试环境：Windows10

![testwindows.png](./testwindows.png)

**指标的值不同是因为使用的测试数据集不同**

# **可视化**

在当前环境安装依赖库

```bash
pip install gradio==3.45.2
```

在当前项目路径下运行命令

```bash
python app.py
```

在app.py的第60行,model_path = "model.pth"可以修改权重文件路径

打开http://0.0.0.0:7579 ,可视化界面如下图

![appsample.png](./appsample.png)
