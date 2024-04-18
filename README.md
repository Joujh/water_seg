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
git clone https://github.com/Joujh/water_seg.git
```

```bash
cd water_seg
```

```bash
pip install -v -e .
# Alternatively: python setup.py develop
```


# **准备工作**

1.在“”下载预训练权重simmim_pretrain__swin_base__img192_window6__800ep.pth放在checkpoints文件夹下

2.将数据集放在data文件夹下

3.custom-tools/water_config.py中修改数据集路径

第128行和155行data_root: data/waterdataset/train, data/waterdataset/train为训练集所在文件夹的路径, 可以根据实际数据集路径修改。该文件夹下只能存放图片。

第182行data_root: data/waterdataset/val, data/waterdataset/val为验证集所在文件夹的路径, 可以根据实际数据集路径修改。该文件夹下只能存放图片。

第205行data_root: data/waterdataset/test, data/waterdataset/test为训练集所在文件夹的路径, 可以根据实际数据集路径修改。该文件夹下只能存放图片。


此时你的文件目录应该类似如下结构:

```plaintext
.
├── checkpoints
|   └── simmim_pretrain__swin_base__img192_window6__800ep.pth
├── custom
├── custom-tools
│   ├── dist_test.sh
│   ├── dist_train.sh
│   ├── 0255_2_01.py
│   ├── pre_process.py
│   ├── swin2mmseg.py
│   ├── water_cfg.py
│   ├── test.py
│   └── train.py
├── data
│   └── waterdataset
│       ├── train
│       |    └── img
│       |    └── lbl
│       ├── val
│       |    └── img
│       |    └── lbl
│       └── test
│            └── img
│            └── lbl
├── mmseg
├── requirements
├── work_dirs
├── readme.md
├── requirements.txt
├── setup.cfg
└── setup.py
```

## 数据集预处理

所有标签图片为0_1像素组成（包括训练集、验证集、测试集标签），如果是0_255像素组成的，使用以下命令格式完成像素的转换：

python ./custom-tools/0255_2_01.py '标签图片的路径'，例如对训练集标签像素转换


```bash
python ./custom-tools/0255_2_01.py '/data/waterdataset/train/lbl'
```

对数据集图片（包括训练集、验证集、测试集）的预处理，使用以下命令格式进行预处理：

python ./custom-tools/pre_process.py '图片的路径' '对应的标签图片路径'，例如对训练集图片和标签进行预处理


```bash
python ./custom-tools/pre_process.py '/data/waterdataset/train/img' '/data/waterdataset/train/lbl'
```

## 训练

训练命令格式为:
python ./custom-tools/train.py '配置文件'，例如

```bash
python custom-tools/train.py 'custom-tools/water_cfg.py'
```

```bash
python custom-tools/train.py checkpoints/night/cfg.py checkpoints/night/night.pth --eval mIoU --aug-test
```

## 测试
测试命令格式为:
python ./custom-tools/train.py '配置文件' '权重文件' --eval mIoU，例如

```bash
python custom-tools/train.py 'custom-tools/water_cfg.py'
```

```bash
python custom-tools/train.py 'custom-tools/water_cfg.py' 'work_dirs/water_cfg/best_mIoU_iter.pth' --eval mIoU
```
实际权重保存在work_dirs/water_cfg下，权重名按实际而定


