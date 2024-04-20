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

1.在[Google Drive](https://drive.google.com/drive/folders/1ftbH58jkMa9M8TA1su2WVh9ps-dN8C6l?usp=sharing)下载sewim-transformer权重simmim_pretrain__swin_base__img192_window6__800ep.pth放在checkpoints（没有就新建一个）文件夹下,将预训练权重day.pth放在custom-tools下

2.将数据集放在data（没有就新建一个）文件夹下

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
│   ├── day.pth
│   ├── app.py
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

所有标签图片为0_1像素组成（包括训练集、验证集、测试集标签），如果是0_255像素组成的，使用以下命令格式依次对训练集、验证集、测试集的标签图片完成像素的转换：

python ./custom-tools/0255_2_01.py '标签图片的路径'，例如对训练集标签像素转换


```bash
python ./custom-tools/0255_2_01.py data/waterdataset/train/lbl
```

对验证集、测试集的预处理，使用以下命令格式依次对验证集、测试集进行预处理：

python ./custom-tools/pre_process.py '图片的路径' '对应的标签图片路径'，例如对验证集图片和标签进行预处理


```bash
python ./custom-tools/pre_process.py data/waterdataset/val/img data/waterdataset/val/lbl
```

## 训练

！！！注意，如果需要保存多次训练结果，需要自己再新建一个不同名的配置文件（以区分water_cfg.py）,训练日志以及权重保存在work_dirs下的与训练配置文件同名的文件夹下。

训练命令格式为:
python ./custom-tools/train.py '配置文件' --load-from '预训练权重'，例如

```bash
python ./custom-tools/train.py custom-tools/water_cfg.py --load-from custom-tools/day.pth
```


## 测试
测试命令格式为:
python ./custom-tools/test.py '配置文件' '权重文件' --eval mIoU，例如


```bash
python ./custom-tools/test.py custom-tools/water_cfg.py work_dirs/water_cfg/best_mIoU_iter.pth --eval mIoU
```
实际权重保存在work_dirs/water_cfg下，权重名按实际而定


# **可视化**

在当前环境安装依赖库

```bash
pip install gradio==3.45.2
```

在当前项目路径下运行命令

```bash
python custom-tools/app.py
```

在app.py的第59行,model_path = "day.pth"可以修改权重文件路径
在app.py的第60行,cfg_path = 'water_cfg.py'可以修改模型配置文件路径

打开http://0.0.0.0:7579 ,可视化界面如下图

![image]()

