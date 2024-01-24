# Frigate: Frugal Spatio-temporal Forecasting on Road Networks
 复现工作参考文章：[**Frigate: Frugal Spatio-temporal Forecasting on Road Networks**](https://doi.org/10.1145/3580305.3599357)

## 依赖包
复现工作的配置如下:  

- Python: 3.9.0
- PyTorch: 1.9.0 (CUDA 11.1)
- PyTorch Geometric: 1.7.2
- Numpy: 1.23.3
- Pandas: 1.5.1
- SciPy: 1.9.1
- NetworkX: 2.2.8
- tensorboardX
- tqdm

## 数据库
采用滴滴官方数据，数据库下载链接： [preprocessed dataset](https://drive.google.com/file/d/1l715iYVktwi8WFs_eOAvoVWS2pPzYiDJ/view?usp=share_link)

data文件夹的结构如下:
```bash
Frigate
├── data
│   ├── Beijing
│   ├── Chengdu
│   └── Harbin
├── logs
├── model
│   ├── __init__.py
│   ├── model.py
│   ├── tester.py
│   └── trainer.py
├── outputs
│   ├── models
│   ├── predictions
│   └── tensorboard
├── run.sh
├── run_test.sh
├── test.py
├── train.py
└── utils
    ├── __init__.py
    ├── data_utils.py
    └── test_data_utils.py
```

## 训练
脚本 ```run.sh``` 用来对模型进行训练。可以改变其中的一些命名来进行实验。

```run.sh``` 用一个输入来规定用哪个GPU，如在GPU 0上运行训练代码的命令如下：

```bash
bash run.sh 0
```

## 测试
脚本 ```run_test.sh``` 用来对模型进行测试. 需要输入以下四个信息：
1. ```dataset```
2. ```seen_path```
3. ```run_num```
4. ```model_name```

```run_num``` 和```model_name```用于定位可以从日志中找到的训练模型。注意，模型名称只是模型文件的名称，而不是它的完整路径。测试脚本根据run_num参数自动加载正确的模型。

在GPU 0上运行测试代码的命令如下：

```bash
bash run_test.sh 0
```

脚本将显示MAE矩阵，并将预测结果保存在```outputs/predictions/run_<run_number>/pred_true.npz```中。
在```outputs/predictions```中还提供了一个矩阵计算脚本，该脚本接受该脚本保存的格式的文件并计算矩阵。


