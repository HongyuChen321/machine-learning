# data文件夹中的各个程序的作用

## 1.decompress.py

执行该脚本文件，能将原数据集中的所有rar文件进行批量化的解压缩，并保存至raw文件夹中

## 2.ECG.py

执行该脚本文件，可提取raw文件夹中各个数据集的ECG通道的数据（x2或25），并保存至dataset_raw中

## 3.pre_process.py

执行该脚本文件，可去除最后30个epoch的内容，最后将数据保存至dataset中，这个dataset就是最后用于训练或测试的文件

## 4.test.py

该脚本文件是检查rec文件中各个通道的含义的，返回各个通道的名称以及数据数量
