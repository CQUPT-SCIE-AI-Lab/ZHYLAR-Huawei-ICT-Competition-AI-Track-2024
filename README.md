# 华为ICT大赛昇腾AI赛道2023-2024国赛实验（训练部分）

## 赛题

### 题目背景

电路板异常检测。

### 题目要求

代码填空。利用raw data生成图片对应的mask，然后使用mindspore框架训练Unet分割蓝色部分。比赛要求在华为云环境中训练，此仓库提供的代码需要在Windows CPU 环境下运行。

## 开始

1. install mindspore == 2.2.13
2. Run `pip install -r requirements.txt`
3. Download the original data compressed file and extract it to `./raw_data` . Download link: [百度网盘](https://pan.baidu.com/s/1AwxqObh0CP0i9WrTVfA9CA?pwd=0308)
4. run `preprocess_dataset.py`. This code will convert the raw data into images and masks, and store them in `./data`
5. run `train.py`
