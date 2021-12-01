# author： mixiaoxin@whu.edu.cn

import numpy as np
import matplotlib.pyplot as plt
import os

# 画loss曲线
def drawLossCurve(epoch_ind, loss_val):
    assert len(epoch_ind) == len(loss_val)
    # 设置画布大小
    #plt.figure(figsize=(2000, 1200))

    # 标题
    plt.title("BBAV Loss Curve")

    # 数据
    plt.plot(epoch_ind, loss_val, label='training loss', linewidth=3, color='r', marker='o',
             markerfacecolor='blue', markersize=5)

    # 横坐标描述
    plt.xlabel('epoch')

    # 纵坐标描述
    plt.ylabel('loss')

    # 设置数字标签, 数字标签保留3位小数
    for a, b in zip(epoch_ind, loss_val):
        if a % 8 == 0 or a == len(epoch_ind) or a == 1:
            plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=10)

    plt.legend()
    plt.show()
    plt.pause(3)

# 从路径中读入每个epoch的loss值
def readLossValues(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    loss_lists = [float(line.strip()) for line in lines]
    return loss_lists

if __name__ == '__main__':
    loss_filename = "./../weights_dota/train_loss.txt"
    loss_list = readLossValues(loss_filename)
    epoch_ids = list(range(1, len(loss_list)+1))

    print('epoch id: ', len(epoch_ids))
    print('epoch loss: ', len(loss_list))
    drawLossCurve(epoch_ids, loss_list)

