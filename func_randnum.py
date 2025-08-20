import numpy as np


def func_randnum(col, row):
    """创建值为1或-1的矩阵"""
    mask = np.random.rand(col, row)
    randmat = np.zeros((col, row), dtype=np.int8)

    randmat[mask >= 0.5] = 1
    randmat[mask < 0.5] = -1

    return randmat