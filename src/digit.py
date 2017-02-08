# coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_numbers(filename, test=True):
    """
    从原始图像中找出示数，返回示数框的阈值图像
    :param filename: 原始图像文件路径
    :param test: 是否可视化调参
    :return: 示数框的阈值图像
    """
    # 一些参数
    thr_level = 2  # 阈值为均值的此值倍
    border_thr = 11     # 边缘检测的阈值
    close_open_size = 3     # 闭、开操作核的尺寸

    # 读为黑白
    img = cv2.imread(filename, 0)
    # 基于均值灰度的阈值处理
    high = img.mean()
    cv2.threshold(img, high * thr_level, 1, cv2.THRESH_BINARY, img)
    # 通过闭操作修复阈值处理空穴
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_open_size, close_open_size))
    cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, img)
    if test:
        cv2.imshow("wa", img), cv2.waitKey()
    # 剪裁图像区域
    row, col = img.shape
    for row_index, row_sum in enumerate(img.sum(1)):
        if row_sum > border_thr:
            up = row_index
            break
    for row_index, row_sum in enumerate(img[::-1].sum(1)):
        if row_sum > border_thr:
            down = row - row_index - 1
            break
    for col_index, col_sum in enumerate(img.sum(0)):
        if col_sum > border_thr:
            left = col_index
            break
    for col_index, col_sum in enumerate(img[:, ::-1].sum(0)):
        if col_sum > border_thr:
            right = col - col_index - 1
            break
    cut = img[up:down, left:right]
    if test:
        cv2.imshow("wa", cut), cv2.waitKey()

    return cut


def cut_numbers(img, test=True):
    """
    根据示数的阈值图像分割各个数字
    :param img: 示数框部分的阈值图像
    :param test: 是否可视化调参
    :return: 单独数字图像的列表
    """
    # 检测数字边界
    border_cols = [0]
    col_sum = img.sum(0)
    for col_index in range(len(col_sum) - 1):
        if (col_sum[col_index] != 0 and col_sum[col_index + 1] == 0) or \
                (col_sum[col_index] == 0 and col_sum[col_index + 1] != 0):
            border_cols.append(col_index)
    border_cols.append(len(col_sum) - 1)
    # 边界数一定为偶
    assert len(border_cols) % 2 == 0

    # 剪裁单个数字
    numbers = []
    for left, right in zip(border_cols[::2], border_cols[1::2]):
        number = img[:, left:right]
        numbers.append(number)
        if test:
            cv2.imshow("wa", number), cv2.waitKey()

    return numbers


def read_number(numbers, test=True):
    """
    根据图像列表得到示数
    :param numbers: 分割后的图像列表
    :param test: 是否可视化调参
    :return: 示数（浮点）
    """
    if test:
        for index, img in zip(range(4), numbers):
            plt.subplot(1, 4, index + 1)
            plt.imshow(img, cmap="gray")
        plt.show()

    # 寻找小数点
    end_cols = []
    for number_index, number in enumerate(numbers):
        row, col = number.shape
        for col_index, col_sum in enumerate(number[:, ::-1].sum(0)):
            if col_sum > row / 5:
                numbers[number_index] = number[:, :col - col_index]
                end_cols.append(col_index)
                break

    # 识别每个数字
    number_reads = []
    for number in numbers:
        # 若图像过窄则为1
        row, col = number.shape
        if col < row / 3:
            number_reads.append(1)
            continue

        # 在图像边界添加黑色边框
        img = np.zeros((row + 2, col + 2))
        img[1:1 + row, 1:1 + col] = number
        row, col = img.shape

        # 中心垂直方向
        mid_lines = 0
        for row_index in range(row - 1):
            if img[row_index, col / 2] == 0 and img[row_index + 1, col / 2] != 0:
                mid_lines += 1
        # 上方水平方向
        upper_left, upper_right = False, False
        for col_index in range(col - 1):
            if img[row / 4, col_index] == 0 and img[row / 4, col_index + 1] != 0:
                if col_index < col / 2:
                    upper_left = True
                else:
                    upper_right = True
        # 下方水平方向
        lower_left, lower_right = False, False
        for col_index in range(col - 1):
            if img[row * 3 / 4, col_index] == 0 and img[row * 3 / 4, col_index + 1] != 0:
                if col_index < col / 2:
                    lower_left = True
                else:
                    lower_right = True

        # 根据扫描结果判断数字
        if mid_lines == 1:
            if upper_left:
                number_read = 4
            else:
                number_read = 7
        elif mid_lines == 2:
            number_read = 0
        else:
            if upper_left and upper_right:
                if lower_left:
                    number_read = 8
                else:
                    number_read = 9
            elif upper_left:
                if lower_left:
                    number_read = 6
                else:
                    number_read = 5
            else:
                if lower_left:
                    number_read = 2
                else:
                    number_read = 3
        number_reads.append(number_read)

    if test:
        for index, number, img in zip(range(4), number_reads, numbers):
            plt.subplot(1, 4, index + 1)
            plt.imshow(img, cmap="gray")
            plt.title(str(number))
        plt.show()

    # 计算结果
    result = 0.0
    for number in number_reads:
        result *= 10
        result += number
    point_pos = end_cols.index(max(end_cols))
    result *= 10 ** (point_pos - len(numbers) + 1)

    return result


def main():
    filenames = ["../pic/dig1.PNG", "../pic/dig2.PNG", "../pic/dig3.PNG", "../pic/dig4.PNG", "../pic/dig5.PNG"]

    for filename in filenames:
        img = find_numbers(filename, test=False)
        number_segs = cut_numbers(img, test=False)
        result = read_number(number_segs, test=True)
        print u"示数为:", result


if __name__ == '__main__':
    main()
