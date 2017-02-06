# coding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


FILENAME = "../pic/cut.png"


def find_lines(filename, test=True):
    """
    读取图像，将一定角度范围的线条找到并返回其相关参数
    :param filename: 图像地址
    :param test: 是否可视化调参
    :return: 由线(rho, theta)为元素的列表
    """

    # 一些参数
    power = 1.5     # 幂律变换的指数
    gauss_size = 9  # 高斯模糊核大小
    canny_thr_lows = np.linspace(70, 120, 6)    # Canny边缘检测低阈值调参范围
    canny_thr_low = 100     # Canny边缘检测低阈值
    thr_high_over_lows = np.linspace(2.5, 3.5, 5)   # Canny高低阈值比值调参范围
    thr_high_over_low = 3.25    # Canny高低阈值比值
    hough_thrs = np.arange(20, 31, 1)    # Hough线检测阈值调参范围
    hough_thr = 28  # Hough线检测阈值
    angle_epsilon = 10  # 线的角度范围允许误差

    # 读取图像
    img = cv2.imread(filename)  # 读为黑白

    # 测试时将调参过程可视化
    if test:
        # 针对Hough线检测阈值
        for index, hough_thr in enumerate(hough_thrs):
            # 转换为灰度图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 高斯模糊去除不必要的细节
            cv2.GaussianBlur(gray, (gauss_size, gauss_size), 0, gray)
            # 拉伸动态范围
            gray = gray - gray.min()
            gray = (gray * 255.0 / gray.max()).astype(np.uint8)
            # 应用幂律变换，压缩低灰度范围
            c = 255.0 / gray.max() ** power
            gray = (c * gray.astype(np.float) ** power).astype(np.uint8)  # 幂律变换
            # Canny边缘检测
            cv2.Canny(gray, canny_thr_low, canny_thr_low * thr_high_over_low, gray)   # Canny边缘检测
            # Hough线检测，并新建图像draw将线画上去
            lines = cv2.HoughLines(gray, 1, np.pi / 180, int(hough_thr))
            draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    # 只画出特定角度的线条
                    if theta <= np.pi * angle_epsilon / 180 or theta >= np.pi * (90 - angle_epsilon) / 180:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * -b)
                        y1 = int(y0 + 1000 * a)
                        x2 = int(x0 - 1000 * -b)
                        y2 = int(y0 - 1000 * a)

                        cv2.line(draw, (x1, y1), (x2, y2), (255, 0, 0), 1)

            plt.subplot(1, len(hough_thrs), index + 1)
            plt.imshow(draw)
            plt.title("{}".format(hough_thr))
            plt.xticks([]), plt.yticks([])

        plt.show()

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊去除不必要的细节
    cv2.GaussianBlur(gray, (gauss_size, gauss_size), 0, gray)
    # 拉伸动态范围
    gray = gray - gray.min()
    gray = (gray * 255.0 / gray.max()).astype(np.uint8)
    # 应用幂律变换，压缩低灰度范围
    c = 255.0 / gray.max() ** power
    gray = (c * gray.astype(np.float) ** power).astype(np.uint8)  # 幂律变换
    # Canny边缘检测
    cv2.Canny(gray, canny_thr_low, canny_thr_low * thr_high_over_low, gray)  # Canny边缘检测
    # Hough线检测，将角度符合要求的线条参数添加到result列表中
    lines = cv2.HoughLines(gray, 1, np.pi / 180, int(hough_thr))
    result = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if theta <= np.pi * angle_epsilon / 180 or theta >= np.pi * (90 - angle_epsilon) / 180:
                result.append((rho, theta))

    return result


def find_angle(lines):
    """
    从一堆线中确定指针所在线的角度
    :param lines: 由线(rho, theta)为元素的列表
    :return: 指针的角度
    """

    pass


def main():
    find_lines(FILENAME, test=True)


if __name__ == '__main__':
    main()
