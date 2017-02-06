# coding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_order(pts):
    """
    输入四个点，返回矩形边界坐标
    :param pts: 以(x, y)为元素的列表
    :return: x1(小), x2(大), y1(小), y2(大)
    """
    if abs(pts[0][0] - pts[1][0]) <= 10:  # 当前两点x轴坐标相同时，y轴坐标必定不同
        if pts[0][1] < pts[1][1]:
            y1 = pts[0][1]
            y2 = pts[1][1]
        else:
            y1 = pts[1][1]
            y2 = pts[0][1]

        if pts[0][0] < pts[2][0]:
            x1 = pts[0][0]
            x2 = pts[2][0]
        else:
            x1 = pts[2][0]
            x2 = pts[0][0]
    elif abs(pts[0][0] - pts[2][0]) <= 10:
        if pts[0][1] < pts[2][1]:
            y1 = pts[0][1]
            y2 = pts[2][1]
        else:
            y1 = pts[2][1]
            y2 = pts[0][1]

        if pts[0][0] < pts[1][0]:
            x1 = pts[0][0]
            x2 = pts[1][0]
        else:
            x1 = pts[1][0]
            x2 = pts[0][0]
    else:
        if pts[0][1] < pts[3][1]:
            y1 = pts[0][1]
            y2 = pts[3][1]
        else:
            y1 = pts[3][1]
            y2 = pts[0][1]

        if pts[0][0] < pts[1][0]:
            x1 = pts[0][0]
            x2 = pts[1][0]
        else:
            x1 = pts[1][0]
            x2 = pts[0][0]

    return x1, x2, y1, y2


def cut_leds(img_filename, template_filename, test=True):
    """
    读取图像、十字标记模版，通过模版匹配定位表盘，并剪裁待处理区域
    :param img_filename: 图像文件名
    :param template_filename: 模版文件名
    :param test: 测试模式
    :return: 剪裁后的表盘图像
    """
    # 读取图像
    img = cv2.imread(img_filename)
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 读取十字标记模版
    template = cv2.imread(template_filename, 0)
    w, h = template.shape[::-1]

    # 模版匹配
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    # 使用阈值处理过滤结果
    threshold = 0.9
    loc = np.where(res >= threshold)
    # 记录十字的四个顶点
    pts1 = []
    for pt in zip(*loc[::-1]):
        # 计算十字中心
        cross_x, cross_y = pt
        cross_x += w / 2
        cross_y += h / 2
        # 判断是否重复选择此点
        to_add = True
        for x, y in pts1:
            if abs(x - cross_x) <= 10 and abs(y - cross_y) <= 10:
                to_add = False
        if not to_add:
            continue
        # 记录此点
        pts1.append([cross_x, cross_y])
        # 测试时画图可视化
        if test:
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    # 测试时画图可视化
    if test:
        cv2.imshow("wa", img), cv2.waitKey(), cv2.destroyAllWindows()

    # 确定透视变换对应的点
    x1, x2, y1, y2 = find_order(pts1)
    pts1 = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    pts2 = np.float32([[0, 0], [360, 0], [360, 140], [0, 140]])
    # 透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (360, 140))

    # 测试时画图可视化
    if test:
        cv2.imshow("wa", img), cv2.waitKey(), cv2.destroyAllWindows()

    return img


def find_led(img, test=True):
    gauss_size = 9  # 高斯模糊核大小
    thr_level = 0.95     # 阈值为最大灰度此值

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊去除不必要的细节
    cv2.GaussianBlur(gray, (gauss_size, gauss_size), 0, gray)
    # 阈值处理
    high = img.max()
    cv2.threshold(gray, high * thr_level, 255, cv2.THRESH_BINARY, gray)
    if test:
        cv2.imshow("wa", gray), cv2.waitKey()
    # 寻找亮度中心（opencv里的x轴，numpy里的y轴）
    x_mean = np.nonzero(gray)[1].mean()
    if x_mean < img.shape[1] / 2:
        return u"绿灯"
    else:
        return u"红灯"


def main():
    files = ["../pic/led1.png", "../pic/led2.png"]
    template = "../pic/cross.png"
    for filename in files:
        img = cut_leds(filename, template, test=True)
        led = find_led(img, test=True)
        print u"亮灯为:", led


if __name__ == '__main__':
    main()