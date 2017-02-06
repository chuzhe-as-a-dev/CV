# coding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


def cut_panel(img_filename, template_filename, test=True):
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
    threshold = 0.90
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

    # 确定表盘的中心
    ratio = 0.65    # 剪裁比例
    x1 = pts1[0][0]
    x2 = pts1[1][0]
    y1 = pts1[0][1]
    y2 = pts1[3][1]
    delta_x = x2 - x1
    delta_y = y2 - y1
    x1 = x1 + (1 - ratio) / 2 * delta_x
    x2 = x2 - (1 - ratio) / 2 * delta_x
    y1 = y1 + (1 - ratio) / 2 * delta_y
    y2 = y2 - (1 - ratio) / 2 * delta_y
    # 确定透视变换对应的点
    pts1 = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    pts2 = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
    # 透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (200, 200))

    # 测试时画图可视化
    if test:
        cv2.imshow("wa", img), cv2.waitKey(), cv2.destroyAllWindows()

    return img


def find_lines(img, test=True):
    """
    从仪表图像，将一定角度范围的线条找到并返回其相关参数
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
    hough_thrs = np.arange(50, 100, 10)    # Hough线检测阈值调参范围
    hough_thr = 60  # Hough线检测阈值
    angle_epsilon = 10  # 线的角度范围允许误差

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

    return 0.0


def main():
    files = ["../pic/uncut1.png", "../pic/uncut2.png", "../pic/uncut3.png", "../pic/uncut4.png", "../pic/uncut5.png"]
    template = "../pic/cross.png"

    for filename in files:
        img = cut_panel(filename, template, test=True)
        lines = find_lines(img, test=True)
        angle = find_angle(lines)
        print u"示数为:", angle


if __name__ == '__main__':
    main()
