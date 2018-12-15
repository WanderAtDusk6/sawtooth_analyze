# test_图像边缘检测(中值滤波)
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2 as cv

def demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)   # 高斯模糊
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 50, 150)
    cv.imshow("Canny Edge", edge_output)
    dst = cv.bitwise_and(image, image, mask= edge_output)
    # cv.imshow("Color Edge", dst)
    cv.imshow("Original image",image)

    plt.figure()
    plt.imshow(edge_output)

src = cv.imread('F:/ML2/1.jpg')   # 图片地址
cv.namedWindow('input_image', cv.WINDOW_NORMAL)    #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
demo(src)
cv.waitKey(0)
cv.destroyAllWindows()

# face = misc.face()   # 该图像
plt.figure()         # 创建图形
plt.imshow(src)
plt.show()

'''
图像滤波  （中值滤波）
图像阈值处理
边缘检测   （Canny算子）

'''
