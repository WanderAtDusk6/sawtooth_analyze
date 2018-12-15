# 边缘检测（拟合）
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
'''
图像滤波  （中值滤波）
图像阈值处理
边缘检测   （Canny算子）
拟合  ？？？（curve_fit()）
'''
def get_edge(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)   # 高斯模糊
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 50, 200)   # 50 150
    cv.imshow("Canny Edge", edge_output)
    # dst = cv.bitwise_and(image, image, mask=edge_output)    #color edge的输出
    # cv.imshow("Color Edge", dst)
    # cv.imshow("Original image",image)
    cv.imwrite("F://ML2/1_50_200.jpg", edge_output)

def ImageToMatrix(im):
    # 读取图片
    # im = Image.open(filename)
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data

# get_edge(src)
image_path = "F://ML2/1_50_150_T.jpg"  # 某路径

src = cv.imread(image_path, cv.WINDOW_NORMAL)
cv.imshow("Original_image",src)
cv.namedWindow('input_image', cv.WINDOW_NORMAL)    # 设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)

data = ImageToMatrix(src)


cv.waitKey(0)
cv.destroyAllWindows()
