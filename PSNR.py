import cv2  # 导入OpenCV库
import math  # 导入数学库
import numpy  # 导入NumPy库

def PSNR(img1, img2):
    """计算图像的峰值信噪比（PSNR）"""
    D = numpy.array(img1 - img2, dtype=numpy.int64)  # 计算差值矩阵
    D[:, :] = D[:, :]**2  # 差值矩阵元素平方
    RMSE = D.sum() / img1.size  # 计算均方根误差（RMSE）
    psnr = 10 * math.log10(float(255. ** 2) / RMSE)  # 计算PSNR
    return psnr

if __name__ == "__main__":
    img1 = cv2.imread("original 2D4F.bmp", cv2.IMREAD_GRAYSCALE)  # 读取第一张灰度图像
    img2 = cv2.imread("Basic2.jpg", cv2.IMREAD_GRAYSCALE)  # 读取第二张灰度图像
    psnr = PSNR(img1, img2)  # 计算两张图像的PSNR
    print("The PSNR between the two img of the two is %f" % psnr)

    img1 = cv2.imread("original 2D4F.bmp", cv2.IMREAD_GRAYSCALE)  # 重新读取第一张灰度图像
    img2 = cv2.imread("Final2.jpg", cv2.IMREAD_GRAYSCALE)  # 重新读取第二张灰度图像
    psnr = PSNR(img1, img2)  # 计算两张图像的PSNR
    print("The PSNR between the two img of the two is %f" % psnr)