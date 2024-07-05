# -*- coding: utf-8 -*-  # 设置文件编码为UTF-8

import cv2  # 导入OpenCV库
import PSNR  # 导入PSNR模块，假设为自定义模块
import numpy  # 导入NumPy库，用于数值计算

cv2.setUseOptimized(True)  # 启用OpenCV的优化模式

# Parameters initialization  # 参数初始化
sigma = 25  # 噪声标准差
Threshold_Hard3D = 2.7 * sigma  # Hard Thresholding的阈值
First_Match_threshold = 2500  # 第一步匹配阈值，用于计算块之间的相似度
Step1_max_matched_cnt = 16  # 第一步匹配的最大块数
Step1_Blk_Size = 8  # 第一步块的大小，即8*8
Step1_Blk_Step = 3  # 第一步块的步进
Step1_Search_Step = 3  # 第一步搜索步进
Step1_Search_Window = 39  # 第一步搜索窗口大小，以像素为单位

Second_Match_threshold = 400  # 第二步匹配阈值，用于计算块之间的相似度
Step2_max_matched_cnt = 32  # 第二步匹配的最大块数
Step2_Blk_Size = 8  # 第二步块的大小，即8*8
Step2_Blk_Step = 3  # 第二步块的步进
Step2_Search_Step = 3  # 第二步搜索步进
Step2_Search_Window = 39  # 第二步搜索窗口大小，以像素为单位

Beta_Kaiser = 2.0  # Kaiser窗口的Beta参数


def init(img, _blk_size, _Beta_Kaiser):
    # 初始化函数，创建用于记录过滤后图像和权重的数组，并构造一个凯撒窗
    m_shape = img.shape  # 获取图像形状
    m_img = numpy.matrix(numpy.zeros(m_shape, dtype=float))  # 初始化过滤后图像数组
    m_wight = numpy.matrix(numpy.zeros(m_shape, dtype=float))  # 初始化权重数组
    K = numpy.matrix(numpy.kaiser(_blk_size, _Beta_Kaiser))  # 构造凯撒窗
    m_Kaiser = numpy.array(K.T * K)  # 计算凯撒窗
    return m_img, m_wight, m_Kaiser


def Locate_blk(i, j, blk_step, block_Size, width, height):
    # 根据索引和步长计算块的左上角坐标
    if i * blk_step + block_Size < width:
        point_x = i * blk_step
    else:
        point_x = width - block_Size  # 超出宽度时调整到边界

    if j * blk_step + block_Size < height:
        point_y = j * blk_step
    else:
        point_y = height - block_Size  # 超出高度时调整到边界

    m_blockPoint = numpy.array((point_x, point_y), dtype=int)  # 当前块的左上角坐标
    return m_blockPoint


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    point_x = _BlockPoint[0]  # 获取块的左上角 x 坐标
    point_y = _BlockPoint[1]  # 获取块的左上角 y 坐标

    # 计算搜索窗口的四个顶点坐标
    LX = point_x + Blk_Size / 2 - _WindowSize / 2  # 左上角 x
    LY = point_y + Blk_Size / 2 - _WindowSize / 2  # 左上角 y
    RX = LX + _WindowSize  # 右下角 x
    RY = LY + _WindowSize  # 右下角 y

    # 确保不超出图像边界
    if LX < 0:
        LX = 0
    elif RX > _noisyImg.shape[0]:
        LX = _noisyImg.shape[0] - _WindowSize
    if LY < 0:
        LY = 0
    elif RY > _noisyImg.shape[1]:
        LY = _noisyImg.shape[1] - _WindowSize

    return numpy.array((LX, LY), dtype=int)  # 返回搜索窗口左上角坐标


def Step1_fast_match(_noisyImg, _BlockPoint):
    (present_x, present_y) = _BlockPoint  # 获取当前块的左上角坐标
    Blk_Size = Step1_Blk_Size  # 设置当前块的大小
    Search_Step = Step1_Search_Step  # 设置搜索步长
    Threshold = First_Match_threshold  # 设置阈值用于匹配相似块
    max_matched = Step1_max_matched_cnt  # 设置最大匹配的块数
    Window_size = Step1_Search_Window  # 设置搜索窗口大小

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)  # 初始化用于记录相似块位置的数组
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)  # 初始化最终记录相似块的数组

    img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]  # 获取当前块的图像数据
    dct_img = cv2.dct(img.astype(numpy.float64))  # 对当前块进行二维余弦变换

    Final_similar_blocks[0, :, :] = dct_img  # 将当前块的变换结果存入最终相似块记录数组中
    blk_positions[0, :] = _BlockPoint  # 记录当前块的位置

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)  # 定义搜索窗口的位置
    blk_num = (Window_size - Blk_Size) / Search_Step  # 计算搜索窗口内可能找到的相似块数量
    blk_num = int(blk_num)

    (present_x, present_y) = Window_location  # 更新搜索窗口的起始位置

    similar_blocks = numpy.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)  # 初始化记录相似块的数组
    m_Blkpositions = numpy.zeros((blk_num ** 2, 2), dtype=int)  # 初始化记录相似块位置的数组
    Distances = numpy.zeros(blk_num ** 2, dtype=float)  # 初始化记录块之间距离的数组

    matched_cnt = 0  # 初始化匹配到的相似块数量
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]  # 获取候选相似块的图像数据
            dct_Tem_img = cv2.dct(tem_img.astype(numpy.float64))  # 对候选相似块进行二维余弦变换
            m_Distance = numpy.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)  # 计算当前块与候选相似块的距离

            if m_Distance < Threshold and m_Distance > 0:  # 判断是否找到符合条件的相似块
                similar_blocks[matched_cnt, :, :] = dct_Tem_img  # 记录相似块的变换结果
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)  # 记录相似块的位置
                Distances[matched_cnt] = m_Distance  # 记录当前块与相似块的距离
                matched_cnt += 1  # 增加匹配到的相似块数量
            present_y += Search_Step  # 更新搜索窗口的位置
        present_x += Search_Step
        present_y = Window_location[1]  # 恢复搜索窗口的纵向位置

    Distances = Distances[:matched_cnt]  # 裁剪距离数组，去除多余部分
    Sort = Distances.argsort()  # 对距离数组进行排序

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]  # 记录最终选定的相似块
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]  # 记录最终选定相似块的位置

    return Final_similar_blocks, blk_positions, Count  # 返回最终相似块数组、位置数组和匹配块数量


def Step1_3DFiltering(_similar_blocks):
    statis_nonzero = 0  # 初始化非零元素个数统计
    m_Shape = _similar_blocks.shape  # 获取相似块数组的形状信息

    # 对每个像素位置的向量进行处理
    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j])  # 对当前位置的向量进行二维余弦变换
            tem_Vct_Trans[numpy.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.  # 应用硬阈值处理
            statis_nonzero += tem_Vct_Trans.nonzero()[0].size  # 统计处理后的非零元素个数
            _similar_blocks[:, i, j] = cv2.idct(tem_Vct_Trans)[0]  # 对处理后的向量进行反余弦变换并更新到相似块数组中

    return _similar_blocks, statis_nonzero  # 返回处理后的相似块数组和非零元素个数统计


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    _shape = _similar_blocks.shape  # 获取相似块数组的形状信息
    if _nonzero_num < 1:
        _nonzero_num = 1  # 确保非零元素个数不小于1，避免除零错误
    block_wight = (1./_nonzero_num) * Kaiser  # 计算块权重，考虑凯撒窗口函数影响
    for i in range(Count):
        point = blk_positions[i, :]  # 获取当前相似块的位置信息
        tem_img = (1./_nonzero_num)*cv2.idct(_similar_blocks[i, :, :]) * Kaiser  # 对当前相似块进行反余弦变换，并乘以块权重和凯撒窗口函数
        m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += tem_img  # 更新基础图像
        m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += block_wight  # 更新权重图像


def BM3D_1st_step(_noisyImg):
    """第一步,基本去噪"""
    # 初始化一些参数：
    (width, height) = _noisyImg.shape   # 获取图像的宽度和高度
    block_Size = Step1_Blk_Size         # 设置块的大小
    blk_step = Step1_Blk_Step           # 设置块的步长
    Width_num = (width - block_Size) / blk_step  # 计算宽度上可以处理的块数
    Height_num = (height - block_Size) / blk_step  # 计算高度上可以处理的块数

    # 初始化几个数组
    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, Step1_Blk_Size, Beta_Kaiser)

    # 开始逐块处理,+2是为了避免边缘上不够的情况
    for i in range(int(Width_num + 2)):
        for j in range(int(Height_num + 2)):
            # 获取当前块的顶点位置
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)  # 获取相似块
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)  # 进行3D滤波
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)  # 硬阈值聚合处理

    Basic_img[:, :] /= m_Wight[:, :]  # 对基础图像进行权重归一化处理
    basic = numpy.matrix(Basic_img, dtype=int)  # 将基础图像转换为整数矩阵
    basic.astype(numpy.uint8)  # 转换为无符号8位整数类型（即灰度图像）
    return basic  # 返回基础去噪后的图像


def Step2_fast_match(_Basic_img, _noisyImg, _BlockPoint):
    (present_x, present_y) = _BlockPoint  # 获取当前坐标
    Blk_Size = Step2_Blk_Size  # 设置块的大小
    Threshold = Second_Match_threshold  # 第二次匹配的阈值
    Search_Step = Step2_Search_Step  # 设置搜索步长
    max_matched = Step2_max_matched_cnt  # 最大匹配块数
    Window_size = Step2_Search_Window  # 搜索窗口大小

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)  # 用于记录相似块的位置
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)  # 记录最终相似块的DCT系数
    Final_noisy_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)  # 记录最终噪声块的DCT系数

    img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float32))  # 对基本图像块进行二维余弦变换
    Final_similar_blocks[0, :, :] = dct_img  # 记录基本图像块的DCT系数

    n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_n_img = cv2.dct(n_img.astype(numpy.float32))  # 对噪声图像块进行二维余弦变换
    Final_noisy_blocks[0, :, :] = dct_n_img  # 记录噪声图像块的DCT系数

    blk_positions[0, :] = _BlockPoint  # 记录当前块的位置

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)  # 定义搜索窗口位置
    blk_num = (Window_size - Blk_Size) / Search_Step  # 计算可以找到的相似块数量
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)  # 初始化相似块数组
    m_Blkpositions = numpy.zeros((blk_num ** 2, 2), dtype=int)  # 初始化相似块位置数组
    Distances = numpy.zeros(blk_num ** 2, dtype=float)  # 初始化距离数组，用于记录块之间的相似度

    # 在搜索窗口中进行遍历搜索相似块
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(numpy.float32))
            m_Distance = numpy.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            # 记录符合要求的相似块
            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]

    Distances = Distances[:matched_cnt]  # 只保留实际找到的相似块距离
    Sort = Distances.argsort()  # 根据距离排序相似块

    # 统计找到的相似块数
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]

            (present_x, present_y) = m_Blkpositions[Sort[i - 1], :]
            n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            Final_noisy_blocks[i, :, :] = cv2.dct(n_img.astype(numpy.float64))

    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count


def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    m_Shape = _Similar_Bscs.shape  # 获取相似基本块的形状信息
    Wiener_wight = numpy.zeros((m_Shape[1], m_Shape[2]), dtype=float)  # 初始化维纳权重数组

    # 遍历处理每一个基本块
    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_vector = _Similar_Bscs[:, i, j]  # 获取当前基本块的向量表示
            tem_Vct_Trans = numpy.matrix(cv2.dct(tem_vector))  # 对基本块进行二维余弦变换
            Norm_2 = numpy.float64(tem_Vct_Trans.T * tem_Vct_Trans)  # 计算变换后向量的二范数平方
            m_weight = Norm_2 / (Norm_2 + sigma**2)  # 计算维纳滤波器的权重
            if m_weight != 0:
                Wiener_wight[i, j] = 1. / (m_weight**2 * sigma**2)  # 计算维纳滤波器的权重值
            tem_vector = _Similar_Imgs[:, i, j]  # 获取当前噪声块的向量表示
            tem_Vct_Trans = m_weight * cv2.dct(tem_vector)  # 对噪声块进行维纳滤波处理
            _Similar_Bscs[:, i, j] = cv2.idct(tem_Vct_Trans)[0]  # 对处理后的块进行逆余弦变换

    return _Similar_Bscs, Wiener_wight  # 返回处理后的相似基本块和维纳权重数组


def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    _shape = _Similar_Blks.shape  # 获取相似块的形状信息
    block_wight = _Wiener_wight  # 维纳权重乘以凯撒窗

    # 遍历处理每一个相似块
    for i in range(Count):
        point = blk_positions[i, :]  # 获取块的位置信息
        tem_img = _Wiener_wight * cv2.idct(_Similar_Blks[i, :, :])  # 对相似块进行维纳滤波和逆余弦变换
        m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += tem_img  # 更新基本图像
        m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2]] += block_wight  # 更新权重图像


def BM3D_2nd_step(_basicImg, _noisyImg):
    # 初始化一些参数：
    (width, height) = _noisyImg.shape  # 获取噪声图像的宽高
    block_Size = Step2_Blk_Size  # 设置块大小
    blk_step = Step2_Blk_Step  # 设置块步长
    Width_num = (width - block_Size) / blk_step  # 计算宽度方向上的块数量
    Height_num = (height - block_Size) / blk_step  # 计算高度方向上的块数量

    # 初始化几个数组
    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)  # 初始化图像、权重和凯撒窗

    # 开始逐block的处理，+2是为了避免边缘上不够
    for i in range(int(Width_num + 2)):
        for j in range(int(Height_num + 2)):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)  # 获取当前块的顶点位置
            Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint)  # 快速匹配找到相似块和噪声块
            Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs)  # 进行3D滤波
            Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)  # 进行维纳聚合

    m_img[:, :] /= m_Wight[:, :]  # 对基本图像进行归一化处理
    Final = numpy.matrix(m_img, dtype=int)  # 转换为整型矩阵
    Final.astype(numpy.uint8)  # 转换为无符号整型

    return Final  # 返回最终去噪结果


if __name__ == '__main__':
    cv2.setUseOptimized(True)   # 开启OpenCV的优化功能

    img_name = "for.jpg"  # 图像的路径
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)    # 以灰度模式读入图像

    # 记录程序运行时间
    e1 = cv2.getTickCount()  # 获取开始时间
    Basic_img = BM3D_1st_step(img)  # 第一步BM3D基本去噪处理
    e2 = cv2.getTickCount()  # 获取第一步结束时间
    time = (e2 - e1) / cv2.getTickFrequency()   # 计算第一步执行时间
    print("The Processing time of the First step is %f s" % time)
    cv2.imwrite("forBasic.jpg", Basic_img)  # 将第一步处理结果保存为图像
    psnr = PSNR.PSNR(img, Basic_img)  # 计算第一步处理后的图像与原始图像的PSNR
    print("The PSNR between the two img of the First step is %f" % psnr)

    Final_img = BM3D_2nd_step(Basic_img, img)  # 第二步BM3D进一步去噪处理
    e3 = cv2.getTickCount()  # 获取第二步结束时间
    time = (e3 - e2) / cv2.getTickFrequency()   # 计算第二步执行时间
    print("The Processing time of the Second step is %f s" % time)
    cv2.imwrite("forFinal.jpg", Final_img)  # 将第二步处理结果保存为图像
    psnr = PSNR.PSNR(img, Final_img)  # 计算第二步处理后的图像与原始图像的PSNR
    print("The PSNR between the two img of the Second step is %f" % psnr)
    time = (e3 - e1) / cv2.getTickFrequency()   # 计算总体执行时间
    print("The total Processing time is %f s" % time)