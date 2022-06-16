import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../resources/images/bb.jpg', 0)

# 使用Numpy实现傅里叶变换：fft包
# fft.fft2() 进行频率变换
# 参数1：输入图像的灰度图
# 参数2：>输入图像 用0填充；  <输入图像 剪切输入图像； 不传递 返回输入图像
f = np.fft.fft2(img)

# 一旦得到结果，零频率分量（直流分量）将出现在左上角。
# 如果要将其置于中心，则需要使用np.fft.fftshift()将结果在两个方向上移动。
# 一旦找到了频率变换，就能找到幅度谱。
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# 找到了频率变换，就可以进行高通滤波和重建图像，也就是求逆DFT
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 255
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# 图像渐变章节学习到：高通滤波是一种边缘检测操作。这也表明大部分图像数据存在于频谱的低频区域。
# 仔细观察结果可以看到最后一张用JET颜色显示的图像，有一些瑕疵（它显示了一些波纹状的结构，这就是所谓的振铃效应。）
# 这是由于用矩形窗口mask造成的，掩码mask被转换为sinc形状，从而导致此问题。所以矩形窗口不用于过滤，更好的选择是高斯mask。）
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()
