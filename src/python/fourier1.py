import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../resources/images/bb.jpg', 0)
rows, cols = img.shape
print(rows, cols)

# 计算DFT效率最佳的尺寸
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
print(nrows, ncols)

nimg = np.zeros((nrows, ncols))
nimg[:rows, :cols] = img
img = nimg

# OpenCV计算快速傅里叶变换，输入图像应首先转换为np.float32，然后使用函数cv2.dft()和cv2.idft()。
# 返回结果与Numpy相同，但有两个通道。第一个通道为有结果的实部，第二个通道为有结果的虚部。
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# 首先创建一个mask，中心正方形为1，其他均为0
# 如何删除图像中的高频内容，即我们将LPF应用于图像。它实际上模糊了图像。
# 为此首先创建一个在低频时具有高值的掩码，即传递LF内容，在HF区域为0。
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# 应用掩码Mask和求逆DTF
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()