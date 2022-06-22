import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# imgPath = r"D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\train\MY_1708_CM_抽面\2_2_1_4_1_1_7_9.jpg"
# imgPath = r"D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\train\MY_1757_BSY_上半玻璃门\9_2_2_2_2_1_12_9.jpg"
# imgPath = r"D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\train\MY_1770_BSY_上半玻璃门\36_3_2_1_3_1_10_3.jpg"
imgPath = r"D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\train\MY_1771_BSY_上半玻璃门\49_2_2_1_3_2_10_7.jpg"
img = Image.open(imgPath)
img = img.resize((256, 256))
print(img)
imgArray = np.array(img)
print(imgArray, imgArray.shape)
imgArray2 = imgArray.mean(axis=-1)
plt.imshow(imgArray2, cmap='gray')
plt.show()
plt.figure()
# plt.imshow(imgArray)
print(img)
img = img.convert('L')
print(img)
imgArray = np.array(img)
plt.figure()
# plt.imshow(imgArray)

print(imgArray, imgArray.shape)
imgArray = np.stack((img,)*3, axis=-1)
plt.figure()
plt.imshow(imgArray)

print(imgArray)
plt.show()

