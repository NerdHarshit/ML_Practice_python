from torchvision import transforms
from PIL import Image

image1 = Image.open("Assets\\cat1.jpg")
image2 = Image.open("Assets\\spacex.jpg")

transform1 = transforms.ToTensor()
transform2 = transforms.ToTensor()

tensor1 = transform1(image1)
tensor2 = transform2(image2)


#print(tensor1.shape)
#print(tensor2.shape)

from matplotlib import pyplot as plt

plt.imshow(tensor1[0],cmap="grey")
plt.title("red channel")
plt.show()