import cv2
import matplotlib.pyplot as plt

full_path = "./data/incline_L.png"
im = cv2.imread(full_path)
image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(image)
x = plt.ginput(4)
print(x)

full_path = "./data/incline_R.png"
im = cv2.imread(full_path)
image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
