import cv2
import numpy as np

# Create a blank image
img = np.zeros((300, 500, 3), dtype=np.uint8)

# Blue shades
for i in range(5):
    img[0:100, i*100:(i+1)*100] = (50*i, 0, 0)

# Green shades
for i in range(5):
    img[100:200, i*100:(i+1)*100] = (0, 50*i, 0)

# Red shades
for i in range(5):
    img[200:300, i*100:(i+1)*100] = (0, 0, 50*i)

# Show the image
cv2.imshow("Color Shades Using Pixels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()