import cv2
import numpy as np

# read the image from file
image = cv2.imread("kot.jpg")
image = cv2.resize(image, [image.shape[1] // 2, image.shape[0] // 2])

# define matrix for transform and do affine transformation
pos1 = np.array([[1, 0, 55],
                 [0, 1, 55]], dtype=np.float32)
inv_image1 = cv2.warpAffine(image, pos1, [image.shape[1], image.shape[0]])

# rotation
theta = (45 * np.pi) / 180
cos = np.cos(theta)
sin = np.sin(theta)
Tx = (1 - cos) * (image.shape[1] / 2) - sin * (image.shape[0] / 2)
Ty = sin * (image.shape[1] / 2) + (1 - cos) * (image.shape[0] / 2)

pos2 = np.array([[cos, sin, Tx],
                 [-sin, cos, Ty]], dtype=np.float32)
inv_image2 = cv2.warpAffine(image, pos2, [image.shape[1], image.shape[0]])

# scale
pos3 = np.array([[0.5, 0, 0],
                 [0, 0.5, 0]], dtype=np.float32)
inv_image3 = cv2.warpAffine(image, pos3, [image.shape[1], image.shape[0]])

# scale
pos4 = np.array([[1, 0.5, 0],
                 [0.5, 1, 0]], dtype=np.float32)
inv_image4 = cv2.warpAffine(image, pos4, [image.shape[1], image.shape[0]])

# doing a 1 channel grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# doing from 1 to 3 channel grayscale for show with another images
gray = np.reshape(gray, [gray.shape[0], gray.shape[1], 1])
gray = np.repeat(gray, 3, axis=-1)

result_image = cv2.vconcat(
    [cv2.hconcat([image, inv_image1, inv_image2]),
     cv2.hconcat([inv_image3, inv_image4, gray])
     ])
cv2.imshow("window", result_image)
cv2.waitKey(0)
