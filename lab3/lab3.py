import cv2
import numpy as np

def nothing(x):
    pass
cv2.namedWindow("bars")
cv2.namedWindow("bars2")
cv2.createTrackbar('dp', 'bars' , 1, 10, nothing)
cv2.createTrackbar('minDist', 'bars' , 10, 100, nothing)
cv2.createTrackbar('circles', 'bars' , 2, 100, nothing)
cv2.createTrackbar('param1', 'bars' , 20, 100, nothing)
cv2.createTrackbar('param2', 'bars' , 54, 500, nothing)
cv2.createTrackbar('minRadius', 'bars' , 18, 100, nothing)
cv2.createTrackbar('maxRadius', 'bars' , 112, 500, nothing)

cv2.createTrackbar('thresh1', 'bars2' , 58, 500, nothing)
cv2.createTrackbar('thresh2', 'bars2' , 272, 500, nothing)



while True:
    image = cv2.imread("owl_1.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    circled_im = gray_blur.copy()

    canny_im = cv2.Canny(gray_blur, threshold1=cv2.getTrackbarPos('thresh1', 'bars2'),
                         threshold2=cv2.getTrackbarPos('thresh2', 'bars2'))
    circles = cv2.HoughCircles(image=canny_im,
                               method=cv2.HOUGH_GRADIENT,
                               dp=cv2.getTrackbarPos('dp', 'bars'),
                               minDist=cv2.getTrackbarPos('minDist', 'bars'),
                               circles=cv2.getTrackbarPos('circles', 'bars'),
                               param1=cv2.getTrackbarPos('param1', 'bars'),
                               param2=cv2.getTrackbarPos('param2', 'bars'),
                               minRadius=cv2.getTrackbarPos('minRadius', 'bars'),
                               maxRadius=cv2.getTrackbarPos('maxRadius', 'bars'))

    if circles is not None:
        circles = np.int16(np.around(circles))
        if len(circles.shape) == 3:
            for i in circles[0, :]:
                center = (i[0], i[1])
                # центр кола
                cv2.circle(circled_im, center, 1, (0, 100, 100), 3)
                # лінія кола
                radius = i[2]
                cv2.circle(circled_im, center, radius, (255, 0, 255), 3)

    result_image = cv2.hconcat([gray_blur, canny_im, circled_im])
    cv2.imshow('win', result_image)
    cv2.waitKey(1)