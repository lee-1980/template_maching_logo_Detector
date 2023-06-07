import cv2

img = cv2.imread("logo/godaddy/logo2.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, binary) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

print(thresh)
cv2.imshow("Original Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Gray Image",gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Black and White Image",binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
