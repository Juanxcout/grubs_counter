import cv2
import numpy as np
import imutils

imagen = cv2.imread("Prueba 2_Cortada.jpg", 0)
imagen = imutils.resize(imagen, width=1000)

umbral, binarizada = cv2.threshold(imagen, 180, 255, cv2.THRESH_BINARY)

contornos, hierarchy = cv2.findContours(binarizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imagen, contornos, -1, (0, 255, 0), 1)
cv2.imshow("binarizada", imagen)
cv2.imshow("imagen", imagen)
cv2.imshow("contornos", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()