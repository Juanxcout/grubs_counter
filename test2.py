import cv2
import numpy as np
import imutils

# Read the image
imagen = cv2.imread("Prueba 2_Cortada.jpg")

# Resize the image
imagen = imutils.resize(imagen, width=1000)

# Convert image to grayscale
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Preprocessing
smoothed_image = cv2.GaussianBlur(gray, (5, 5), 0)
adaptive_threshold = cv2.adaptiveThreshold(smoothed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Contour detection
contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image and count worms
num_worms = 0
for idx, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area < 50:  # Filter out small contours
        continue
    # Get centroid of contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # Draw contour
    cv2.drawContours(imagen, [contour], -1, (0, 255, 0), 1)
    # Draw number above contour
    num_worms += 1
    cv2.putText(imagen, str(num_worms), (cX, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the result and show the count
cv2.putText(imagen, f"Number of worms: {num_worms}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Detected Worms", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

