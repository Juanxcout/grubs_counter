import cv2
import numpy as np
import imutils

# Leer la imagen
imagen = cv2.imread("Prueba 2_Cortada.jpg")

# Redimensionar la imagen
imagen = imutils.resize(imagen, width=800, height=800)

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Suavizado de imagen
smoothed_image = cv2.GaussianBlur(gray, (7, 7), 0)

# Umbral adaptativo
adaptive_threshold = cv2.adaptiveThreshold(smoothed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

# Detección de contornos
contours, hierarchy = cv2.findContours(adaptive_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

# Inicializar el número de gusanos
num_worms = 0

# Dibujar contornos en la imagen original y contar gusanos
for i, contour in enumerate(contours):
    if cv2.contourArea(contour) < 39:  # Filtrar contornos pequeños
        continue
    # Obtener la jerarquía del contorno actual
    current_hierarchy = hierarchy[0][i]
    # Si el contorno actual no tiene un padre (es un contorno principal)
    if current_hierarchy[3] == -1:
        num_worms += 1
        # Dibujar contorno
        cv2.drawContours(imagen, [contour], -1, (0, 255, 0), 2)

# Mostrar el resultado y mostrar el conteo
cv2.putText(imagen, f"Number of worms: {num_worms}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Detected Worms", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
