import cv2
import numpy as np

# Create blank image (black background)
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw circles (center, radius)
cv2.circle(img, (150, 150), 60, (255, 255, 255), -1)  # filled
cv2.circle(img, (350, 150), 40, (255, 255, 255), 3)   # outline
cv2.circle(img, (250, 350), 80, (255, 255, 255), -1)

# Save image
cv2.imwrite("test_circles.jpg", img)

print("Saved as test_circles.jpg")