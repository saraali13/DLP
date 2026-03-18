import numpy as np
import cv2

# Read image
image = cv2.imread('dog.jpg')

if image is None:
    print("Image not found")


# Generate Gaussian noise
mean = 0
std = 15
noise = np.random.normal(mean, std, image.shape)

# Add noise to the original image
noisy_image = image + noise

# Clip values to valid pixel range(8-bit range)
noisy_image = np.clip(noisy_image, 0, 255) #gaussian noise can produce negative values or values greater than 255 that are not valid pixel vals so that why clipping the vals between 0 and 255
noisy_image = noisy_image.astype(np.uint8) #after adding noise some vals might be converted into float so converting them back to unsigned 8 bit int

cv2.imshow("Noisy Image", noisy_image)
cv2.waitKey(0)
