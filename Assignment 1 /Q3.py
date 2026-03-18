import numpy as np
import cv2

def convolution_filter(image, kernel):
    # get the dim of the image and the kernel
    height, width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape

    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))  # first flip left to right, then flip up and down

    # check how much padding is required
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # Handle both grayscale and color images
    # colored image
    if len(image.shape) == 3:
        channels = image.shape[2]
        padded = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)),
                        mode="constant")
        output = np.zeros((height, width, channels))  # output with same number of channels

        # Apply convolution to each channel (per pixel) separately
        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    region = padded[i:i + kernel_height, j:j + kernel_width, c]
                    output[i, j, c] = np.sum(region * kernel)
    else:
        # Grayscale image
        padded = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode="constant")
        output = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                region = padded[i:i + kernel_height, j:j + kernel_width]
                output[i, j] = np.sum(region * kernel)

    return output

def gaussian_kernel(size, sigma): #for smoothing

    arr = np.arange(-(size // 2), size // 2 + 1) #create a 1D array
    x, y = np.meshgrid(arr, arr) #coverts that 1D array to 2D by repeating the 1D rows

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2)) #Gaussian function formula
    kernel = kernel / np.sum(kernel)#normalize the kernel

    return kernel

# Load grayscale image
image = cv2.imread('Q1.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image file not found. Check the path.")

cv2.imshow("Noisy Image", image)
cv2.waitKey(0)

# Denoise with 7x7 Gaussian kernel (sigma=1.0)
g_kernel = gaussian_kernel(7, 1.0)
denoised = convolution_filter(image.astype(np.float32), g_kernel)
denoised = np.clip(denoised, 0, 255).astype(np.uint8)

cv2.imshow("Denoised Image", denoised)
cv2.waitKey(0)

# Sharpening kernel (5x5)
sharpening_kernel = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, -476, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32) * (-1.0 / 256.0)

# Apply sharpening filter to original image
sharpened = convolution_filter(image.astype(np.float32), sharpening_kernel)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

cv2.imshow("Sharpened Image", sharpened)
cv2.waitKey(0)
