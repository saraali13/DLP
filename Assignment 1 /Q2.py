import numpy as np
import cv2


def convolution_filter(image, kernel):
    # get the dim of the image and the kernel
    height, width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape

    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))  # first flip left to right, then flip up and down

    # check how much padding is required
    padding_height = (kernel_height) // 2
    padding_width = (kernel_width) // 2

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


# Read image
sample_image = cv2.imread('dog.jpg')
kernel_ = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # given

filter_img = convolution_filter(sample_image, kernel_)

# Clip values to valid pixel range(8-bit range)
filter_img = np.abs(filter_img)
noisy_image = np.clip(filter_img, 0,
                      255)  # gaussian noise can produce negative values or values greater than 255 that are not valid pixel vals so that why clipping the vals between 0 and 255
noisy_image = noisy_image.astype(
    np.uint8)  # after adding noise some vals might be converted into float so converting them back to unsigned 8 bit int

# Show result
cv2.imshow("Noisy Image", noisy_image)
cv2.waitKey(0)
