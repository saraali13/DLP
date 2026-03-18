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

def correlation(image, kernel):
    # get the dim of the image and the kernel
    height, width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape

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

image = cv2.imread('shelf.jpg', 0)
template = cv2.imread('template.jpg', 0)

#mean subtraction
image = image - np.mean(image)
template = template - np.mean(template)

conv_result = convolution_filter(image, template)

y1, x1 = np.unravel_index(np.argmax(conv_result), conv_result.shape) #return row,col value of agrmax-> most similar region
print("Convolution Location:", x1, y1)

corr_result = correlation(image, template)

y2, x2 = np.unravel_index(np.argmax(corr_result), corr_result.shape)
print("Correlation Location:", x2, y2)

h, w = template.shape

top_left_conv_x = x1 - w // 2
top_left_conv_y = y1 - h // 2

top_left_corr_x = x2 - w // 2
top_left_corr_y = y2 - h // 2

image_color = cv2.imread('shelf.jpg')

# Convolution rectangle (Blue)
cv2.rectangle(image_color,
              (top_left_conv_x, top_left_conv_y),
              (top_left_conv_x + w, top_left_conv_y + h),
              (255, 0, 0), 2)

# Correlation rectangle (Green)
cv2.rectangle(image_color,
              (top_left_corr_x, top_left_corr_y),
              (top_left_corr_x + w, top_left_corr_y + h),
              (0, 255, 0), 2)

cv2.imshow("Convolution vs Correlation", image_color)
cv2.waitKey(0)

#e
#correlation is more accurate for template matching
#f
#correlation as no flipping is required
#g
#Correlation is better for template matching as we want to match the exact structure and in convolution flipping is done which changes the orientation
#h
#the output image shows that using correlation approach is more effective
