
#REFERENCES 
# https://blog.demofox.org/2022/02/26/image-sharpening-convolution-kernels/
# https://www.youtube.com/watch?v=MGDOrCpQwO4 (sharpen image using OpenCV)

import cv2
import numpy as np
import timeit

# reading the image
image = cv2.imread('image.png')

# sharpening matrix
sharpen_matrix = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

# sharpening from scratch
def sharp_image_from_scratch(image):
    scratch_image = np.empty_like(image)
    # applying convolution for each channel
    for channel in range(3):
        scratch_image[:, :, channel] = cv2.filter2D(image[:, :, channel], -1, sharpen_matrix)
    return scratch_image

# sharpeing image from opencv
def sharp_image_from_opencv (image):
    return cv2.filter2D(src=image, ddepth=-1, kernel=sharpen_matrix)

# gets the execurion time of both functiond
scratch_image_execution_time = timeit.timeit(lambda: sharp_image_from_scratch(image), number=10)
opencv_image_execution_time = timeit.timeit(lambda: sharp_image_from_opencv (image), number=10)

# daves the images
cv2.imwrite('sharp_image/scratch_image.png', sharp_image_from_scratch(image))
cv2.imwrite('sharp_image/opencv_image.png', sharp_image_from_opencv (image))

# print the execution time
print(f" scratch execution time: {scratch_image_execution_time:.4f} seconds")
print(f"OpenCV execution time: {opencv_image_execution_time:.4f} seconds")
