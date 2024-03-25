
# REFERENCE LINKS
# https://www.geeksforgeeks.org/image-compression-using-k-means-clustering/
# https://www.youtube.com/watch?v=DG7YTlGnCEo (singular value decomposition image compression video)
# PNSR value calculation is referenced from AI


import cv2
import numpy as np

# loading the image
original_image = cv2.imread('image.png')

# converting the image into grayscale
grsyscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# number of clusters for k-means compression
num_clusters = 16

# performing the k-means clustering compression
pixel_values = grsyscale_image.reshape(-1, 1).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
compressed_image_kmeans = centers[labels.flatten()].reshape(grsyscale_image.shape).astype(np.uint8)

# performing the singular value decomposition compression
k = 100 
U, S, Vt = np.linalg.svd(grsyscale_image, full_matrices=False)
compressed_image_svd = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

# calculating PSNR value
def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

kmeans_pnsr_value = calculate_psnr(grsyscale_image, compressed_image_kmeans)
svd_psnr_value = calculate_psnr(grsyscale_image, compressed_image_svd)

print(f'K-means PSNR value: {kmeans_pnsr_value:.2f}')
print(f'SVD PSNR value: {svd_psnr_value:.2f}')

cv2.imwrite('kmeans_compressed.png', compressed_image_kmeans)
cv2.imwrite('svd_compressed.png', compressed_image_svd)
