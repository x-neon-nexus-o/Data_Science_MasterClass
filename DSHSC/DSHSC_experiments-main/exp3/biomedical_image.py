import cv2
import matplotlib.pyplot as plt

# Load the glioma MRI image
image = cv2.imread("DSHSC_experiments-main\\exp3\\glioma_img.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal using Gaussian Blur
denoised = cv2.GaussianBlur(gray, (5, 5), 0)

# Contrast enhancement using Histogram Equalization
enhanced = cv2.equalizeHist(denoised)

# Image segmentation using Otsu's thresholding
_, segmented = cv2.threshold(
    enhanced, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original MRI")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(enhanced, cmap="gray")
plt.title("Preprocessed Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(segmented, cmap="gray")
plt.title("Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()