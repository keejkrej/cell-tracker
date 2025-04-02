import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

# --- Parameters ---

# == Image Generation Parameters ==
width = 256
height = 256
# background_level = 80  # Base gray level for the background - REMOVED
gradient_min = 50      # Minimum intensity for the gradient background
gradient_max = 150     # Maximum intensity for the gradient background
noise_std_dev = 10     # Standard deviation of Gaussian noise ADDED ON TOP of the gradient (reduce if too noisy)
# Blob definitions: (center_x, center_y, amplitude, sigma)
blobs = [
    (60, 70,   180, 10),  # Smaller, intense blob
    (180, 100, 150, 18),  # Larger, less intense peak blob
    (120, 190, 170, 12),  # Medium blob
    (200, 200, 100, 8),   # Small, dimmer blob
]

# == DoG Filter Parameters ==
sigma1 = 2.0  # Standard deviation for the narrower Gaussian (adjust as needed)
sigma2 = sigma1 * 1.6 # Standard deviation for the wider Gaussian (common ratio)

# --- Helper Function for Fourier Spectrum ---

def calculate_magnitude_spectrum(image):
    """Calculates the log-scaled magnitude spectrum of an image."""
    # Ensure input is float type for FFT
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)

    # Compute the 2D Discrete Fourier Transform
    dft = fft2(image)

    # Shift the zero-frequency component (DC component) to the center
    dft_shifted = fftshift(dft)

    # Calculate the magnitude spectrum (absolute value of complex numbers)
    magnitude_spectrum = np.abs(dft_shifted)

    # Apply logarithmic scale for better visualization (log(1 + x))
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)

    return log_magnitude_spectrum

# --- 1. Generate Sample Image ---

print("Generating sample image with gradient background...")
# Generate Coordinate Grid
x_coords = np.arange(0, width)
y_coords = np.arange(0, height)
X, Y = np.meshgrid(x_coords, y_coords)

# --- Create Gradient Background with Optional Noise ---
print("Generating gradient background...")
# Normalized coordinates [0, 1]
# Using width-1 and height-1 handles potential division by zero if width/height is 1
norm_X = X / (width - 1) if width > 1 else np.zeros_like(X)
norm_Y = Y / (height - 1) if height > 1 else np.zeros_like(Y)

# Create a diagonal gradient factor (0=top-left, 1=bottom-right)
# Other options: norm_X for horizontal, norm_Y for vertical
gradient_factor = (norm_X + norm_Y) / 2.0

# Calculate the gradient base
gradient = gradient_min + (gradient_max - gradient_min) * gradient_factor
gradient = gradient.astype(np.float32) # Ensure float32

# Optional: Add low-amplitude noise on top of the gradient
if noise_std_dev > 0:
    print(f"Adding noise (std dev: {noise_std_dev}) to gradient...")
    noise = np.random.normal(loc=0, scale=noise_std_dev, size=(height, width)).astype(np.float32)
    background = gradient + noise # Base for adding blobs is now gradient + noise
else:
    background = gradient # Base for adding blobs is just the gradient

# Add Gaussian Blobs to the gradient background
image_generated_float = background.copy() # Start with gradient(+noise)
print("Adding blobs...")
for (cx, cy, amp, sigma) in blobs:
    dist_sq = (X - cx)**2 + (Y - cy)**2
    gaussian_blob = amp * np.exp(-dist_sq / (2 * sigma**2))
    image_generated_float += gaussian_blob # Add blob intensity

# Clip values and Convert to uint8 for typical image format
image_generated_clipped = np.clip(image_generated_float, 0, 255)
generated_image_uint8 = image_generated_clipped.astype(np.uint8)
print("Image generation complete.")

# --- 2. Prepare Image for Filtering ---
image_to_filter_float = generated_image_uint8.astype(np.float32) / 255.0

# --- 3. Apply Difference of Gaussians (DoG) Band-Pass Filter ---

print(f"Applying DoG filter (sigma1={sigma1}, sigma2={sigma2:.2f})...")
ksize1 = (0, 0)
ksize2 = (0, 0)
blurred1 = cv2.GaussianBlur(image_to_filter_float, ksize1, sigmaX=sigma1, sigmaY=sigma1)
blurred2 = cv2.GaussianBlur(image_to_filter_float, ksize2, sigmaX=sigma2, sigmaY=sigma2) # <-- Keep this result
dog_image = blurred1 - blurred2 # Result is float, centered around 0
print("DoG filtering complete.")

# --- 4. Calculate Spectrums ---
print("Calculating Fourier Spectrums...")
original_spectrum = calculate_magnitude_spectrum(image_to_filter_float)
blurred1_spectrum = calculate_magnitude_spectrum(blurred1)
blurred2_spectrum = calculate_magnitude_spectrum(blurred2)
dog_spectrum = calculate_magnitude_spectrum(dog_image) # Use raw DoG result for spectrum
print("Spectrum calculation complete.")


# --- 5. Visualization ---

print("Preparing visualization...")
# Normalize the DoG result *image* to [0, 1] range for grayscale display
dog_display = cv2.normalize(dog_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Create figure - 2 rows, 4 columns
plt.figure(figsize=(10, 5))

# == Row 1: Images ==
plt.subplot(2, 4, 1)
plt.imshow(generated_image_uint8, cmap='gray')
plt.title('Original Generated Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(blurred1, cmap='gray')
plt.title(f'Blurred 1 (LP, σ={sigma1:.1f})')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(blurred2, cmap='gray')
plt.title(f'Blurred 2 (LP, σ={sigma2:.1f})')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(dog_display, cmap='gray')
plt.title('DoG Filtered (Band-Pass)')
plt.axis('off')

# == Row 2: Spectrums ==
plt.subplot(2, 4, 5)
plt.imshow(original_spectrum, cmap='magma')
plt.title('Original Spectrum')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(blurred1_spectrum, cmap='magma')
plt.title('Blurred 1 Spectrum')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(blurred2_spectrum, cmap='magma')
plt.title('Blurred 2 Spectrum')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(dog_spectrum, cmap='magma')
plt.title('DoG Spectrum')
plt.axis('off')


plt.tight_layout()
print("Displaying results...")
plt.show()

print("Script finished.")