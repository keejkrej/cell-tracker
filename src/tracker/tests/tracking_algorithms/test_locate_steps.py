import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import ndimage
from skimage.filters import gaussian
from skimage.feature import peak_local_max

class LocateStepsTest:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path(__file__).parent
        
        self.test_data_path = Path(base_path) / 'test_data'
        self.results_path = Path(base_path) / 'results'
        self.viz_path = Path(base_path) / 'visualizations'
        
        # Ensure directories exist
        for path in [self.test_data_path, self.results_path, self.viz_path]:
            path.mkdir(parents=True, exist_ok=True)

    def generate_test_image(self, size=100, n_particles=3, noise_level=0.1):
        """Generate a test image with known particle positions."""
        # Create empty image
        image = np.zeros((size, size))
        
        # Generate random positions for particles
        positions = np.random.rand(n_particles, 2) * (size * 0.8) + (size * 0.1)
        
        # Add particles with Gaussian profiles
        for pos in positions:
            y, x = np.ogrid[-pos[0]:size-pos[0], -pos[1]:size-pos[1]]
            # Create Gaussian particle
            particle = np.exp(-(x*x + y*y)/(2*3*3))
            mask = particle > 0.01
            image[mask] += particle[mask]
        
        # Add noise
        image += np.random.normal(0, noise_level, image.shape)
        
        # Add background variation
        y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
        background = 0.2 * (1 + np.sin(x/20) * np.sin(y/20))
        image += background
        
        return image, positions

    def bandpass(self, image, sigma_low=1, sigma_high=None, threshold=None):
        """Apply bandpass filter to isolate features of interest.
        
        Args:
            image: Input image
            sigma_low: Gaussian sigma for smoothing noise
            sigma_high: Gaussian sigma for background subtraction
            threshold: Optional threshold for final image
        """
        # Remove noise with low-pass filter
        smoothed = gaussian(image, sigma=sigma_low)
        
        # Remove background with high-pass filter
        if sigma_high is not None:
            background = gaussian(image, sigma=sigma_high)
            result = smoothed - background
        else:
            result = smoothed
            
        # Apply threshold if specified
        if threshold is not None:
            result[result < threshold] = 0
            
        return result

    def find_peaks(self, image, min_distance=5, threshold_abs=None, threshold_rel=None):
        """Find local maxima in the image.
        
        Args:
            image: Input image
            min_distance: Minimum distance between peaks
            threshold_abs: Minimum absolute intensity of peaks
            threshold_rel: Minimum relative intensity of peaks
        """
        peaks = peak_local_max(
            image,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel
        )
        return peaks

    def refine_positions(self, image, peaks, diameter=9):
        """Refine peak positions using moment calculations.
        
        Args:
            image: Input image
            peaks: Initial peak positions
            diameter: Size of region to use for refinement
        """
        refined_positions = []
        masses = []
        sizes = []
        eccentricities = []
        
        radius = diameter // 2
        
        for peak in peaks:
            # Extract region around peak
            y, x = peak
            y_slice = slice(max(0, y-radius), min(image.shape[0], y+radius+1))
            x_slice = slice(max(0, x-radius), min(image.shape[1], x+radius+1))
            region = image[y_slice, x_slice]
            
            # Calculate mass (total intensity)
            mass = np.sum(region)
            
            # Calculate center of mass
            y_coords, x_coords = np.mgrid[y_slice, x_slice]
            y_cm = np.sum(region * y_coords) / mass if mass > 0 else y
            x_cm = np.sum(region * x_coords) / mass if mass > 0 else x
            
            # Calculate size (radius of gyration)
            dy = y_coords - y_cm
            dx = x_coords - x_cm
            r2 = dy*dy + dx*dx
            size = np.sqrt(np.sum(region * r2) / mass) if mass > 0 else radius
            
            # Calculate eccentricity using moment of inertia
            Ixx = np.sum(region * dx*dx) / mass if mass > 0 else 0
            Iyy = np.sum(region * dy*dy) / mass if mass > 0 else 0
            Ixy = np.sum(region * dx*dy) / mass if mass > 0 else 0
            
            # Eigenvalues of the inertia tensor
            trace = Ixx + Iyy
            det = Ixx*Iyy - Ixy*Ixy
            discriminant = np.sqrt(trace*trace - 4*det)
            lambda1 = (trace + discriminant) / 2
            lambda2 = (trace - discriminant) / 2
            
            # Eccentricity
            eccentricity = np.sqrt(1 - lambda2/lambda1) if lambda1 > 0 else 0
            
            refined_positions.append([y_cm, x_cm])
            masses.append(mass)
            sizes.append(size)
            eccentricities.append(eccentricity)
        
        return np.array(refined_positions), masses, sizes, eccentricities

    def visualize_step(self, image, peaks=None, refined_positions=None, 
                      save_name='step_results.png', title='Step Results'):
        """Visualize results of each step."""
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        
        if peaks is not None:
            plt.plot(peaks[:, 1], peaks[:, 0], 'r+', markersize=10, 
                    label='Initial Peaks')
            
        if refined_positions is not None:
            refined_positions = np.array(refined_positions)
            plt.plot(refined_positions[:, 1], refined_positions[:, 0], 'go', 
                    fillstyle='none', markersize=10, label='Refined Positions')
            
        plt.title(title)
        if peaks is not None or refined_positions is not None:
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(str(self.viz_path / save_name))
        plt.close()

def test_all_steps():
    # Create test instance
    test = LocateStepsTest()
    
    # Generate test image
    image, true_positions = test.generate_test_image()
    
    # Test and visualize each step
    
    # 1. Original image
    test.visualize_step(image, save_name='1_original.png', 
                       title='Original Image')
    
    # 2. After bandpass
    filtered = test.bandpass(image, sigma_low=1, sigma_high=10)
    test.visualize_step(filtered, save_name='2_bandpass.png', 
                       title='After Bandpass Filter')
    
    # 3. After peak finding
    peaks = test.find_peaks(filtered, min_distance=5, threshold_rel=0.2)
    test.visualize_step(filtered, peaks=peaks, save_name='3_peaks.png',
                       title='Located Peaks')
    
    # 4. After refinement
    refined_positions, masses, sizes, eccentricities = test.refine_positions(
        filtered, peaks)
    test.visualize_step(filtered, peaks=peaks, refined_positions=refined_positions,
                       save_name='4_refined.png', title='Refined Positions')
    
    # Save numerical results
    results = pd.DataFrame({
        'y': refined_positions[:, 0],
        'x': refined_positions[:, 1],
        'mass': masses,
        'size': sizes,
        'eccentricity': eccentricities
    })
    results.to_csv(test.results_path / 'particle_properties.csv', index=False)

if __name__ == "__main__":
    test_all_steps() 