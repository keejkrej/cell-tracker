import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import tifffile
from cellpose import models
from cellpose.core import use_gpu


class CellposeSegmenter:
    """
    A class for segmenting cells in microscopy images using Cellpose models.
    
    This class handles multi-channel timelapse microscopy data and applies
    Cellpose segmentation to identify individual cells.
    """
    
    def __init__(
        self,
        model_type: str = 'cyto2',
        gpu: bool = True,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0
    ):
        """
        Initialize the CellposeSegmenter.
        
        Parameters
        ----------
        model_type : str
            Type of Cellpose model to use ('cyto', 'cyto2', 'nuclei', etc.)
        gpu : bool
            Whether to use GPU for computation
        diameter : float, optional
            Expected cell diameter in pixels. If None, will be estimated
        flow_threshold : float
            Flow threshold parameter for Cellpose
        cellprob_threshold : float
            Cell probability threshold for Cellpose
        """
        self.gpu = use_gpu() if gpu else False
        self.model = models.CellposeModel(model_type=model_type, gpu=self.gpu)
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        
    def segment_image(
        self,
        image: np.ndarray,
        channels: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment a single image using Cellpose.
        
        Parameters
        ----------
        image : np.ndarray
            Input image with shape (height, width) or (height, width, channels)
        channels : list of int, optional
            Channel configuration [cyto_channel, nuclear_channel]
            Default is [0, 0] for grayscale or [1, 3] for RGB with nuclei
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'masks': Segmentation masks (2D array)
            - 'flows': Flow fields from Cellpose
            - 'styles': Style vectors
            - 'diams': Estimated diameters
        """
        if channels is None:
            if image.ndim == 2:
                channels = [0, 0]
            else:
                channels = [2, 1]  # membrane channel for cyto, nuclear channel
        
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=self.diameter,
            channels=channels,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold
        )
        
        return {
            'masks': masks,
            'flows': flows,
            'styles': styles,
            'diams': diams
        }
    
    def segment_timelapse(
        self,
        timelapse_path: str,
        channels: Optional[List[int]] = None,
        frames: Optional[Union[int, List[int]]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Segment a timelapse microscopy file.
        
        Parameters
        ----------
        timelapse_path : str
            Path to the timelapse TIFF file
        channels : list of int, optional
            Channel configuration for Cellpose
        frames : int or list of int, optional
            Specific frames to process. If None, processes all frames
            
        Returns
        -------
        list of dict
            List of segmentation results for each frame
        """
        with tifffile.TiffFile(timelapse_path) as tif:
            data = tif.asarray()
        
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D data, got shape {data.shape}")
        
        n_frames = data.shape[0]
        
        if frames is None:
            frames = list(range(n_frames))
        elif isinstance(frames, int):
            frames = [frames]
            
        results = []
        for frame_idx in frames:
            frame_data = data[frame_idx]
            
            if frame_data.ndim == 3 and frame_data.shape[0] <= 3:
                frame_data = np.transpose(frame_data, (1, 2, 0))
            
            result = self.segment_image(frame_data, channels=channels)
            result['frame'] = frame_idx
            results.append(result)
            
        return results
    
    def save_segmentation(
        self,
        segmentation_result: Dict[str, np.ndarray],
        output_path: str
    ):
        """
        Save segmentation results to a numpy file.
        
        Parameters
        ----------
        segmentation_result : dict
            Segmentation result from segment_image
        output_path : str
            Path to save the .npy file
        """
        np.save(output_path, segmentation_result)
    
    def load_segmentation(self, path: str) -> Dict[str, np.ndarray]:
        """
        Load segmentation results from a numpy file.
        
        Parameters
        ----------
        path : str
            Path to the .npy file
            
        Returns
        -------
        dict
            Segmentation data
        """
        return np.load(path, allow_pickle=True).item()