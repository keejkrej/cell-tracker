import ipywidgets as widgets
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
import pandas as pd
import numpy as np
from skimage.segmentation import find_boundaries
from tqdm import tqdm
from collections.abc import Iterable
import trackpy as tp
import os
from cellpose import models
from cellpose.io import logger_setup 
from skimage.morphology import binary_erosion
from skimage.io import imread
import matplotlib.collections as collections

class BaseViewer:
    """Base class for all viewers with common functionality."""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.im = None
        
    def create_sliders(self, f, include_time=True, include_channel=True, include_fov=True):
        """Create standard sliders for time, channel, and FOV."""
        sliders = {}
        
        if include_time and f.sizes['t'] > 0:
            t_max = f.sizes['t'] - 1
            sliders['t'] = widgets.IntSlider(min=0, max=t_max, step=1, description="t", continuous_update=True)
        
        if include_channel and f.sizes['c'] > 0:
            c_max = f.sizes['c'] - 1
            sliders['c'] = widgets.IntSlider(min=0, max=c_max, step=1, description="c", continuous_update=True)
        
        if include_fov and f.sizes['v'] > 0:
            v_max = f.sizes['v'] - 1
            sliders['v'] = widgets.IntSlider(min=0, max=v_max, step=1, description="v", continuous_update=False)
            
        return sliders
    
    def create_clip_slider(self, min_val=0, max_val=2**16-1, value=[0, 30000]):
        """Create a contrast adjustment slider."""
        return widgets.IntRangeSlider(
            min=min_val,
            max=max_val,
            step=1,
            value=value,
            description="clip",
            continuous_update=True,
            width='200px'
        )
    
    def setup_figure(self, figsize=(10, 8)):
        """Setup matplotlib figure with standard configuration."""
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.tight_layout()
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = True
        plt.ion()
        
    def show_tracking(self, df, scat, t):
        """Display tracking data on the plot."""
        dft = df[df.frame == t]
        data = np.hstack((dft.x.values[:, np.newaxis], dft.y.values[:, np.newaxis]))
        scat.set_offsets(data)
        
    def load_df(self, experiment_id=None, fov=0, df=None, path_to_db=None):
        """Load tracking data from database or DataFrame."""
        if df is not None:
            self.df = df
        elif path_to_db is not None:
            conn = sqlite3.connect(path_to_db)
            query = f'''Select * From Raw_tracks where Lane_id in (
                SELECT Lane_id FROM Lanes WHERE (Experiment_id={experiment_id} and fov={fov}))'''
            self.df = pd.read_sql(query, conn)
        else:
            self.df = pd.DataFrame() 