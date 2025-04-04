from .base_viewer import BaseViewer
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from cellpose import models
import os
import json
from skimage.morphology import binary_erosion
from skimage.segmentation import find_boundaries
from ..core import functions

class CellposeViewer(BaseViewer):
    """Viewer for Cellpose-based cell segmentation and tracking."""
    
    def __init__(self, nd2file, bf_channel=None, nuc_channel=None, pretrained_model='mdamb231', omni=False):
        super().__init__()
        self.link_dfs = {}
        
        self.f = ND2Reader(nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
        # Determine channels
        channels = self.f.metadata['channels']
        if bf_channel is None or nuc_channel is None:
            if 'erry' in channels[0] or 'exas' in channels[0] and not 'phc' in channels[0]:
                self.nucleus_channel = 0
                self.cyto_channel = 1
            elif 'erry' in channels[1] or 'exas' in channels[1] and not 'phc' in channels[1]:
                self.nucleus_channel = 1
                self.cyto_channel = 0
            else:
                raise ValueError(
                    f"""The channels could not be automatically detected! 
                    The following channels are available: {channels}. 
                    Please specify the indices of bf_channel and nuc_channel as keyword arguments. 
                    i.e: bf_channel=0, nuc_channel=1"""
                )
        else:
            self.cyto_channel = bf_channel
            self.nucleus_channel = nuc_channel

        # Create sliders
        sliders = self.create_sliders(self.f, include_channel=False)
        self.t = sliders['t']
        self.v = sliders['v']
        
        self.nclip = widgets.FloatRangeSlider(
            min=0, max=2**16, step=1, 
            value=[0, 1000], 
            description="clip nuclei", 
            continuous_update=False, 
            width='200px'
        )
        
        self.cclip = widgets.FloatRangeSlider(
            min=0, max=2**16, step=1, 
            value=[50, 8000], 
            description="clip cyto", 
            continuous_update=False, 
            width='200px'
        )
        
        self.flow_threshold = widgets.FloatSlider(
            min=0, max=1.5, step=0.05, 
            description="flow_threshold", 
            value=1.25, 
            continuous_update=False
        )
        
        self.diameter = widgets.IntSlider(
            min=0, max=1000, step=2, 
            description="diameter", 
            value=29, 
            continuous_update=False
        )
        
        self.mask_threshold = widgets.FloatSlider(
            min=-3, max=3, step=0.1, 
            description="mask_threshold", 
            value=0, 
            continuous_update=False
        )
        
        self.max_travel = widgets.IntSlider(
            min=3, max=50, step=1, 
            description="max_travel", 
            value=5, 
            continuous_update=False
        )
        
        self.track_memory = widgets.IntSlider(
            min=0, max=20, step=1, 
            description="track memory", 
            value=5, 
            continuous_update=False
        )

        # Create track button
        self.tp_method = widgets.Button(
            description='Track',
            disabled=False,
            button_style='',
            tooltip='Click me',
            icon=''
        )
        
        # Initialize Cellpose model
        self.init_cellpose(pretrained_model=pretrained_model, omni=omni)
        
        # Initialize figure and image
        vmin, vmax = self.cclip.value
        cyto = (255*(np.clip(self.f.get_frame_2D(v=0, c=self.cyto_channel, t=0), vmin, vmax)/vmax)).astype('uint8')
        vmin, vmax = self.nclip.value
        nucleus = (255*(np.clip(self.f.get_frame_2D(v=0, c=self.nucleus_channel, t=0), vmin, vmax)/vmax)).astype('uint8')
        red = np.zeros_like(nucleus)
        
        image = np.stack((red, red, nucleus), axis=-1).astype('float32')
        image += (cyto[:,:,np.newaxis]/3)
        image = np.clip(image, 0, 255).astype('uint8')
        
        self.setup_figure()
        self.im = self.ax.imshow(image)
        
        # Create interactive output
        out = widgets.interactive_output(
            self.update, 
            {
                't': self.t, 'v': self.v, 
                'cclip': self.cclip, 'nclip': self.nclip,
                'flow_threshold': self.flow_threshold,
                'diameter': self.diameter,
                'mask_threshold': self.mask_threshold,
                'max_travel': self.max_travel
            }
        )
        
        # Organize layout
        box = widgets.VBox([
            self.t, self.v, self.cclip, self.nclip,
            self.flow_threshold, self.diameter,
            self.mask_threshold, self.tp_method
        ])
        
        box1 = widgets.VBox([out, box])
        grid = widgets.widgets.GridspecLayout(3, 3)
        
        grid[:, :2] = self.fig.canvas
        grid[1:,2] = box
        grid[0, 2] = out
        
        display(grid)
        plt.ion()
    
    def init_cellpose(self, pretrained_model='mdamb231', omni=False, model='cyto', gpu=True):
        """Initialize the Cellpose model with specified parameters."""
        if omni:
            from cellpose_omni.models import CellposeModel
            self.model = CellposeModel(
                gpu=gpu, omni=True, nclasses=4, nchan=2,
                pretrained_model=pretrained_model
            )
            return
        
        elif pretrained_model is None:
            self.model = models.Cellpose(gpu=gpu, model_type='cyto')
        else:
            path_to_models = os.path.join(os.path.dirname(__file__), '../models')
            with open(os.path.join(path_to_models, 'models.json'), 'r') as f:
                dic = json.load(f)
            if pretrained_model in dic.keys():
                path_to_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
                if os.path.isfile(path_to_model):
                    pretrained_model = path_to_model
                else:
                    url = dic[pretrained_model]['link']
                    print('Downloading model from Nextcloud...')
                    request.urlretrieve(url, os.path.join(path_to_models, path_to_model))
                    pretrained_model = os.path.join(path_to_models, dic[pretrained_model]['path'])
            
            if not omni:
                self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    
    def update(self, t, v, cclip, nclip, flow_threshold, diameter, mask_threshold, max_travel):
        """Update the visualization based on current parameters."""
        self.segment(t, v, cclip, nclip, flow_threshold, mask_threshold, diameter)
        self.im.set_data(self.image)
    
    def segment(self, t, v, cclip, nclip, flow_threshold, mask_threshold, diameter, normalize=True, verbose=False):
        """Perform cell segmentation using Cellpose."""
        nucleus = self.f.get_frame_2D(v=v, t=t, c=self.nucleus_channel)
        cyto = self.f.get_frame_2D(v=v, t=t, c=self.cyto_channel)
        
        image = np.stack((cyto, nucleus), axis=1)
        if diameter == 0:
            diameter = None
            
        mask = self.model.eval(
            image, diameter=diameter, channels=[1,0],
            flow_threshold=flow_threshold,
            cellprob_threshold=mask_threshold,
            normalize=normalize,
            progress=verbose
        )[0].astype('uint8')
        
        bin_mask = np.zeros(mask.shape, dtype='bool')
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids != 0]
        
        for cell_id in cell_ids:
            bin_mask += binary_erosion(mask == cell_id)
        
        outlines = find_boundaries(bin_mask, mode='outer')
        try:
            print(f'{cell_ids.max()} Masks detected')
        except ValueError:
            print('No masks detected')
            
        self.outlines = outlines
        self.mask = mask
        self.image = self.get_8bit(outlines, cyto)
    
    def get_8bit(self, outlines, cyto, nuclei=None):
        """Convert image to 8-bit format for display."""
        vmin, vmax = self.cclip.value
        cyto = np.clip(cyto, vmin, vmax)
        cyto = (255*(cyto-vmin)/(vmax-vmin)).astype('uint8')
        
        image = np.stack((cyto, cyto, cyto), axis=-1)
        image[(outlines > 0)] = [255, 0, 0]
        
        return image 