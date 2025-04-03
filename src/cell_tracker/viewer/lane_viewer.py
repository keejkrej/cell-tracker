from .base_viewer import BaseViewer
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from .. import lane_detection

class LaneViewer(BaseViewer):
    """Viewer for lane detection and visualization."""
    
    def __init__(self, nd2_file, df=None):
        super().__init__()
        self.v = 0
        f = ND2Reader(nd2_file)
        
        self.lanes = np.array([f.get_frame_2D(v=v) for v in range(f.sizes['v'])])
        
        # Create lane detection parameters
        self.ld = widgets.IntSlider(
            min=10, max=60, step=1, 
            description="lane_d", value=30, 
            continuous_update=True
        )
        
        # Create standard sliders
        sliders = self.create_sliders(f, include_time=False, include_channel=False)
        self.v = sliders['v']
        
        self.clip = self.create_clip_slider(max_val=5000, value=[0, 5000])
        
        self.threshold = widgets.FloatSlider(
            min=0, max=1, step=0.05,
            description="threshold", value=0.3,
            continuous_update=False
        )
        
        self.kernel_width = 5
        
        # Create recompute button
        self.button = widgets.Button(
            description='Recompute',
            disabled=False,
            button_style='',
            tooltip='Click me',
            icon='check'
        )
        self.button.on_click(self.recompute_v)
        
        # Initialize figure and image
        self.setup_figure()
        self.im = self.ax.imshow(self.lanes[0], cmap='gray', vmin=0, vmax=5000)
        
        self.min_coordinates = list(range(f.sizes['v']))
        self.max_coordinates = list(range(f.sizes['v']))

        self.recompute_v(self.v.value)
        self.ax.plot([], [], color='red')
        
        # Create interactive output
        out = widgets.interactive_output(
            self.update, 
            {'v': self.v, 'clip': self.clip}
        )

        # Organize layout
        box = widgets.VBox([
            out, 
            widgets.VBox([
                self.v, self.clip, self.ld, 
                self.threshold, self.button
            ], layout=widgets.Layout(width='400px'))
        ])

        display(box)
        
    def update(self, v, clip):
        vmin, vmax = self.clip.value
        image = self.lanes[self.v.value]
        self.im.set_data(image)
        self.im.set_clim([vmin, vmax])
        
        if isinstance(self.min_coordinates[v], Iterable):
            [self.ax.axes.lines[0].remove() for j in range(len(self.ax.axes.lines))]
            x = [0, image.shape[1]-1]
            for i in range(self.min_coordinates[v].shape[0]):
                self.ax.plot(x, [self.min_coordinates[v][i,1], sum(self.min_coordinates[v][i])], color='red')
            
            for i in range(self.max_coordinates[v].shape[0]):
                self.ax.plot(x, [self.max_coordinates[v][i,1], sum(self.max_coordinates[v][i])], color='red')
    
    def recompute_v(self, v):
        v = self.v.value
        vmin, vmax = self.clip.value
        lanes_clipped = np.clip(self.lanes[v], vmin, vmax, dtype=self.lanes.dtype)
        
        print('recomputing')
        self.min_coordinates[v], self.max_coordinates[v] = lane_detection.get_lane_mask(
            lanes_clipped, 
            kernel_width=self.kernel_width, 
            line_distance=self.ld.value, 
            debug=True, 
            gpu=True, 
            threshold=self.threshold.value
        )
        print('updating')
        self.update(v, self.clip) 