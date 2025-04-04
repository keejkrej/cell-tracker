from .base_viewer import BaseViewer
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import trackpy as tp
from ..core import functions

class TrackpyViewer(BaseViewer):
    """Viewer for trackpy-based particle tracking visualization."""
    
    def __init__(self, nd2file, channel=0):
        super().__init__()
        self.link_dfs = {}
        self.channel = channel
        
        self.f = ND2Reader(nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
        # Create standard sliders
        sliders = self.create_sliders(self.f)
        self.t = sliders['t']
        self.c = sliders['c']
        self.v = sliders['v']
        
        self.clip = self.create_clip_slider()
        
        # Create trackpy parameters
        self.min_mass = widgets.FloatSlider(
            min=1e5, max=1e6, step=0.01e5,
            description="min_mass", value=2.65e5,
            continuous_update=True
        )
        
        self.diameter = widgets.IntSlider(
            min=9, max=35, step=2,
            description="diameter", value=15,
            continuous_update=True
        )
        
        self.min_frames = widgets.FloatSlider(
            min=0, max=50, step=1,
            description="min_frames", value=10,
            continuous_update=False
        )
        
        self.max_travel = widgets.IntSlider(
            min=3, max=50, step=1,
            description="max_travel", value=15,
            continuous_update=False
        )
        
        self.track_memory = widgets.IntSlider(
            min=0, max=20, step=1,
            description="track memory", value=5,
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
        self.tp_method.on_click(self.link_update)
        
        # Initialize figure and image
        vmin, vmax = self.clip.value
        image = self.f.get_frame_2D(
            v=self.v.value,
            c=self.c.value,
            t=self.t.value
        )
        
        self.setup_figure()
        self.im = self.ax.imshow(image, cmap='gray')
        self.bscat = self.ax.scatter(
            [0,0], [0,0], 
            s=0.2*plt.rcParams['lines.markersize'] ** 2, 
            color='blue', 
            alpha=0.5
        )
        self.lscat = self.ax.scatter(
            [0,0], [0,0], 
            s=0.2*plt.rcParams['lines.markersize'] ** 2, 
            color='red', 
            alpha=0.5
        )
        
        # Create interactive output
        out = widgets.interactive_output(
            self.update, 
            {
                't': self.t, 'c': self.c, 'v': self.v, 
                'clip': self.clip, 'min_mass': self.min_mass,
                'diameter': self.diameter, 'min_frames': self.min_frames,
                'max_travel': self.max_travel
            }
        )
        
        # Organize layout
        box = widgets.VBox([
            self.t, self.c, self.v, self.clip, 
            self.min_mass, self.diameter, self.min_frames,
            self.max_travel, self.track_memory, self.tp_method
        ])
        
        box1 = widgets.VBox([out, box])
        grid = widgets.widgets.GridspecLayout(3, 3)
        
        grid[:, :2] = self.fig.canvas
        grid[1:,2] = box
        grid[0, 2] = out
        
        display(grid)
        plt.ion()
        
    def update(self, t, c, v, clip, min_mass, diameter, min_frames, max_travel):
        vmin, vmax = clip
        image = self.f.get_frame_2D(v=v, c=c, t=t)
        self.im.set_data(image)
        self.im.set_clim([vmin, vmax])
        
        self.batch_update(v, t, min_mass, diameter)
        self.show_tracking(self.batch_df, self.bscat)
        
        if v in self.link_dfs.keys():
            df = self.link_dfs[v][self.link_df.frame==t]
            self.show_tracking(df, self.lscat)
        else:
            self.lscat.set_offsets([[0,0]])
    
    def batch_update(self, v, t, min_mass, diameter):
        nuclei = self.f.get_frame_2D(v=v, t=t, c=self.channel)
        nuclei = functions.preprocess(
            nuclei, 
            bottom_percentile=0.05, 
            top_percentile=99.95, 
            log=True, 
            return_type='uint16'
        )
        self.batch_df = tp.locate(nuclei, diameter=diameter, minmass=min_mass)
        
    def link_update(self, a):
        v, min_mass, max_travel, track_memory, diameter, min_frames = (
            self.v.value, self.min_mass.value, self.max_travel.value,
            self.track_memory.value, self.diameter.value, self.min_frames.value
        )
        
        if v not in self.link_dfs.keys():
            nuclei = np.array([
                self.f.get_frame_2D(v=v, t=t, c=self.channel) 
                for t in range(self.f.sizes['t'])
            ])
            nuclei = functions.preprocess(
                nuclei, 
                bottom_percentile=0.05, 
                top_percentile=99.95, 
                log=True, 
                return_type='uint16'
            )
            dft = tp.batch(nuclei, diameter=diameter, minmass=min_mass)
            dftp = tp.link(dft, max_travel, memory=track_memory)
            self.link_df = tp.filter_stubs(dftp, min_frames)
            self.link_dfs[v] = self.link_df
        
        self.update(
            self.t.value, self.c.value, self.v.value, 
            self.clip.value, self.min_mass.value, 
            self.diameter.value, self.min_frames.value, 
            self.max_travel.value
        ) 