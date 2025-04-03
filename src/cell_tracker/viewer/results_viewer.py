from .base_viewer import BaseViewer
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import pandas as pd
import sqlite3
import os
from skimage.morphology import binary_erosion
from skimage.segmentation import find_boundaries
from skimage.io import imread
import matplotlib.collections as collections
from ..core import functions
from ..classify import cp

class ResultsViewer(BaseViewer):
    """Viewer for visualizing and analyzing tracking results."""
    
    def __init__(self, nd2file, outpath, base_path=None, experiment_paths=[], 
                 db_path='/project/ag-moonraedler/MAtienza/database/onedcellmigration.db', 
                 path_to_patterns=None):
        super().__init__()
        self.link_dfs = {}
        self.base_path = base_path
        self.cyto_locator = None
        self.path_to_patterns = path_to_patterns
        self.nd2file = nd2file
        self.f = ND2Reader(nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        self.outpath = outpath
        self.db_path = db_path
        
        # Create sliders
        sliders = self.create_sliders(self.f)
        self.t = sliders['t']
        self.c = sliders['c']
        self.v = sliders['v']
        
        self.clip = widgets.IntRangeSlider(
            min=0, max=int(2**16 - 1), step=1,
            value=[0, 12000], description="clip",
            continuous_update=True, width='200px'
        )
        
        # Create checkboxes
        self.view_nuclei = widgets.Checkbox(
            value=True,
            description='Nuclei',
            disabled=False,
            button_style='',
            tooltip='Click me',
            icon=''
        )
        
        self.view_cellpose = widgets.Checkbox(
            value=True,
            description='Contours',
            disabled=False,
            button_style='',
            tooltip='Click me',
            icon=''
        )
        
        # Initialize image
        vmin, vmax = self.clip.value
        cyto = self.f.get_frame_2D(v=self.v.value, c=self.c.value, t=self.t.value)
        cyto = np.clip(cyto, vmin, vmax).astype('float32')
        cyto = (255*(cyto-vmin)/(vmax-vmin)).astype('uint8')
        image = np.stack((cyto, cyto, cyto), axis=-1)
        
        # Load initial data
        self.load_masks(outpath, self.v.value)
        self.load_df(self.db_path, self.v.value)
        self.oldv = 0
        self.update_lanes()
        
        # Create figures
        self.setup_figure()
        self.im = self.ax.imshow(image, cmap='gray')
        
        self.bscat = self.ax.scatter(
            [0,0], [0,0],
            s=0.4*plt.rcParams['lines.markersize'] ** 2,
            color='blue', alpha=0.5
        )
        self.lscat = self.ax.scatter(
            [0,0], [0,0],
            s=0.2*plt.rcParams['lines.markersize'] ** 2,
            color='red', alpha=0.5
        )
        
        # Create second figure for time series
        self.fig2, self.ax2 = plt.subplots(constrained_layout=True, figsize=(8,6))
        self.ax2.plot(np.arange(100), np.ones(100), color='blue')
        self.tmarker = self.ax2.axvline(self.t.value, color='black', lw=1)
        self.ax2.margins(x=0)
        
        # Connect event handlers
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.fig2.canvas.mpl_connect('button_press_event', self.onclick_plot)
        
        # Create experiment selector
        self.file_menu = widgets.Dropdown(options=experiment_paths)
        self.file_menu.observe(self.update_experiment, 'value')
        
        # Organize layout
        buttons = [self.t, self.c, self.v, self.clip]
        for button in buttons:
            button.observe(self.update, 'value')
            
        self.buttons_box = widgets.VBox(buttons + [self.view_nuclei, self.view_cellpose])
        self.left_box = widgets.VBox([self.buttons_box, self.file_menu])
        
        # Create output widgets for figures
        fig1 = widgets.Output()
        with plt.ioff():
            with fig1:
                display(self.fig.canvas)
        
        fig2 = widgets.Output()
        with plt.ioff():
            with fig2:
                display(self.fig2.canvas)
        
        self.grid = widgets.HBox([self.left_box, fig1, fig2])
        self.update(None)
        display(self.grid)
        plt.ion()
    
    def update(self, change, t=None, c=None, v=None, clip=None):
        """Update the visualization based on current parameters."""
        vmin, vmax = self.clip.value
        clip = self.clip.value
        t = self.t.value
        c = self.c.value
        v = self.v.value
        
        if v != self.oldv:
            self.cyto_locator = None
            self.update_lanes()
        
        if self.view_nuclei.value:
            if v != self.oldv:
                self.load_df(self.db_path, v)
            self.update_tracks()
            
        if self.view_cellpose.value:
            if v != self.oldv:
                self.load_masks(self.outpath, v)
        
        self.update_image(t, v, clip)
        self.im.set_data(self.image)
        self.tmarker.set_xdata(t)
        self.oldv = v
    
    def update_tracks(self):
        """Update the tracking visualization."""
        t, v = self.t.value, self.v.value
        scat = self.bscat
        df = self.clean_df[self.clean_df.frame == self.t.value]
        
        data = np.hstack((df.x.values[:,np.newaxis], df.y.values[:, np.newaxis]))
        scat.set_offsets(data)
        
        scat = self.lscat
        if 'segment' in df.columns:
            df = df[(df.segment > 0)]
        
        data = np.hstack((df.x.values[:,np.newaxis], df.y.values[:, np.newaxis]))
        scat.set_offsets(data)
    
    def update_lanes(self):
        """Update the lane visualization."""
        fov = self.v.value
        path_to_lane = os.path.join(self.outpath, f'XY{fov}/lanes/lanes_mask.tif')
        self.lanes = imread(path_to_lane) > 0
    
    def update_image(self, t, v, clip):
        """Update the main image display."""
        vmin, vmax = clip
        cyto = self.f.get_frame_2D(v=self.v.value, c=self.c.value, t=self.t.value)
        cyto = np.clip(cyto, vmin, vmax).astype('float32')
        cyto = (255*(cyto-vmin)/(vmax-vmin)).astype('uint8')
        image = np.stack((cyto, cyto, cyto), axis=-1)
        image[:,:,0] = np.clip((self.lanes*10).astype('uint16') + image[:,:,0].astype('uint16'), 0, 255).astype('uint8')
        
        if self.view_cellpose.value:
            mask = self.masks[t]
            bin_mask = np.zeros(mask.shape, dtype='bool')
            cell_ids = np.unique(mask)
            cell_ids = cell_ids[cell_ids != 0]
            
            for cell_id in cell_ids:
                bin_mask += binary_erosion(mask == cell_id)
            
            outlines = find_boundaries(bin_mask, mode='outer')
            self.outlines = outlines
            image[(outlines > 0)] = [255, 0, 0]
            
            if self.cyto_locator is not None:
                mask_id = self.cyto_locator[t]
                if mask_id != 0:
                    g_outline = find_boundaries(mask == mask_id)
                    image[g_outline] = [0, 255, 0]
        
        self.image = image
    
    def load_df(self, db_path, fov, from_db=False):
        """Load tracking data from database or CSV file."""
        if from_db:
            conn = sqlite3.connect(db_path)
            experiment_id = 5
            
            self.experiment_df = pd.read_sql("Select * from Experiments", conn)
            self.lanes_df = pd.read_sql(
                f"""Select * from Lanes 
                Where (Experiment_id={experiment_id} and fov={fov})""",
                conn
            )
            self.df = pd.read_sql(
                f"""Select * from Raw_tracks 
                Where Lane_id in 
                (Select Lane_id FROM Lanes WHERE
                (Experiment_id={experiment_id} and fov={fov}))""",
                conn
            )
            
            self.metadata = self.experiment_df[self.experiment_df.Experiment_id == experiment_id]
            self.pixelperum = self.metadata['pixels/um'].values
            self.fpm = self.metadata['fpm'].values
            self.lane_ids = np.unique(self.df.Lane_id.values)
            self.lane_ids.sort()
            
            conn.close()
        else:
            self.df = pd.read_csv(f'{self.outpath}/XY{fov}/tracking_data.csv')
            self.clean_df = pd.read_csv(f'{self.outpath}/XY{fov}/clean_tracking_data.csv')
    
    def load_masks(self, outpath, fov):
        """Load cell masks from MP4 file."""
        path_to_mask = os.path.join(outpath, f'XY{fov}/cyto_masks.mp4')
        self.masks = functions.mp4_to_np(path_to_mask)
    
    def onclick_plot(self, event):
        """Handle click events on the time series plot."""
        t = event.xdata
        self.t.value = t
        self.update(self.t.value, self.c.value, self.v.value, self.clip.value)
    
    def onclick(self, event):
        """Handle click events on the main image."""
        ix, iy = event.xdata, event.ydata
        self.coords = (ix, iy)
        
        mask_id = self.masks[self.t.value, np.round(iy).astype(int), np.round(ix).astype(int)]
        if mask_id == 0:
            return
        
        particle_id = self.df.loc[
            (self.df.frame == self.t.value) & 
            (self.df.cyto_locator == mask_id)
        ].particle.values
        
        if len(particle_id) < 1:
            return
        
        self.particle_id = particle_id[0]
        self.dfp = self.clean_df[self.clean_df.particle == self.particle_id]
        
        # Update time series plot
        self.ax2.clear()
        self.ax2.plot(self.dfp.frame, self.dfp.nucleus, color='blue')
        self.ax2.plot(self.dfp.frame, self.dfp.front, color='red')
        self.ax2.plot(self.dfp.frame, self.dfp.rear, color='red')
        self.ax2.margins(x=0)
        
        # Add time axis
        tres = 30
        def t_to_frame(t):
            return t/(tres/60)
        def frame_to_t(frame):
            return frame*tres/60
        
        tax = self.ax2.secondary_xaxis('top', functions=(frame_to_t, t_to_frame))
        tax.set_xlabel('Time in minutes')
        self.tmarker = self.ax2.axvline(self.t.value, color='black', lw=1)
        
        # Add segment visualization
        if 'segment' in self.dfp.columns:
            low, high = self.ax2.get_ylim()
            collection = collections.BrokenBarHCollection.span_where(
                self.dfp.frame.values, ymin=low, ymax=high,
                where=self.dfp.segment == 0,
                facecolor='gray', alpha=0.5
            )
            self.ax2.add_collection(collection)
        
        # Add state visualization
        if 'state' in self.dfp.columns:
            MO_bool = self.dfp.state == 'MO'
            MS_bool = self.dfp.state == 'MS'
            SO_bool = self.dfp.state == 'SO'
            SS_bool = self.dfp.state == 'SS'
            
            x_collection = self.dfp.frame.values
            low, high = self.ax2.get_ylim()
            
            MO_collection = collections.BrokenBarHCollection.span_where(
                x_collection, ymin=low, ymax=high,
                where=MO_bool, facecolor=[1,0,1], alpha=0.2
            )
            MS_collection = collections.BrokenBarHCollection.span_where(
                x_collection, ymin=low, ymax=high,
                where=MS_bool, facecolor=[1,0,0], alpha=0.2
            )
            SO_collection = collections.BrokenBarHCollection.span_where(
                x_collection, ymin=low, ymax=high,
                where=SO_bool, facecolor=[0,0,1], alpha=0.2
            )
            SS_collection = collections.BrokenBarHCollection.span_where(
                x_collection, ymin=low, ymax=high,
                where=SS_bool, facecolor=[0,1,0], alpha=0.2
            )
            
            self.ax2.add_collection(MO_collection)
            self.ax2.add_collection(MS_collection)
            self.ax2.add_collection(SO_collection)
            self.ax2.add_collection(SS_collection)
            
            o_change = np.concatenate(([0], np.diff(self.dfp.O.values) != 0))
            v_change = np.concatenate(([0], np.diff(self.dfp.V.values) != 0))
            
            cps = np.argwhere(o_change & v_change & (self.dfp.state.notnull().values))
            for cp_index in cps:
                self.ax2.axvline(cp_index, color='green')
        
        # Update cyto locator
        self.cyto_locator = np.zeros(self.masks.shape[0], dtype='uint8')
        self.cyto_locator[self.dfp.frame] = self.masks[
            self.dfp.frame,
            np.round(self.dfp.y).astype(int),
            np.round(self.dfp.x).astype(int)
        ]
        
        self.update_image(self.t.value, self.v.value, self.clip.value)
        self.im.set_data(self.image)
    
    def update_experiment(self, file_name):
        """Update the viewer for a new experiment."""
        file_name = file_name.new
        base_path = self.base_path
        path_to_meta_data = os.path.join(base_path, file_name, 'Experiment_data.csv')
        experiment_data = pd.read_csv(path_to_meta_data)
        
        self.nd2file = os.path.join(
            experiment_data.Path.values[0],
            experiment_data.time_lapse_file.values[0]
        )
        
        self.f = ND2Reader(self.nd2file)
        self.nfov, self.nframes = self.f.sizes['v'], self.f.sizes['t']
        
        self.outpath = os.path.join(base_path, file_name, 'extraction')
        
        # Update sliders
        self.t.max = self.f.sizes['t'] - 1
        self.v.max = self.f.sizes['v'] - 1
        
        self.load_masks(self.outpath, self.v.value)
        self.load_df(self.db_path, self.v.value)
        self.update(None) 