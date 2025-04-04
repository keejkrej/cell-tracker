from .base_viewer import BaseViewer
import ipywidgets as widgets
from IPython.display import display
import numpy as np

class BasicViewer(BaseViewer):
    """Basic viewer for image and tracking visualization."""
    
    def __init__(self, file, pattern_file, experiment_id=5, df=None):
        super().__init__()
        self.experiment_id = experiment_id
        self.t, self.v, self.c = [0, 0, 0]
        self.load_df(self.experiment_id, self.v, df=df)
        f = ND2Reader(file)

        # Create sliders
        sliders = self.create_sliders(f)
        t = sliders.get('t')
        c = sliders.get('c')
        v = sliders.get('v')
        
        clip = self.create_clip_slider()
        
        show_nuclei = widgets.Checkbox(
            value=False,
            description='Show trackpy',
            disabled=False,
            indent=False
        )

        # Initialize image
        im_0 = f.get_frame_2D(t=t.value, c=c.value, v=v.value)
        self.setup_figure()
        self.im = self.ax.imshow(im_0, cmap='gray')
        scat = self.ax.scatter([0,0], [0,0])

        def update(t, c, v, clip, show_nuclei):
            vmin, vmax = clip
            image = f.get_frame_2D(v=v, c=c, t=t)
            self.im.set_data(image)
            self.im.set_clim([vmin, vmax])

            if show_nuclei:
                if v != self.v:
                    self.load_df(experiment_id=self.experiment_id, fov=v, df=df)
                self.show_tracking(self.df, scat, t)
            else:
                scat.set_offsets([[0,0], [0,0]])
                
            self.t, self.v, self.c = [t, v, c]

        # Create interactive output
        out = widgets.interactive_output(
            update, 
            {'t': t, 'c': c, 'v': v, 'clip': clip, 'show_nuclei': show_nuclei}
        )

        # Organize layout
        box = widgets.VBox([
            out, 
            widgets.VBox([t, c, v, clip, show_nuclei], layout=widgets.Layout(width='400px'))
        ])

        display(box)
        self.t, self.v = [t.value, v.value] 