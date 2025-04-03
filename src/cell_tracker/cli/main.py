import os
import time
from nd2reader import ND2Reader
from ..core.pipeline import run_pipeline
from ..core.database import Database
from ..viewer import LaneViewer, TpViewer, CellposeViewer
import matplotlib.pyplot as plt
import numpy as np

data_path = './'
nd2_file = 'timelapse.nd2'
lanes_file = 'pattern.nd2'
path_out = './extraction/'
db_path = '/project/ag-moonraedler/MAtienza/database/onedcellmigration.db'

laneviewer = LaneViewer(data_path+lanes_file)
lane_distance = laneviewer.ld.value
lane_low_clip, lane_high_clip = laneviewer.clip.value

tpviewer = TpViewer(data_path+nd2_file)
min_mass, max_travel, track_memory, diameter, min_frames = tpviewer.min_mass.value, tpviewer.max_travel.value, tpviewer.track_memory.value, tpviewer.diameter.value, tpviewer.min_frames.value

cellposeviewer = CellposeViewer(data_path+nd2_file)
cyto_diameter=cellposeviewer.diameter.value
flow_threshold=cellposeviewer.flow_threshold.value
mask_threshold=cellposeviewer.mask_threshold.value

normalize=True
f = ND2Reader(os.path.join(data_path, nd2_file))
metadata = f.metadata
Experiment_data = {
    'Experiment_id':1,
    'Path':data_path,
    'Date': metadata['date'].strftime('%d-%m-%Y %H:%m'),
    'celltype': 'MDA-MB-231',
    'microscope': 'UNikon',
    'nframes': f.sizes['t'],
    'nfov': f.sizes['v'],
    'channels': str(metadata['channels']),
    'fpm': f.sizes['t']/(1e-3*(f.metadata['events'][-1]['time'] - f.metadata['events'][0]['time'])/60),
    'pixels/um': 1.538,
    'bitsperpixel': 16,
    'Author': 'Miguel Atienza'}
1/Experiment_data['fpm']
db = Database(db_path)
experiment_id=6
print(experiment_id)
nfov = f.sizes['v']
start_time = time.time()
fovs = list(range(0, nfov))
sql=False
run_pipeline(data_path, nd2_file, lanes_file, path_out, frame_indices, manual=False, fovs=None, sql=False, lane_distance=30,
lane_low_clip=0, lane_high_clip=2000, min_mass=2.65e5, max_travel=15, track_memory=15, diameter=15, min_frames=10, cyto_diameter=29, 
flow_threshold=1.25, mask_threshold=0, pretrained_model='mdamb231', use_existing_parameters=False)