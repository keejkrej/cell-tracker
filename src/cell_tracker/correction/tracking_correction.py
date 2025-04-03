import os
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from tifffile import imread
from animation_player import Player
from matplotlib.widgets import Slider
from mpl_point_clicker import clicker

# Change figure size
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')


def change_one_position(X, Y, frame, particle, new_x, new_y):
    """Modify the position of one particle at one specific point."""
    p = int(particle)
    frame_ind = int(frame)
    X.at[frame_ind, p] = new_x
    Y.at[frame_ind, p] = new_y
    return X, Y


def exchange_particles(X, Y, frame, particle1, particle2):
    """Exchange two particles from a specific time point."""
    frame_ind = int(frame)

    keep_traj_X = copy.deepcopy(X.iloc[frame_ind:,particle1])
    X.iloc[frame_ind:,particle1] = X.iloc[frame_ind:,particle2]
    X.iloc[frame_ind:,particle2] = keep_traj_X

    keep_traj_Y = copy.deepcopy(Y.iloc[frame_ind:,particle1])
    Y.iloc[frame_ind:,particle1] = Y.iloc[frame_ind:,particle2]
    Y.iloc[frame_ind:,particle2] = keep_traj_Y

    return X, Y


def correct_position(frames, X, Y, cell_nb=3, frame_no=None):
    """Correct the position of a particle by asking imput to the user."""

    if frame_no == None:
        frame_no = input("Enter the number ot the frame that you want to correct: ")
        # Check that the frame number is valid
        if not frame_no.isdigit() or int(frame_no)<0 or int(frame_no)>=len(X):
            print("Warning: This is not a valid number.")
            return X, Y

    i = int(frame_no)
    fig_frame, axes_frame, _, _, _ = plot_one_frame(frames, X, Y, i, cell_nb=cell_nb, legend=False)

    klicker = clicker(axes_frame, ['Cell 1', 'Cell 2', 'Cell 3'][:cell_nb], 
                        markers = ['x', 'x', 'x'][:cell_nb], 
                        colors = ['tab:blue', 'tab:orange', 'tab:green'][:cell_nb])

    user_input = input("When the new positions are set press Enter to go to the next frame or 'x' Enter to exit.")

    new_positions = klicker.get_positions()
    frame_ind = i
    for p, k in enumerate(new_positions):
        if new_positions[k].shape[0]:
            # Keep the position of the last click
            change_one_position(X, Y, frame_ind, p, new_positions[k][-1][0], new_positions[k][-1][1])

    plt.close(fig_frame)

    if user_input == '':
        correct_position(frames, X, Y, cell_nb=cell_nb, frame_no=(i+1)%frames.shape[0])

    return X, Y


def correct_swapping(X, Y, cell_nb=3):
    """Correct an exchange between two particles by asking input to the user."""

    frame_no = input("Enter the number ot the frame that you want to correct: ")
    if not frame_no.isdigit() or int(frame_no)<0 or int(frame_no)>=len(X):
        print("Warning: This is not a valid number.")
        return X, Y

    frame_number = int(frame_no)
    if cell_nb>2:
        while True:
            p1 = input("Enter first cell number:")
            if not p1.isdigit() or int(p1)>cell_nb:
                print("Warning: this is not a valid cell number")
                continue
            p2 = input("Enter second cell number:")
            if not p2.isdigit() or int(p2)>cell_nb:
                print("Warning: this is not a valid cell number")
                continue

            particle1 = int(p1) - 1
            particle2 = int(p2) - 1
            break
    else:
        particle1 = 0
        particle2 = 1

    X, Y = exchange_particles(X, Y, frame_number, particle1, particle2)
    print("Exchange performed!")

    return X, Y


def show_stacks(nuclei, bf, X, Y, cell_nb):
    """Open the visualization of frames and tracking results."""

    fig_bf, _, slider1, _ = plot_tracking_results(bf, X, Y, cell_nb=cell_nb, player=False)
    # Choose where the figure appears on the screen
    fig_bf.canvas.manager.window.wm_geometry("+%d+%d" % (2200, -200))

    fig_nuclei, _, slider2, player_anim = plot_tracking_results(nuclei, X, Y, cell_nb=cell_nb, player=True)
    fig_nuclei.canvas.manager.window.wm_geometry("+%d+%d" % (2200, -200))
    return fig_nuclei, fig_bf, slider2, slider1, player_anim


def correct_frame(frames, X, Y, cell_nb):
    """Ask user input for the different corrections to do."""
    while True:
        correction_type = input("Enter 'r' if you want to reposition a point, enter 's' if you want to switch two points. If all frames are good, enter 'x': ")
        if correction_type == 'r':
            X, Y = correct_position(frames, X, Y, cell_nb=cell_nb)
        elif correction_type == 's':
            X, Y = correct_swapping(X, Y, cell_nb=cell_nb)
        elif correction_type == 'x':
            break
        else:
            continue
    return X,Y


def plot_tracking_results(frames, X, Y, cell_nb=3, player=False):
    """Plot the frames and the tracking results, together with a slider, and eventually an automatic
    player."""

    plt.ion()

    nframes = len(frames)
    fig, axes, im, frame_text, points = plot_one_frame(frames, X, Y, 0, cell_nb=cell_nb, legend=True)

    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    slider=Slider(ax=slider_ax, label="frame", valmin=0, valmax=nframes-1, valstep=1)

    def update_slider(val):
        ind = slider.val
        ind = int(val)
        im.set_data(frames[ind])
        frame_text.set_text(f'{ind}/{nframes-1}')
        for part in range(cell_nb):
            points[part].set_offsets((X[part].values[ind],Y[part].values[ind]))
        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    if player:
        player_anim = Player(fig, update_slider, maxi=nframes-1, interval=500)
        plt.show()
    else:
        player_anim = None
    return fig, axes, slider, player_anim


def open_tracking(data_path, track_path, nuclei_file, bf_file, pattern_file, 
                  track_file, corrected_track=False):
    """Open all the data."""
   
    data_path = os.path.abspath(data_path)
    track_path = os.path.abspath(track_path)
    nuclei_file = os.path.join(data_path, nuclei_file)
    bf_file = os.path.join(data_path, bf_file)
    pattern_file = os.path.join(data_path, pattern_file)
    track_file = os.path.join(track_path, track_file)
    nuclei = imread(nuclei_file)
    bf = imread(bf_file)
    pattern = imread(pattern_file)
    if corrected_track:
        index_col = False
    else:
        index_col = 0
    trajectories = pd.read_csv(track_file, index_col=index_col)
    X = trajectories.set_index(['frame', 'particle'])['x'].unstack()
    Y = trajectories.set_index(['frame', 'particle'])['y'].unstack()
    print(f'There are {X.shape[1]} trajectories detected.')
    return nuclei, bf, pattern, X, Y


def save_results(X, Y, save_path, trackfile):
    """Save the corrected trajectories."""
    filename = trackfile.removesuffix('.csv') + '-corrected.csv'
    print(filename)
    filepath = os.path.join(os.path.abspath(save_path), filename)
    X2 = X.unstack().to_frame('x')
    Y2 = Y.unstack().to_frame('y')
    tracks = pd.concat([X2, Y2], axis=1)
    tracks.to_csv(filepath, sep=',')


def plot_one_frame(frames, X, Y, frame_no, cell_nb=3, legend=False):
    """Plot a single frame with the tracking result."""

    i = frame_no
    fig_frame, axes_frame = plt.subplots(constrained_layout=True) # Check with other code

    im = axes_frame.imshow(frames[i])
    frame_text = plt.text(5,5,f'{i}/{frames.shape[0]-1}',size='large',color='white',va='top',ha='left')

    dot_color = ['tab:blue', 'tab:orange', 'tab:green']

    points = [0 for _ in range(cell_nb)]
    for part in range(cell_nb):
        points[part] = plt.scatter(X[part].values[i],Y[part].values[i], color=dot_color[part])
        if legend:
            points[part].set_label(f'Cell {part+1}')

    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig_frame, axes_frame, im, frame_text, points


def set_origin(X, Y, origin_x, origin_y):
    """Set the origin of the trajectories by analysing the patter."""
    X = X - origin_x
    Y = Y - origin_y
    return X, Y

def find_center(pattern):
    """Find the center of the pattern."""
    img16 = (pattern).astype('uint16')
    ratio = np.amax(img16) / 256        
    img8 = (img16 / ratio).astype('uint8')
    blur = cv2.GaussianBlur(img8, (9,9), 0)
    denoise = cv2.fastNlMeansDenoising(blur,None,10,7,21)
    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,201,2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
            area = cv2.contourArea(i)
            # One might need to change the value of the minimal area depending on the pattern
            if area > 1000:
                rect = cv2.minAreaRect(i)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
    (cx, cy), (width, height), angle = rect

    fig, axes = plt.subplots(figsize = (7,4.5), constrained_layout=True)
    img2 = cv2.drawContours(img8,[box],0,(0,0,255),2)
    plt.ion()
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.scatter(cx, cy, marker='x', s=7000, color='r')
    plt.show()
    while True:
        user_input = input('Are you happy with the pattern detection? [Enter/n]')
        if user_input == 'n'or user_input == 'N':
            print('Unfortunately there is nothing we can do at the moment. If the centre detection is wrong, do not save the trajectories (no need to correct the tracking).')
            break
        elif user_input == '':
            print("Let's move to tracking correction!")
            break
        else: 
            print('This is not a valid answer.')
        # TODO: Implement a way for the user to correct the wrong detection
    plt.close(fig)
    return cx, cy
