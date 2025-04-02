import os
import pandas as pd
import tracking_correction as tc
import glob
import matplotlib.pyplot as plt
from multiprocessing import freeze_support



def main():

    cell_nb = 3
    data_path = r"Z:\user\Agathe.Jouneau\3-cell-project\20240426\Data_analysis\cropped_sequences_20240426\3_cells"
    track_path = r"Z:\user\Agathe.Jouneau\3-cell-project\20240426\Data_analysis\cropped_sequences_20240426\3_cells\tracking_results"
    save_path = r"Z:\user\Agathe.Jouneau\3-cell-project\20240426\Data_analysis\cropped_sequences_20240426\3_cells\corrected_tracks"
    nuclei_suffix = '-Hoechst.tif'
    bf_suffix = '.tif'
    pattern_suffix = '-Alexa488.tif'
    track_suffix = '-tracking.csv'
    start_position = 35
    
    # um_per_pixel = 0.66  # Add this for the next project


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = sorted(glob.glob(track_path +r'\*-tracking.csv'))
    for filename in files:
        if os.path.basename(filename)[5]=='-':
            position = os.path.basename(filename)[:5]
        else:
            position = os.path.basename(filename)[:6]
        position_int = int(position[:3])
        if position_int>=start_position:

            mcherry_file = position + nuclei_suffix
            bf_file = position + bf_suffix
            pattern_file = position + pattern_suffix
            track_file = position + track_suffix

            print('File: ', position)
            nuclei, bf, pattern, X, Y = tc.open_tracking(data_path, track_path, mcherry_file,
                                                         bf_file, pattern_file, track_file)
            origin_x, origin_y = tc.find_center(pattern)
            fig_nuclei, fig_bf, keep_slider1, keep_slider2, keep_player_anim = tc.show_stacks(nuclei, bf, X, Y, cell_nb=cell_nb)
            X, Y = tc.correct_frame(nuclei, X, Y, cell_nb=cell_nb)
            plt.close(fig_nuclei)
            plt.close(fig_bf)

            X, Y = tc.set_origin(X, Y, origin_x, origin_y)
            # TODO:  X, Y = tc.rescale(um_per_pixel)

            while True:
                save = input('Do you want to save the result? [Enter/n]:')
                if save == '':
                    tc.save_results(X, Y, save_path, track_file)
                    break
                elif save == 'n' or save == 'N':
                    break
                else:
                    continue

    return keep_slider1, keep_slider2, keep_player_anim

if __name__ == '__main__':
    freeze_support()
    keep_slider1, keep_slider2, keep_player_anim = main()
    