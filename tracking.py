import numpy as np
import trackpy as tp

def track(nuclei, diameter=25, minmass=0, separation=7, topn=3, track_memory=20, max_travel=300,
          velocity_predict=False, verbose=False):
    """Track the cell's nuclei in a sequence of frames.
    Parameters:
    nuclei -- stack of image
    diameter -- the approximate diameter of a nucleus. If chosen too small, the position may not be
                in the center of the nuclei. If chosen too big, two nuclei might be detected as one.
    minmass -- the minimum brightness to recognize a nucleus as such.
    separation -- the minimum distance between two particles detected. If too small, two particles
                  can be detected in one nucleus. If too big, two nuclei can be detected as one.
    topn -- number of nuclei that one wants to detect.
    track_memory -- the maximum number of frame during which the nuclei can be not detected and 
                    reappear.
    max_travel -- The maximum distance in pixels a nuclei can move between frames. Keep it big.
    velocity_predict -- if True, use a predictor based on the current velocity of the particles. 
                        It might be that it's also the default predictor of trackpy, to be checked.
    verbose -- if True, print some output.
    Returns:
    trajectories (DataFrame).
    """
    max_travel = np.round(max_travel) 
    diameter = int(diameter + (not diameter%2))  # force odd diameter required by tp.batch

    if not verbose:
        tp.quiet()

    if verbose:
        print('Tracking nuclei using trackpy...')
    f = tp.batch(nuclei, diameter, minmass=minmass, separation=separation, topn=topn)
    if velocity_predict:
        pred = tp.predict.NearestVelocityPredict()
        trajectories = pred.link_df(f, max_travel, memory=track_memory)
    else:
        trajectories = tp.link(f, max_travel, memory=track_memory)
    #trajectories = tp.filter_stubs(t, min_frames)
    if verbose:
        print('Tracking of nuclei completed.')

    return trajectories
