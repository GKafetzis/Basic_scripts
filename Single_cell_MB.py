import random
import numpy as np
import pandas as pd
import bisect

rng = np.random.default_rng()

from MEA_analysis import stimulus_and_spikes as sas
from Basic_scripts import Basic
from math import radians, degrees

from itertools import compress
import pyspike
import pycircstat
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors

from ipywidgets import HBox, VBox
import ipywidgets as widgets


# print ("You have imported 'Single_cell_MB' ")
# print("Please run the function 'check_order' with order of directions you played")


def check_order(dir_degrees):
    re_order = estimate_reorder_indices(dir_degrees)
    angles_ordered_a, angles_ordered_d = order_angles(dir_degrees, re_order)
    return re_order, angles_ordered_a, angles_ordered_d


def estimate_reorder_indices(inpresentation_order):
    re_order = []
    for i in inpresentation_order:
        re_order.append(np.where(np.sort(inpresentation_order) == i)[0][0])

    return re_order


def reorder(arr, index, n):
    temp = [0] * n;
    for i in range(0, n):
        temp[index[i]] = arr[i]
    return temp


def order_angles(dir_degrees, re_order):
    angles_ordered_d = reorder(dir_degrees, re_order, len(dir_degrees))
    # angles_ordered_d = np.append(angles_ordered_d, angles_ordered_d[0])

    angles_ordered_a = np.zeros(len(angles_ordered_d))
    for i in range(len(angles_ordered_d)):
        angles_ordered_a[i] = radians(angles_ordered_d[i])

    return angles_ordered_a, angles_ordered_d


def spikes_and_traits(single_stimulus, cell_index, df_spikes, df_stimulus, ):
    if type(single_stimulus) == int:
        cell_spikes = df_spikes.query("`Stimulus ID`==@single_stimulus and `Cell index`==@cell_index")['Spikes'].values[
            0].compressed()
        stim_traits = Basic.get_stimulus_traits(df_stimulus, single_stimulus)

    elif type(single_stimulus) == str:
        stim_idx = df_stimulus[df_stimulus['Stimulus_name'] == single_stimulus].index.get_level_values(0)[0]

        cell_spikes = df_spikes.query("`Stimulus ID`==@stim_idx and `Cell index`==@cell_index")['Spikes'].values[
            0].compressed()
        stim_traits = Basic.get_stimulus_traits(df_stimulus, stim_idx)

    return cell_spikes, stim_traits


def spikes_per_seg(single_stimulus, cell_index, df_spikes, df_stimulus, inrange=None, return_traits=False):
    cell_spikes, stim_traits = spikes_and_traits(single_stimulus, cell_index, df_spikes, df_stimulus)

    spikes_container = []
    if inrange:
        for trial in range(inrange[0], inrange[1]):
            spikes_container.append([int(val) for sublist in (sas.get_spikes_per_trigger_type_new(
                cell_spikes, stim_traits['stim_rel_trig'],
                trial, stim_traits['stim_trials'])[0]) for val in sublist])
    else:
        for trial in range(stim_traits['stim_trials']):
            spikes_container.append([int(val) for sublist in (sas.get_spikes_per_trigger_type_new(
                cell_spikes, stim_traits['stim_rel_trig'],
                trial, stim_traits['stim_trials'])[0]) for val in sublist])
    if return_traits:
        return spikes_container, stim_traits
    else:
        return spikes_container


def place_holder():
    return


def calc_pol(spikes_container, thresh):
    spikes_container = np.array(np.sort(Basic.flatten_list(spikes_container)))
    all_spikes = len(spikes_container)
    ON_spikes = bisect.bisect_left(spikes_container, thresh)
    OFF_spikes = all_spikes - ON_spikes
    try:
        polarity_index = (ON_spikes - OFF_spikes) / (ON_spikes + OFF_spikes)
    except ZeroDivisionError:
        polarity_index = np.nan
    return polarity_index


def pol_all(single_stimulus, cell_idces, df_spikes, df_stimulus, inrange=None, thresh=None):
    polarities = np.zeros(len(cell_idces))
    if thresh:
        for idx, cell_idx in enumerate(cell_idces):
            spikes_container, stim_traits = spikes_per_seg(single_stimulus, cell_idx, df_spikes, df_stimulus,
                                                           inrange=inrange, return_traits=True)
            polarities[idx] = calc_pol(spikes_container, thresh=thresh)
    else:
        print('Threshold was not given, calculating it internally for stimuli with sublogic=2')
        for idx, cell_idx in enumerate(cell_idces):
            spikes_container, stim_traits = spikes_per_seg(single_stimulus, cell_idx, df_spikes, df_stimulus,
                                                           inrange=inrange, return_traits=True)
            polarities[idx] = calc_pol(spikes_container,
                                       thresh=stim_traits['sampling_freq'] * stim_traits['stim_phase_dur'] / 2)
    return polarities


def nspikes_per_seg(single_stimulus, cell_index, df_spikes, df_stimulus, toreorder=False, re_order=None, ashist=False,
                    nr_bins=12):
    """
    cell_index: value of 'Cell index'
    re_order: sequence of re_ordering. To be generated by running the check_order function with dir_degrees
    as input.
    """
    cell_spikes, stim_traits = spikes_and_traits(single_stimulus, cell_index, df_spikes, df_stimulus)
    spikes_per_direction = np.zeros(stim_traits['stim_trials'])
    spikes_per_segment = np.zeros([stim_traits['stim_repeats'], stim_traits['stim_trials']])

    hist_per_direction = np.zeros((stim_traits['stim_trials'] + 1, nr_bins))
    for trial in range(stim_traits['stim_trials']):

        spikes_per_dir = (sas.get_spikes_per_trigger_type_new(
            cell_spikes, stim_traits['stim_rel_trig'],
            trial, stim_traits['stim_trials'])[0])

        spikes_per_direction[trial] = sum([len(listElem) for listElem in spikes_per_dir]) / stim_traits['stim_repeats']

        if ashist:
            hist_per_direction[trial, :] = \
                np.histogram(
                    np.sort([val / stim_traits['sampling_freq'] for sublist in spikes_per_dir for val in sublist]),
                    bins=nr_bins, range=[0, stim_traits['stim_phase_dur']])[0]
            if trial == stim_traits['stim_trials'] - 1:
                hist_per_direction[trial + 1, :] = np.histogram(
                    np.sort([val / stim_traits['sampling_freq'] for sublist in spikes_per_dir for val in sublist]),
                    bins=nr_bins, range=[0, stim_traits['stim_phase_dur']])[1][1:]

        for repeat in range(stim_traits['stim_repeats']):
            spikes_per_segment[repeat, trial] = len(spikes_per_dir[repeat])

    if toreorder:
        spikes_per_direction = reorder_spikes_per_dir(spikes_per_direction, re_order)
        spikes_per_segment = reorder_spikes_per_seg(spikes_per_segment, re_order)

    return spikes_per_direction, spikes_per_segment, hist_per_direction,


def reorder_spikes_per_dir(spikes_per_dir, re_order):
    spikes_per_dir = reorder(spikes_per_dir, re_order, len(spikes_per_dir))
    return spikes_per_dir


def reorder_spikes_per_seg(spikes_per_seg, re_order):
    for repeat in range(spikes_per_seg.shape[0]):
        spikes_per_seg[repeat, :] = reorder(spikes_per_seg[repeat, :], re_order, len(spikes_per_seg[repeat]))
    return spikes_per_seg


def norm_byarea(spikes_per_segment):
    """
    Normalization by area, default
    """
    normed_spikes_per_segment = np.mean(spikes_per_segment, axis=0) / (
            np.nansum(spikes_per_segment) / (spikes_per_segment.shape[0] * spikes_per_segment.shape[1]))

    return normed_spikes_per_segment


def dominant_direction(arr):
    return int(np.argmax(arr))


def get_vector_angle_degrees(normed_spikes_per_segment, dir_degrees, d=0.78539816):
    return degrees(pycircstat.mean(check_order(dir_degrees)[1], normed_spikes_per_segment, d=d))


def plot_polar_histogram(df_spikes, stim_id, default_pop, bins=15, title_name='Sampling Visual Space', weights=False, ):
    """
    Currently implemented for single-recs, same as the majority of functions in this package.
    """
    if weights:
        print("Implementation of weighted histograms is still missing")
        return
    df_part = df_spikes.loc[default_pop, stim_id, slice(None)].copy()
    DScells = df_part.index.get_level_values(0)[df_part['DSI_sig']].values
    nonDScells = df_part.index.get_level_values(0)[~df_part['DSI_sig']].values

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "polar"}] * 2] * 1,
                        subplot_titles=("Other RGCs", "DS RGCs"))

    fig.add_trace(go.Barpolar(

        r=np.histogram(df_part.loc[nonDScells, slice(None), (slice(None))]['Prefangle'].values,
                       bins=bins, range=[0, 360])[0],
        theta=np.histogram(df_part.loc[nonDScells, slice(None), (slice(None))]['Prefangle'].values,
                           bins=bins, range=[0, 360])[1][:-1],
        width=360 / bins,
        marker=dict(color='lightgray')
    ), 1, 1)

    fig.add_trace(go.Barpolar(
        r=np.histogram(df_part.loc[DScells, slice(None), (slice(None))]['Prefangle'].values,
                       bins=bins, range=[0, 360])[0],
        theta=np.histogram(df_part.loc[DScells, slice(None), (slice(None))]['Prefangle'].values,
                           bins=bins, range=[0, 360])[1][:-1],
        width=360 / bins,
        marker=dict(color='red')
    ), 1, 2)
    fig.update_polars(
        radialaxis=dict(showticklabels=False, ticks='', tickmode='array', ),
        angularaxis=dict(showticklabels=False, ticks='')
    )
    # tickvals=[0.5, 1, ], range=[0, 1]),
    fig.update_layout(height=500, width=800,
                      title_text=f"{title_name} <br> {df_part['Stimulus name'].unique()[0]}", title_x=0.5,
                      showlegend=False)

    return fig


def calculate_ds(normed_spikes_per_segment):
    """
    Trick is to tile it once so it becomes circularly symmetric irrespective of index of dominant direction
    """

    dominant_dir = dominant_direction(normed_spikes_per_segment)
    dummy_array = np.tile(normed_spikes_per_segment, 2)
    dominant_ds = dummy_array[dominant_dir]
    null_ds = dummy_array[dominant_dir + 4]

    ds = (dominant_ds - null_ds) / (dominant_ds + null_ds)

    return ds


def calculate_os(normed_spikes_per_segment):
    """
    Trick is to tile it once so it becomes circularly symmetric irrespective of index of dominant direction
    """

    dominant_dir = dominant_direction(normed_spikes_per_segment)
    dummy_array = np.tile(normed_spikes_per_segment, 2)
    dominant_ds = dummy_array[dominant_dir]

    dominant_os = dominant_ds + dummy_array[dominant_dir + 4]
    null_os = dummy_array[dominant_dir + 2] + dummy_array[dominant_dir + 6]

    os = (dominant_os - null_os) / (dominant_os + null_os)

    return os


def calculate_dos(normed_spikes_per_segment):
    return calculate_ds(normed_spikes_per_segment), calculate_os(normed_spikes_per_segment)


def calculate_cross(normed_spikes_per_segment):
    """
    Trick is to tile it once so it becomes circularly symmetric irrespective of index of dominant direction
    """

    dominant_dir = dominant_direction(normed_spikes_per_segment)
    dummy_array = np.tile(normed_spikes_per_segment, 2)
    dominant_ds = dummy_array[dominant_dir]

    dominant_cross = dominant_ds + dummy_array[dominant_dir + 2] + dummy_array[dominant_dir + 4] + dummy_array[
        dominant_dir + 6]
    null_cross = dummy_array[dominant_dir + 1] + dummy_array[dominant_dir + 3] + dummy_array[dominant_dir + 5] + \
                 dummy_array[dominant_dir + 7]

    cross = (dominant_cross - null_cross) / (dominant_cross + null_cross)

    return cross


def create_permuted_spikes(single_stimulus, cell_index, df_spikes, df_stimulus, npermut=1000, ):
    cell_spikes, stim_traits = spikes_and_traits(single_stimulus, cell_index, df_spikes, df_stimulus)
    nspikes_per_repeat = np.round(len(cell_spikes) / stim_traits['stim_repeats'], 0).astype(int)
    perm_pop = np.zeros([npermut, stim_traits['stim_repeats'], stim_traits['stim_trials']])

    for perm in range(npermut):
        for repeat in range(stim_traits['stim_repeats']):
            perm_pop[perm, repeat] = np.bincount(
                rng.integers(0, stim_traits['stim_trials'], size=nspikes_per_repeat),
                minlength=stim_traits['stim_trials'])

    return perm_pop


def sig_test(selectivity_index, perm_pop, a=0.05, idx='ds'):
    dsi_perm_vals = np.zeros(perm_pop.shape[0])
    if idx == 'ds':
        for neuron in range(len(dsi_perm_vals)):
            dsi_perm_vals[neuron] = calculate_ds(norm_byarea(perm_pop[neuron]))
    elif idx == 'os':
        for neuron in range(len(dsi_perm_vals)):
            dsi_perm_vals[neuron] = calculate_os(norm_byarea(perm_pop[neuron]))

    if sum(dsi_perm_vals > selectivity_index) / perm_pop.shape[0] < a:
        return True
    else:
        return False


def is_quiet(spikes_per_dir, level=0.8):
    return True if np.max(spikes_per_dir) < level else False


def is_silent(hist_per_dir, t_thresh=1, sig_thresh=4):
    if sum(np.unique([len(find_threshold_events(hist_per_dir, t_thresh, sig_thresh)[i]) for i in
                      range(hist_per_dir.shape[0] - 1)])) < 1:
        return True
    else:
        return False


def find_threshold_events(hist_per_dir, t_thresh, sig_thresh):
    """
    The last row of hist_per_dir contains the time bins, and as such it is considered as the time_profile
    """

    events = []
    spike_profile = hist_per_dir[:-1]
    time_profile = hist_per_dir[-1]

    for trial in range(spike_profile.shape[0]):
        idces = []
        points = get_persistent_homology(spike_profile[trial])
        for point in points:
            if point.died == None:
                point.died = 0
            if point.born == None:
                point.born = 0
            if (time_profile[point.born] > t_thresh) & (
                    spike_profile[trial][point.born] - spike_profile[trial][point.died] > sig_thresh):
                idces.append(True)
            else:
                idces.append(False)
        events.append(list(compress(points, idces)))

    return events


def moving_cell(single_stimulus, cell_index, df_spikes, df_stimulus, toreorder=False, dir_degrees=None, ashist=False,
                nr_bins=12, t_thresh=1, sig_thresh=4, quiet_level=0.8, npermut=1000):
    if toreorder:
        re_order = check_order(dir_degrees)[0]
    else:
        re_order = None
    spikes_per_dir, spikesperseg, hist_per_dir = nspikes_per_seg(single_stimulus, cell_index, df_spikes, df_stimulus,
                                                                 toreorder,
                                                                 re_order, ashist,
                                                                 nr_bins)
    if is_silent(hist_per_dir, t_thresh, sig_thresh):
        return [None] * 6
    elif is_quiet(spikes_per_dir, level=quiet_level):
        return [None] * 6
    else:
        ds, os = calculate_dos(norm_byarea(spikesperseg))
        # print('DSI:', ds)
        # print('OSI:', os)

        perm_pop = create_permuted_spikes(single_stimulus, cell_index, df_spikes, df_stimulus, npermut=npermut)
        # print('Signif:', dsi_test(calculate_dos(norm_byarea(spikesperseg))[0], perm_pop))

        if calculate_cross(norm_byarea(spikesperseg)) > 0.4:
            print(cell_index, ds, os)
        return dominant_direction(norm_byarea(spikesperseg)), ds, sig_test(ds, perm_pop, idx='ds'), os, sig_test(os,
                                                                                                                 perm_pop,
                                                                                                                 idx='os'), get_vector_angle_degrees(
            norm_byarea(spikesperseg), dir_degrees)


def moving_all(df_spikes, df_stimulus, toreorder=False, dir_degrees=None, ashist=False,
               nr_bins=12, t_thresh=1, sig_thresh=4, quiet_level=0.8):
    dirs = np.zeros(len(df_spikes))
    angles = np.zeros(len(df_spikes))
    dss = np.zeros(len(df_spikes))
    dstests = np.full(len(df_spikes), False)
    oss = np.zeros(len(df_spikes))
    osstests = np.full(len(df_spikes), False)
    # cell_idces = np.zeros(len(df_spikes))
    for row, idx in zip(df_spikes.itertuples(), range(len(df_spikes))):
        cell_index = row[0][0]
        single_stimulus = row[0][1]

        storage = moving_cell(single_stimulus, cell_index, df_spikes, df_stimulus, toreorder, dir_degrees, ashist,
                              nr_bins, t_thresh, sig_thresh, quiet_level)
        if storage:
            dirs[idx], dss[idx], dstests[idx], oss[idx], osstests[idx], angles[idx] = storage
            for index, test in enumerate([dstests, osstests]):

                if (test[idx]) and ('New_qi' in df_spikes.columns):
                    sel_index = [dss, oss][index]
                    """The presence of 0-comparison rules out unwanted Falsifications when Qi is not
                    computed for a stimulus but is computed for others"""
                    if ('New_qi' in df_spikes.columns) and (0 < df_spikes.loc[row[0]]['New_qi'] < 0.14):
                        test[idx] = False
                    elif sel_index[idx] < 0.3:
                        test[idx] = False
        else:
            pass
        # cell_idces[idx] = cell_index
    df_spikes["DSI_sig"] = dstests
    df_spikes["DSI"] = dss
    df_spikes["OSI"] = oss
    df_spikes["OSI_sig"] = osstests
    df_spikes["Prefdir"] = dirs
    df_spikes["Prefangle"] = angles
    # data = np.array([cell_idces, dirs, dss, dstests, oss])
    # return pd.DataFrame(data=data[1:].T, index=data[0], columns=['dom_dir', 'ds', 'sig', 'os'])
    return df_spikes


################ Plotting part #########################

def get_cell_spiketrains_per_stimulus(cell_spikes, stim_traits, toreorder=False,
                                      re_order=None, ):
    spiketrains_list = []

    for trial in range(stim_traits['stim_trials']):

        spikes_per_trial = (sas.get_spikes_per_trigger_type_new(cell_spikes, stim_traits['stim_rel_trig'], trial,
                                                                stim_traits['stim_trials'])[0])

        spiketrain_list = []
        for repeat in range(stim_traits['stim_repeats']):
            spikes_per_trial[repeat] = spikes_per_trial[repeat] / stim_traits['sampling_freq']
            spiketrain_list.append(
                pyspike.SpikeTrain(spikes_per_trial[repeat], edges=[0, stim_traits['stim_phase_dur']]))

        spiketrains_list.append(spiketrain_list)

    if toreorder:
        spiketrains_list = reorder_spikes_per_dir(spiketrains_list, re_order)

    return spiketrains_list


def plot_ds_spiketrains(single_stimulus, cell_index, df_spikes, df_stimulus, toreorder=False,
                        re_order=None, ):
    cell_spikes, stim_traits = spikes_and_traits(single_stimulus, cell_index, df_spikes, df_stimulus)

    spiketrains_list = get_cell_spiketrains_per_stimulus(cell_spikes, stim_traits, toreorder, re_order, )

    return Basic.plot_sc_aligned_new(cell_index, spiketrains_list, stim_traits, stim_traits['stim_phase_dur'])


def _encircle(arr):
    return np.insert(arr, len(arr), arr[0])


def plot_polar_ds(spikes_per_segment, dir_degrees):
    """
    Input needs to be correctly reordered already
    """
    re_order, angles_ordered_a, angles_ordered_d = check_order(dir_degrees)
    cols = colors.DEFAULT_PLOTLY_COLORS
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}] * 1] * 1, subplot_titles='')

    for repeat in range(spikes_per_segment.shape[0]):
        sp = spikes_per_segment[repeat] / (
                np.nansum(spikes_per_segment) / (spikes_per_segment.shape[0] * spikes_per_segment.shape[1]))

        fig.add_trace(go.Scatterpolar(
            r=_encircle(sp),
            theta=_encircle(angles_ordered_d),
            mode='lines',
            marker=dict(color='#d6d6d6')
        ))

    fig.add_trace(go.Scatterpolar(
        r=_encircle(norm_byarea(spikes_per_segment)),
        theta=_encircle(angles_ordered_d),
        mode='lines',
        marker=dict(color=cols[0])
    ))

    fig.add_trace(go.Scatterpolar(
        r=[0, pycircstat.resultant_vector_length(angles_ordered_a,
                                                 norm_byarea(spikes_per_segment), d=0.78539816)],
        theta=[0, degrees(pycircstat.mean(angles_ordered_a,
                                          norm_byarea(spikes_per_segment), d=0.78539816))],
        mode='lines',
        marker=dict(color=cols[3])
    ))

    fig.update_annotations(yshift=25, xshift=-45)

    fig.update_layout(autosize=False,
                      width=580,
                      height=580,
                      showlegend=False)
    fig.update_polars(radialaxis=dict(tickmode='array', tickvals=[0.5, 1], ticktext=['', ''], range=[0, 3]))

    return fig


def plot_ds_overview_cell(single_stimulus, cell_index, df_spikes, df_stimulus, dir_degrees, toreorder=False, ):
    spikes_per_segment = nspikes_per_seg(single_stimulus, cell_index, df_spikes, df_stimulus, toreorder=True,
                                         re_order=check_order(dir_degrees)[0])[1]
    fig_polar_ds = plot_polar_ds(spikes_per_segment, dir_degrees)

    fig_ds_spiketrains = plot_ds_spiketrains(single_stimulus, cell_index, df_spikes, df_stimulus, toreorder,
                                             re_order=check_order(dir_degrees)[0])
    fig_ds_spiketrains.update_layout(autosize=False, width=1600, height=620)
    fig_ds_spiketrains.show()
    fig_polar_ds.show()

    return fig_ds_spiketrains, fig_polar_ds,


class Peak:
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return float("inf") if self.died is None else seq[self.born] - seq[self.died]


def get_persistent_homology(seq):
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key=lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx - 1] is not None)
        rgtdone = (idx < len(seq) - 1 and idxtopeak[idx + 1] is not None)
        il = idxtopeak[idx - 1] if lftdone else None
        ir = idxtopeak[idx + 1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks) - 1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)
