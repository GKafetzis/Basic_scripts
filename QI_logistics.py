import traceback
import multiprocessing as mp
from functools import partial
from typing import Union

import numpy as np
import math
import bisect
from MEA_analysis import stimulus_and_spikes as sp
import scipy.signal as signal
import pandas as pd
import plotly.express as px


# TODO Allow calculate_qi for some stimuli in dataframe, currently all
def kernel_template(width=0.01, sampling_freq=17852.767845719834):
    """
    create Gaussian kernel by providing the width (FWHM) value. According to wiki, this is approx. 2.355*std of dist.
    width=given in seconds, converted internally in sampling points.

    Returns
    Gaussian kernel
    """
    fwhm = int((sampling_freq) * width)  # in points

    # normalized time vector in ms
    k = int((sampling_freq) * 0.02)
    gtime = np.arange(-k, k)

    # create Gaussian window
    gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)
    gauswin = gauswin / np.sum(gauswin)

    # initialize filtered signal vector
    return gauswin


def spike_padding(spikes, trial_n, sampling_freq=17852.767845719834):
    """
    Deprecated, please do not use
    """
    stim_time = np.zeros(len(np.linspace(0, int(4.05 * 6 * sampling_freq),
                                         num=int(4.05 * 6 * sampling_freq), endpoint=False)))

    for sp_times in range(len(spikes[trial_n])):
        stim_time[int(spikes[trial_n][sp_times])] = 1

    return stim_time


def spike_padding_new(stim_time, spikes, trial_n):
    """
    Takes a stim_time window (array of zeros), and populates it with 1 at each index a spike happened. Trial_specific.

    Returns
    The 0-and-1s stim_time window
    """
    stim_time[trial_n][list(spikes[trial_n].astype(int))] = 1
    return stim_time


def calc_tradqi(kernelfits):
    """
    Calculates the Quality_index of the (Gaussian-kernel) fitted spike data. Cell-specific and stimulus-specific,
    but of nr-of-trials length

    Returns
    The QI value for the cells-response to this stimulus
    """
    return np.var(np.mean(kernelfits, 0)) / np.mean(np.var(kernelfits, 0))


def fast_qi(spikes, stimulus_traits, repeat_duration, kernel_width, sampling_freq=17852.767845719834):
    """
    Faster implementation to the actual convolution of the spikes stim_time with a Gaussian window, BUT only for few
    spikes. Works by fitting hats around the location of each spike, and deals with border cases (where the hat does not
    fit either at the beginning or end of trial) by adding part of the hat.
    Alternative to this function is the fast_qi_exclusive, where these spikes are omitted from further consideration.
    """

    gauswin = kernel_template(width=kernel_width)
    exs = np.zeros((stimulus_traits['stim_repeats'], int(sampling_freq) * int(repeat_duration)))
    for trial in range(len(spikes)):
        a = np.zeros((len(spikes[trial]), int(sampling_freq) * int(repeat_duration)))
        for idx, spike in enumerate(spikes[trial]):
            # print(np.ceil(spike-(len(gauswin)/2)))
            try:
                a[idx][int(spike - (len(gauswin) / 2)):int(spike - (len(gauswin) / 2)) + len(gauswin)] = gauswin
            except ValueError:
                if np.ceil(spike - (len(gauswin) / 2)) < 0:
                    a[idx][0:len(gauswin) - int((len(gauswin) / 2) - np.ceil(spike))] = gauswin[int((
                                                                                                            len(gauswin) / 2) - np.ceil(
                        spike)):]
                elif np.ceil(spike + (len(gauswin) / 2)) > int(repeat_duration) * int(
                        Test_stimulus.sampling_frequency[0]):
                    # try:
                    a[idx][int(spike - (len(gauswin) / 2)):] = gauswin[:int(repeat_duration) * int(
                        sampling_freq) - int(spike - (len(gauswin) / 2))]
                    # except ValueError:
                    # print(spike, int(24*Test_stimulus.sampling_frequency[0]-int(spike-(len(gauswin)/2))) )
                else:
                    print('TF')
        exs[trial] = np.sum(a, 0)
    return calc_tradqi(exs)


def fast_qi_exclusive_old(spikes, stimulus_traits, repeat_duration, sampling_freq=17852.767845719834):
    """
    Deprecated, please do not use
    """
    exs = np.zeros((stimulus_traits['stim_repeats'], int(sampling_freq) * int(repeat_duration)))
    for trial in range(len(spikes)):
        a = np.zeros((len(spikes[trial]), int(sampling_freq) * int(repeat_duration)))
        for idx, spike in enumerate(spikes[trial]):
            # print(np.ceil(spike-(len(gauswin)/2)))
            try:
                a[idx][int(spike - (len(gauswin) / 2)):int(spike - (len(gauswin) / 2)) + len(gauswin)] = gauswin
            except ValueError:
                continue
                # except ValueError:
                # print(spike, int(24*Test_stimulus.sampling_frequency[0]-int(spike-(len(gauswin)/2))) )
        exs[trial] = np.sum(a, 0)
    return calc_tradqi(exs)


def fast_qi_exclusive(spikes, stimulus_traits, kernel_width, ):
    """
    Faster implementation to the actual convolution of the spikes stim_time with a Gaussian window, BUT only for few
    spikes. Works by fitting hats around the location of each spike, and deals with border cases (where the hat does not
    fit either at the beginning or end of trial) by omitting them from further consideration.
    Alternative to this function is the fast_qi, where these spikes are fitted with partial hats.
    """

    gauswin = kernel_template(width=kernel_width)

    exs = np.zeros(
        (stimulus_traits['repeats'], int(stimulus_traits['sampling_freq']) * int(stimulus_traits['repeat_duration'])))
    for trial in range(len(spikes)):
        a = np.zeros(
            (len(spikes[trial]), int(stimulus_traits['sampling_freq']) * int(stimulus_traits['repeat_duration'])))
        for sp_idx, spike in enumerate(spikes[trial]):
            # print(np.ceil(spike-(len(gauswin)/2)))
            try:
                a[sp_idx][int(spike - (len(gauswin) / 2)):int(spike - (len(gauswin) / 2)) + len(gauswin)] = gauswin
            except ValueError:
                continue
                # except ValueError:
                # print(spike, int(24*Test_stimulus.sampling_frequency[0]-int(spike-(len(gauswin)/2))) )
        exs[trial] = np.sum(a, 0)

    return calc_tradqi(exs)


def rebin_spikes(spike_idxs: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Calculate the spike indices after downsampling.
    Args:
        spike_idxs: The spike indices before downsampling.
    This method is very simple: just floor divide the indicies. It's good to
    have a dedicated function just so we are clear what the behaviour is,
    and to insure that we are doing it consistently.
    """
    res = np.floor_divide(spike_idxs, downsample_factor)
    return res


def decompress_spikes(
        spikes: Union[np.ndarray, np.ma.MaskedArray],
        num_sensor_samples: int,
        downsample_factor: int = 1,
) -> np.ndarray:
    """
    Fills an integer array counting the number of spikes that occurred.
    If downsample_factor is 1 (no downampling), then the output array will
    be an array of 0s and 1s, where 1 indicates that a spike occurred in that
    time bin.
    Setting downsample_factor to an integer greater than 1 will result in
    the spikes being counted in larger bin sizes that the original sensor
    sample period. So we are not talking about signal downsampling, Nyquist
    rates etc., rather we are talking about histogram binning where the bin
    size is scaled by downsample_factor. This behaviour is similar to Pandas's
    resample().sum() pattern.
    Binning behaviour
    -----------------
    As only integer values are accepted for downsample_factor, the binning is
    achieved by floor division of the original spike index. Examples:
        1. Input: [0, 0, 0, 1, 1], downsample_factor=2, output: [0, 1, 1]
        1. Input: [0, 0, 0, 1, 1, 1], downsample_factor=2, output: [0, 1, 2]
    """
    if np.ma.isMaskedArray(spikes):
        spikes = spikes.compressed()
    downsampled_spikes = rebin_spikes(spikes, downsample_factor)
    res = np.zeros(
        shape=[
            math.ceil(num_sensor_samples / downsample_factor),
        ],
        dtype=int,
    )
    np.add.at(res, downsampled_spikes, 1)
    return res


def create_full_logbook(stimulus_df):
    """
    Either queries or computes useful stimulus-specific properties and stores them in separate lists.
    Currently to be used exclusively with 'calculate_qi' function, need to think of further uses.
    """
    nr_stimuli = stimulus_df.index.unique(0)
    sampling_freq = 17852.767845719834
    trigger_completes = [stimulus_df.loc[i][
                             "Trigger_Fr_relative"
                         ] for i in nr_stimuli]

    repeat_logics = [int(
        stimulus_df.loc[i]["Stimulus_repeat_logic"
        ]) for i in nr_stimuli]

    repeats = [int(
        (len(trigger_completes[i]) - 1) / repeat_logics[i])
        for i in nr_stimuli]

    repeat_durations = [np.floor((trigger_completes[i][repeat_logics[i]] - trigger_completes[i][0]) /
                                 int(sampling_freq)
                                 ) for i in nr_stimuli]

    return sampling_freq, trigger_completes, repeat_logics, repeats, repeat_durations


def calculate_qi(stimulus_df, spikes_df, forwhich: list = [0, 1, 2, 3, 4, 5, ], nspikes_thres=30, kernel_width=0.0125,
                 fuse='decompress', dsf=200):
    """
    The main function. Given a FULL stimulus_df(i.e. qi to be computed for ALL stimuli), it calculates the QI value with
    the Gaussian kernel implementation, and returns the spikes_df with a 'New_qi' column storing the computed values.

    nspikes_thres: Sets the cell-specific threshold of averages spikes per trial, and determines whether we will proceed
    to the computation of the qi by gaussian convolution or by hat fitting
    """
    ####cell_idces currently not part of the script
    qi_per_cell = np.zeros(len(spikes_df))

    sampling_freq, trigger_completes, repeat_logics, repeats, repeat_durations = create_full_logbook(stimulus_df)
    stimulus_traits = {
        'sampling_freq': sampling_freq
    }

    for row, idx in zip(spikes_df.itertuples(), range(len(spikes_df))):

        stimulus_index = row[0][1]

        if stimulus_index not in forwhich:
            qi_per_cell[idx] = 0
        else:
            cell_spikes = row[1].compressed()

            if np.sum(cell_spikes) < 1:
                qi_per_cell[idx] = np.nan
                continue
            else:

                if idx == 0 or idx == 1000:
                    print(stimulus_df.loc[stimulus_index]['Stimulus_name'], repeat_durations[stimulus_index])

                spikes = sp.get_spikes_whole_stimulus_new_trainsomitted(
                    cell_spikes, trigger_completes[stimulus_index], repeat_logics[stimulus_index], 1)
                spikes = spikes[0]

                if sum([len(spikes[i]) for i in range(len(spikes))]) / repeats[stimulus_index] < nspikes_thres:
                    stimulus_traits['repeats'] = repeats[stimulus_index]
                    stimulus_traits['repeat_duration'] = repeat_durations[stimulus_index]

                    qi_per_cell[idx] = fast_qi_exclusive(spikes, stimulus_traits, kernel_width)
                else:
                    if fuse == 'decompress':
                        frames = math.ceil(repeat_durations[stimulus_index] * int(sampling_freq))
                        spiketimes = np.zeros((len(spikes), math.ceil(frames / dsf)))
                        # print(spiketimes.shape)
                        for trial in range(len(spikes)):
                            # TODO figure out why there are spikes of longer times
                            if len(spikes[trial]) == 0:
                                continue
                            else:
                                spikes_hold = spikes[trial][:bisect.bisect_left(spikes[trial], frames)]
                                # print(list(spikes[trial].astype(int)))
                                # try:
                                spiketimes[trial] = decompress_spikes(list(spikes_hold.astype(int)),
                                                                      (repeat_durations[
                                                                           stimulus_index] * sampling_freq),
                                                                      dsf)
                            # except IndexError:
                            # print(row, stimulus_index, trial)
                            # break

                        gauswins = np.tile(kernel_template(width=kernel_width / dsf)[::-1],
                                           (repeats[stimulus_index], 1))
                        exs = np.zeros((len(spikes), len(spiketimes[1]) - gauswins.shape[1]))
                        exs = signal.oaconvolve(spiketimes, gauswins, mode='valid', axes=1)
                        qi_per_cell[idx] = calc_tradqi(exs)


                    else:
                        frames = int((repeat_durations[stimulus_index] + 0.4) * int(sampling_freq))
                        spiketimes = np.zeros((len(spikes), frames - 1))
                        gauswins = np.tile(kernel_template(width=kernel_width)[::-1], (repeats[stimulus_index], 1))
                        exs = np.zeros((len(spikes), frames - gauswins.shape[1]))
                        for trial in range(len(spikes)):
                            spiketimes = spike_padding_new(spiketimes, spikes, trial)
                        exs = signal.oaconvolve(spiketimes, gauswins, mode='valid', axes=1)
                        qi_per_cell[idx] = calc_tradqi(exs)

                if idx % 1000 == 0:
                    print(row[0][0], qi_per_cell[idx])
    spikes_df['New_qi'] = qi_per_cell

    return spikes_df


def QI_parallel(stimulus_df, spikes_df, ):
    """
    Function to be seen by the 'run_multi' class for implementing parallelised (multiprocessing) workflow.
    """
    df_split = np.array_split(spikes_df, mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    try:
        par_func = partial(calculate_qi, stimulus_df)
        result_df = pd.concat(pool.map(par_func, df_split))
    except Exception:
        traceback.print_exc()
    pool.close()
    pool.join()
    return result_df


class run_multi:
    """
    Class for implementing the multiprocessing(parallelisation) of QI_estimation in Jupyter Notebook.
    """

    def __init__(self):
        print("run_multi initialized")

    def run_multi(self, stimulus_df, spikes_df):
        return QI_parallel(stimulus_df, spikes_df)


def plot_qc_locations_newqi(spikes, savename, color_col="previously_included", invert=False, inverty=False, save=False,
                            color_discrete_map=None):
    """
    Title is self-descriptive, function enables to plot cell_locations on chip. Not reliable for distances etc., takes
    the info from the Centres_x and Centres_y levels of the mIndex
    -spikes dataframe NEEDS to contain "New_qi", "total qc new" and "previously included" as columns. Stimulus- and
    experiment- specific dataframe, others do not make sense to begin with.
    -includes corrections for displaying the REAL chip orientation. If you prefer the default, from the mhenning package,
    leave the invert, inverty arguments as False
    save_name: str, name for the saving
    tosave: bool

    TODO: Consider adding arguments for plot colormap or dimensions internally? Of course some can be done post-assignment
    """
    if not color_discrete_map:
        color_discrete_map = {True: 'rgb(112,112,112)', False: 'rgb(205,92,92)'}
    print('Received', color_discrete_map)

    if invert == False:
        qc_locations_plt = px.scatter(
            spikes,
            x="Centres x",
            y="Centres y",
            color=color_col,
            color_discrete_map={
                '0': 'red',
                '1': 'blue',
                '2': 'green'
            },
            size='Nr of Spikes',
            hover_data=[
                "Cell index",
                "New_qi",
                "Nr of Spikes",

            ],
        )
        qc_locations_plt.update_traces(
            marker=dict(size=3), selector=dict(mode="markers")
        )

        qc_locations_plt.update_xaxes(range=[-10, 2750])
        qc_locations_plt.update_yaxes(range=[-10, 2750])

        qc_locations_plt.update_layout(
            showlegend=False,
            height=1000,
            width=1000,
            title_text="Cell locations quality index and nr of spikes per cluster",

        )

    elif invert == True:
        if inverty == True:
            spikes_temp = spikes.copy()
            spikes_temp['Centres x'] = -spikes_temp['Centres x']
            qc_locations_plt = px.scatter(
                spikes_temp,
                x="Centres y",
                y="Centres x",
                color=color_col,
                color_discrete_sequence=list(color_discrete_map.values()),
                category_orders={color_col: list(color_discrete_map.keys())},
                hover_data=[
                    "Cell index",
                    "New_qi",
                    "Nr of Spikes",
                ],
            )
            qc_locations_plt.update_traces(
                marker=dict(size=12, ), selector=dict(mode="markers"),
            )

            qc_locations_plt.update_xaxes(range=[-10, 2750], showgrid=False, zeroline=False, visible=False,
                                          showticklabels=False)
            qc_locations_plt.update_yaxes(range=[-2750, 10], showgrid=False, zeroline=False, visible=False,
                                          showticklabels=False)

            qc_locations_plt.update_layout(

                height=1000,
                width=1000,
                showlegend=False,
                title_text="Cell locations on Chip - N of cells: %d" % len(spikes),

            )

    if save == True:
        qc_locations_plt.write_image("%s.pdf" % savename, width=1000, height=1000, )
    return qc_locations_plt


def chip_mask_image(spikes, maskname=None, mask=None, color_col='chip_levels', save_name='Random', tosave=False,
                    colormap=None):
    """
    TODO: if spikes df does not contain mask column, make sure to provide it. So arguments "maskname" and "mask are placeholders
    """
    spikes[color_col] = spikes[color_col].astype(str)
    return plot_qc_locations_newqi(spikes.reset_index(), savename=save_name, color_col=color_col,
                                   invert=True,
                                   inverty=True, save=tosave, color_discrete_map=colormap)


def chip_image(threshold_old, threshold_new, spikes, stim_name, fill_na=True, case='inclusive', save_name='Random',
               tosave=False):
    """
    threshold_old:int or float to bottom filter "total qc new" column
    threshold_new:int or float to bottom filter "New_qi" column. Greatly unlikely that it will be above 1
    spikes: dataframe NEEDS to contain "New_qi", "total qc new" as columns
    stim_name:str of the stimulus of interest
    fill_na:If True, it will fill the na in the "New_qi" column with 0 so plotting of the points is possible
    case: inclusive, end_inspection, end_clean, end_exclusive, end_agreement, only_old, only_new,
    save_name: str, name for the saving
    tosave: bool


    Returns
    spikes_thresholded :dataframe with additional bool "previously included" column of values determined by the
    case argument
    plotly_plot: full savename that includes the stimulus name


    TODO: Add documentation about what each case is supposed to represent
    """
    try:
        spikes_filtered = spikes.xs(stim_name, level='Stimulus name')
    except KeyError:
        spikes_filtered = spikes[spikes['Stimulus name'] == stim_name]
    if fill_na:
        spikes_filtered = spikes_filtered.fillna({'New_qi': 0})

    if case == 'inclusive':
        spikes_thres = spikes_filtered[
            (spikes_filtered['total qc new'] > threshold_old) | (spikes_filtered['New_qi'] > threshold_new)]
        spikes_thres['previously_included'] = spikes_thres['total qc new'] > threshold_old
    elif case == 'end_inspection':
        spikes_thres = spikes_filtered[
            spikes_filtered['New_qi'] > threshold_new]
        spikes_thres['previously_included'] = (spikes_filtered['total qc new'] > threshold_old) & (
                spikes_filtered['New_qi'] > threshold_new)
    elif case == 'end_clean':
        spikes_thres = spikes_filtered[
            spikes_filtered['New_qi'] > threshold_new]
        spikes_thres['previously_included'] = [True] * len(spikes_thres)
    elif case == 'only_old':
        spikes_thres = spikes_filtered[
            spikes_filtered['total qc new'] > threshold_old]
        spikes_thres['previously_included'] = (spikes_filtered['total qc new'] > threshold_old)
    elif case == 'only_new':
        spikes_thres = spikes_filtered[
            (spikes_filtered['New_qi'] > threshold_new) & (spikes_filtered['total qc new'] < threshold_old)]
        spikes_thres['previously_included'] = (spikes_filtered['New_qi'] > threshold_new)
    elif case == 'end_exclusive':
        spikes_thres = spikes_filtered[
            (spikes_filtered['total qc new'] > threshold_old) & (spikes_filtered['New_qi'] < threshold_new)]
        spikes_thres['previously_included'] = spikes_thres['total qc new'] > threshold_old
    elif case == 'end_agreement':
        spikes_thres = spikes_filtered[
            (spikes_filtered['total qc new'] > threshold_old) & (spikes_filtered['New_qi'] > threshold_new)]
        spikes_thres['previously_included'] = spikes_thres['total qc new'] > threshold_old

    return spikes_thres, plot_qc_locations_newqi(spikes_thres.reset_index(), savename=save_name + "" + stim_name,
                                                 invert=True,
                                                 inverty=True, save=tosave)
