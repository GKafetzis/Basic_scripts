import traceback
import multiprocessing as mp
from functools import partial

import numpy as np
from MEA_analysis import stimulus_and_spikes as sp
import scipy.signal as signal
import pandas as pd
import plotly.express as px


def kernel_template(width=0.01, sampling_freq=17852.767845719834):
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
    stim_time = np.zeros(len(np.linspace(0, int(4.05 * 6 * sampling_freq),
                                         num=int(4.05 * 6 * sampling_freq), endpoint=False)))

    for sp_times in range(len(spikes[trial_n])):
        stim_time[int(spikes[trial_n][sp_times])] = 1

    return stim_time


def spike_padding_new(stim_time, spikes, trial_n):
    stim_time[trial_n][list(spikes[trial_n].astype(int))] = 1
    return stim_time


def calc_tradqi(kernelfits):
    return np.var(np.mean(kernelfits, 0)) / np.mean(np.var(kernelfits, 0))


def fast_qi(spikes, stimulus_traits, repeat_duration, sampling_freq=17852.767845719834):
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


def fast_qi_exclusive(spikes, stimulus_traits, kernel_width,):
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

def create_full_logbook(stimulus_df):
    nr_stimuli = stimulus_df.index.unique()
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

def calculate_qi(stimulus_df, spikes_df, kernel_width=0.0125):
    ####cell_idces currently not part of the script
    qi_per_cell = np.zeros(len(spikes_df))

    sampling_freq, trigger_completes, repeat_logics, repeats, repeat_durations = create_full_logbook(stimulus_df)
    stimulus_traits = {
        'sampling_freq': sampling_freq
    }



    for row, idx in zip(spikes_df.itertuples(), range(len(spikes_df))):

        stimulus_index = row[0][1]

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

            if sum([len(spikes[i]) for i in range(len(spikes))]) / repeats[stimulus_index] < 30:
                stimulus_traits['repeats'] = repeats[stimulus_index]
                stimulus_traits['repeat_duration'] = repeat_durations[stimulus_index]

                qi_per_cell[idx] = fast_qi_exclusive(spikes, stimulus_traits, kernel_width)
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

    # Quality_df = single_stimulus.calculate_quality_index(stimulus_extr.spikes_stimulus, stimulus_extr.trigger_complete,
    # int(stimulus_extr.stimulus_info["Stimulus_repeat_logic"]),
    # Test_stimulus.sampling_frequency[0])
    return spikes_df

def QI_parallel(stimulus_df, spikes_df, ):
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
    def __init__(self):
        print("run_multi initialized")

    def run_multi(self, stimulus_df, spikes_df):
        return QI_parallel(stimulus_df, spikes_df)


def plot_qc_locations_newqi(spikes, savename, invert=False, inverty=False, save=False, ):
    color_discrete_map = {True: 'rgb(112,112,112)', False: 'rgb(205,92,92)'}

    if invert == False:
        qc_locations_plt = px.scatter(
            spikes,
            x="Centres x",
            y="Centres y",
            color="previously_included",
            color_discrete_map=color_discrete_map,
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
                color="previously_included",
                color_discrete_map=color_discrete_map,
                hover_data=[
                    "Cell index",
                    "New_qi",
                    "Nr of Spikes",
                    "total qc new"
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
                title_text="Cell locations on Chip - N of cells: %d" %len(spikes),

            )

            # qc_locations_plt.update_yaxes(
            #                scaleanchor = "x",
            #                scaleratio = 1,
            #                )
    if save == True:
        qc_locations_plt.write_image("%s.pdf" % savename, width=1000, height=1000,)
    return qc_locations_plt


def chip_image(threshold_old, threshold_new, spikes, stim_name, fill_na=True, case='inclusive', save_name='Random', tosave=False):
    """
    Options for case: inclusive, end_inspection, end_exclusive, end_agreement
    """
    spikes_filtered = spikes.xs(stim_name, level='Stimulus name')
    if fill_na:
        print('On it')
        spikes_filtered= spikes_filtered.fillna({'New_qi': 0})

    if case == 'inclusive':
        spikes_thres = spikes_filtered[
            (spikes_filtered['total qc new'] > threshold_old) | (spikes_filtered['New_qi'] > threshold_new)]
        spikes_thres['previously_included'] = spikes_thres['total qc new'] > threshold_old
    elif case == 'end_inspection':
        spikes_thres = spikes_filtered[
            (spikes_filtered['total qc new'] > threshold_old) | (spikes_filtered['New_qi'] > threshold_new)]
        spikes_thres['previously_included'] = (spikes_filtered['total qc new'] > threshold_old) & (
                    spikes_filtered['New_qi'] > threshold_new)
    elif case == 'end_exclusive':
        spikes_thres = spikes_filtered[
            (spikes_filtered['total qc new'] > threshold_old) & (spikes_filtered['New_qi'] < threshold_new)]
        spikes_thres['previously_included'] = spikes_thres['total qc new'] > threshold_old
    elif case == 'end_agreement':
        spikes_thres = spikes_filtered[
            (spikes_filtered['total qc new'] > threshold_old) & (spikes_filtered['New_qi'] > threshold_new)]
        spikes_thres['previously_included'] = spikes_thres['total qc new'] > threshold_old

    return plot_qc_locations_newqi(spikes_thres.reset_index(), savename=save_name + "" + stim_name, invert=True,
                                   inverty=True, save=tosave)