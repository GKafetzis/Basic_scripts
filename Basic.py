from MEA_analysis import stimulus_and_spikes as sas
from MEA_analysis import Overview
import MEA_analysis
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import pyspike
import json

import itertools


def reorder_elements(arr, index, n):
    temp = [0] * n;

    for i in range(0, n):
        temp[index[i]] = arr[i]

    return temp


def flatten_list(list_in):
    return [val for sublist in list_in for val in sublist]


def create_chip_mask(full_df, arr1, arr2):
    start = full_df.index.get_level_values(0)[0]
    full_mask = np.full(len(full_df), np.nan)
    full_mask[np.union1d(arr1, arr2) - start] = 0
    full_mask[np.intersect1d(arr1, arr2) - start] = 1
    full_mask[np.setdiff1d(arr1, arr2) - start] = 2

    return full_mask


def stim_col_filter(recording, stim_name: str, col_name: str, thresh: float, rec_name=None) -> list:
    if type(recording) == MEA_analysis.Overview.Dataframe:
        df_filtered = recording.get_stimulus_subset(name=stim_name)[0]
    else:
        df_filtered = recording[recording['Stimulus name'] == stim_name]
        if ('Recording' in recording.index.names) and (len(recording.index.get_level_values('Recording').unique()) > 1):
            if not rec_name:
                print(
                    'Running it irrespective of Recording.. Please ensure you provide a rec_name if otherwise intended')
            else:
                df_filtered = df_filtered.loc[(slice(None)), (slice(None)), [rec_name]]
    idces_filter = list(df_filtered[df_filtered[col_name] > thresh].index.get_level_values(0))
    return idces_filter


def get_vals(spikes_df, cell_idx, stim_idces, col_val, ):
    if col_val == 'Spikes':
        len_spikes = [len(spikes_df.loc[[cell_idx], stim_idces, (slice(None))][col_val].values[i].compressed()) for i in
                      range(len(stim_idces))]

        return len_spikes
    else:
        return spikes_df.loc[[cell_idx], stim_idces, (slice(None))][col_val].values


def prepare_dataframes_bystring(recording: MEA_analysis.Overview.Dataframe, bystring: str) -> pd.core.frame.DataFrame:
    dataframe_stimulus = recording.stimulus_df.loc[
        [idx for idx, val in enumerate(recording.stimulus_df['Stimulus_name'].values) if bystring in val]]
    dataframe_spikes = recording.spikes_df
    dataframe_spikes = dataframe_spikes.reset_index()
    dataframe_spikes = dataframe_spikes[np.isin(dataframe_spikes['Stimulus name'], dataframe_stimulus['Stimulus_name'])]

    return dataframe_spikes, dataframe_stimulus


def please():
    return None


def prepare_dataframes(recording: MEA_analysis.Overview.Dataframe, stimuli_indices: list,
                       cell_indices: list == True) -> pd.core.frame.DataFrame:
    dataframe_stimulus = recording.stimulus_df.loc[stimuli_indices]

    dataframe_spikes = recording.spikes_df

    ###THIS NEEDS TO BE USED, TODO
    if len(cell_indices) != 0:
        dataframe_spikes = dataframe_spikes.loc[cell_indices, :, :]

    dataframe_spikes = dataframe_spikes.reset_index()

    dataframe_spikes = dataframe_spikes[np.isin(dataframe_spikes['Stimulus name'], dataframe_stimulus['Stimulus_name'])]

    return dataframe_spikes, dataframe_stimulus


def get_stimulus_traits(dataframe_stimulus: pd.core.frame.DataFrame, stimulus_index: int, sampling_freq=None) -> dict:
    # TO-DO: Maybe include here whether stimulus is homogeneous or not by comparing the phase_dur intervals within repeat
    stimulus_name = dataframe_stimulus['Stimulus_name'].values[stimulus_index]
    stimulus_trig_rel = dataframe_stimulus['Trigger_Fr_relative'].values[stimulus_index]
    stimulus_trials = int(dataframe_stimulus['Stimulus_repeat_logic'].values[stimulus_index])
    stimulus_repeats = int(
        (len(dataframe_stimulus['Trigger_Fr_relative'].values[stimulus_index]) - 1) / stimulus_trials)
    stimulus_subphases = int(dataframe_stimulus['Stimulus_repeat_sublogic'].values[stimulus_index])
    if not sampling_freq:
        sampling_freq = 17852.767845719834

    stimulus_phase_dur = int(np.round((stimulus_trig_rel[1] - stimulus_trig_rel[0]) / sampling_freq, 0))
    stimulus_traits = {
        "stim_name": stimulus_name,
        "stim_rel_trig": stimulus_trig_rel,
        "stim_trials": stimulus_trials,
        "stim_repeats": stimulus_repeats,
        "stim_subphases": stimulus_subphases,
        "stim_phase_dur": stimulus_phase_dur,
        "sampling_freq": sampling_freq
    }

    return stimulus_traits


def get_cell_index_stimulus_matching(cell: int, dataframe_spikes: pd.core.frame.DataFrame,
                                     stimulus_traits: dict) -> int:
    cell_df_idx = dataframe_spikes[(dataframe_spikes['Cell index'] == cell) &
                                   (dataframe_spikes['Stimulus name'] == stimulus_traits['stim_name'])].index[0]

    return cell_df_idx


def get_cell_stimulus_spikes(cell_df_idx: int, dataframe_spikes: pd.core.frame.DataFrame) -> np.ndarray:
    cell_spikes = dataframe_spikes.loc[cell_df_idx]['Spikes'].compressed()

    return cell_spikes


def get_cell_spiketrains_per_stimulus(cell: int, dataframe_spikes: pd.core.frame.DataFrame, stimulus_traits: dict,
                                      phase_dur: int, sampling_freq: float, ) -> list:
    spiketrains_list = []

    cell_df_idx = get_cell_index_stimulus_matching(cell, dataframe_spikes, stimulus_traits)
    cell_spikes = get_cell_stimulus_spikes(cell_df_idx, dataframe_spikes)

    for trial in range(stimulus_traits['stim_trials']):

        spikes_per_trial = (sas.get_spikes_per_trigger_type_new(cell_spikes, stimulus_traits['stim_rel_trig'], trial,
                                                                stimulus_traits['stim_trials'])[0])

        spiketrain_list = []
        for repeat in range(stimulus_traits['stim_repeats']):
            spikes_per_trial[repeat] = spikes_per_trial[repeat] / sampling_freq
            spiketrain_list.append(pyspike.SpikeTrain(spikes_per_trial[repeat], edges=[0, phase_dur]))

        spiketrains_list.append(spiketrain_list)

    return spiketrains_list


def kerberos_spiketrains_per_stimulus(cell: int, dataframe_spikes: pd.core.frame.DataFrame, stimulus_traits: dict,
                                      phase_dur: int, sampling_freq: float, inhomogeneous: bool = False):
    ###TO-DO: Consider estimating if function is homogeneous by comparing windows of phase_dur

    if inhomogeneous:
        _, spiketrains_list = sas.get_spikes_whole_stimulus(
            dataframe_spikes, stimulus_traits['stim_rel_trig'],
            cell,
            stimulus_traits['stim_trials'],
            sampling_freq)
    else:
        spiketrains_list = get_cell_spiketrains_per_stimulus(cell, dataframe_spikes.reset_index(),
                                                             stimulus_traits, phase_dur, sampling_freq)

    return spiketrains_list


def reorder_spiketrains(spiketrains: list, stimulus_traits: dict, pseudorder: list) -> list:
    reordered_spiketrains = []

    pseudorder_sampled = np.reshape(pseudorder,
                                    (stimulus_traits['stim_repeats'], stimulus_traits['stim_trials'])).T.flatten()

    flat_spiketrain = [item for sublist in spiketrains for item in sublist]
    for elem in np.unique(pseudorder_sampled):
        reordered_spiketrains.append(
            list(compress(flat_spiketrain, list(map(lambda x: x == elem, pseudorder_sampled)))))

    return reordered_spiketrains


def plot_raster_only_new(spiketrains, phase_dur, markersize=2):
    raster_plot = plotly.subplots.make_subplots(
        rows=len(spiketrains), cols=2, vertical_spacing=0.005, shared_xaxes=True,
        column_widths=[0.2, 0.3])
    # First, plot spiketrain
    for single_spiketrain in range(len(spiketrains)):
        spiketrain = spiketrains[single_spiketrain]
        for repeat in range(len(spiketrain)):
            spikes_temp = spiketrain[repeat].spikes
            nr_spikes = np.shape(spikes_temp)[0]
            yvalue = np.ones(nr_spikes) * repeat
            raster_plot.add_trace(
                go.Scatter(
                    mode="markers",
                    marker_symbol='line-ns-open',
                    line_width=4,
                    marker_size=3,
                    x=spikes_temp,
                    y=yvalue,
                    name="Repeat " + str(repeat),
                    marker=dict(color="Black", size=markersize),
                ), 1 + single_spiketrain, 1
            )

    raster_plot.update_xaxes(range=[-0.2, phase_dur], title_text="Time in Seconds", row=len(spiketrains), col=1)
    raster_plot.update_yaxes(range=[-0.1, len(spiketrain) + 0.1], visible=False)
    # raster_axis[0].set_title('Spiketrains in Recording')
    return raster_plot


def plot_raster_inhomogeneous(spiketrains, phase_dur, markersize=2):
    raster_plot = plotly.subplots.make_subplots(
        rows=len(spiketrains), cols=2, vertical_spacing=0.005, shared_xaxes=True,
        column_widths=[0.7, 0.1])
    # First, plot spiketrain

    for repeat in range(len(spiketrains)):
        spikes_temp = spiketrains[repeat].spikes
        nr_spikes = np.shape(spikes_temp)[0]
        yvalue = np.ones(nr_spikes) * repeat
        raster_plot.add_trace(
            go.Scatter(
                mode="markers",
                marker_symbol='line-ns-open',
                line_width=4,
                marker_size=3,
                x=spikes_temp,
                y=yvalue,
                name="Repeat " + str(repeat),
                marker=dict(color="Black", size=markersize),
            ), 1 + repeat, 1
        )

    raster_plot.update_xaxes(range=[-0.2, phase_dur], title_text="Time in Seconds", row=len(spiketrains), col=1)
    raster_plot.update_yaxes(range=[-0.1, len(spiketrains) + 0.1], visible=False)
    # raster_axis[0].set_title('Spiketrains in Recording')
    return raster_plot


def plot_sc_aligned_new(cell: int, spiketrains_list: list, stimulus_traits: pd.core.frame.DataFrame, phase_dur: int,
                        colors: list = ['#FE7C7C', '#FAFE7C', '#8AFE7C', '#7CFCFE', '#7C86FE',
                                        '#FE7CFE']) -> plotly.graph_objs._figure.Figure:
    """
    To do: add save option and title for stimulus that played
    """
    # reload(spike_plotly)
    if 'hirp' in stimulus_traits["stim_name"]:
        figure = plot_raster_inhomogeneous(spiketrains_list, phase_dur, )
    else:
        figure = plot_raster_only_new(spiketrains_list, phase_dur, )

        if 'FFF' in stimulus_traits["stim_name"]:

            for trial in range(stimulus_traits["stim_trials"]):
                figure.add_shape(dict(type="rect",
                                      x0=0, y0=0, x1=phase_dur / 2, y1=stimulus_traits["stim_repeats"],
                                      line_color=colors[trial], fillcolor=colors[trial], line_width=1, opacity=0.2,
                                      layer="below", ), row=trial + 1, col=1)

            figure.for_each_xaxis(lambda x: x.update(showgrid=False))
            figure.for_each_yaxis(lambda x: x.update(showgrid=False))

        elif 'MB' in stimulus_traits["stim_name"]:
            figure = update_plot_withdirections(figure)
            figure.for_each_xaxis(lambda x: x.update(range=[-0.8, phase_dur]))

    figure.update_layout(title='Cell %d - Stimulus %s' % (cell, stimulus_traits["stim_name"]),
                         showlegend=False)

    return figure


def plot_sc_aligned_new_new(dfr_mIdx: tuple, overview_dfr: pd.core.frame.DataFrame, phase_dur: int,
                            colors: list) -> plotly.graph_objs._figure.Figure:
    cell = dfr_mIdx[0]
    recording_name = dfr_mIdx[2]
    stimulus_name = overview_dfr.spikes_df.loc[dfr_mIdx]['Stimulus name']
    spiketrains_list = overview_dfr.spikes_df.loc[dfr_mIdx]['Spiketrains']

    figure = plot_raster_only_new(spiketrains_list, phase_dur, )

    for trial in range(len(spiketrains_list)):
        figure.add_shape(dict(type="rect",
                              x0=0, y0=0, x1=phase_dur / 2, y1=len(spiketrains_list[0]),
                              line_color=colors[trial], fillcolor=colors[trial], line_width=1, opacity=0.2,
                              layer="below", ), row=trial + 1, col=1)

    figure.for_each_xaxis(lambda x: x.update(showgrid=False))
    figure.for_each_yaxis(lambda x: x.update(showgrid=False))
    figure.update_layout(title='Cell %d - Stimulus %s - Recording %s' % (cell, stimulus_name, recording_name))

    return figure


def plot_sc_aligned_new_workingon(dfr_mIdx: tuple, overview_dfr: pd.core.frame.DataFrame, phase_dur: int, colors: list,
                                  bystimulus='index', filter_val=None, ) -> plotly.graph_objs._figure.Figure:
    """
    Provide first input a tuple of cell_index & recording name
    """
    figures = []
    assert filter_val is not None

    def plot_sc_aligned_byindex(dfr, stim_index):
        stimulus_name = get_columnval_fromdfr(overview_dfr, (cell, stim_index, recording_name), 'Stimulus name')
        spiketrains_list = get_columnval_fromdfr(overview_dfr, (cell, stim_index, recording_name), 'Spiketrains')

        if type(spiketrains_list) == float:
            print('No spiketrains found for %d %s . Please ensure you have computed them' % (stim_index, stimulus_name))
            figure = np.nan
        else:
            figure = plotting_aligned(spiketrains_list, phase_dur, colors)
            figure.update_layout(title='Cell %d - Stimulus %s - Recording %s' % (cell, stimulus_name, recording_name))
            figure.show()

        return figure

    cell = dfr_mIdx[0]
    recording_name = dfr_mIdx[1]
    if bystimulus == 'index':
        print('Stimulus index expected as a list')
        for stim_index in filter_val:
            figure = plot_sc_aligned_byindex(overview_dfr, stim_index)
            figures.append(figure)
    elif bystimulus == 'name':
        print('Stimulus names expected as a list')
        # for stim_name in filter_val:
        dfr_temp = overview_dfr
        # dfr_temp= overview_dfr.get_stimulus_subset(name=stim_name)[0]
        for stim_index in np.unique(dfr_temp.index.get_level_values(level=1)):
            figure = plot_sc_aligned_byindex(dfr_temp, stim_index)
            figures.append(figure)

    return figures


def plotting_aligned(spiketrains_list, phase_dur, colors):
    figure = plot_raster_only_new(spiketrains_list, phase_dur, )

    for trial in range(len(spiketrains_list)):
        figure.add_shape(dict(type="rect",
                              x0=0, y0=0, x1=phase_dur / 2, y1=len(spiketrains_list[0]),
                              line_color=colors[trial], fillcolor=colors[trial], line_width=1, opacity=0.2,
                              layer="below", ), row=trial + 1, col=1)

    figure.for_each_xaxis(lambda x: x.update(showgrid=False))
    figure.for_each_yaxis(lambda x: x.update(showgrid=False))

    return figure


def get_columnval_fromdfr(dfr, mIdx: tuple, col_name: 'str'):
    return dfr.loc[mIdx][col_name]


def plot_c_spiketrain_per_stimulus_new(cell_idx: int, arrays_select: list, df_spikes: pd.core.frame.DataFrame,
                                       df_stimulus: pd.core.frame.DataFrame,
                                       phase_dur: int, sampling_freq: float, colors_box: list,
                                       toappend: bool) -> plotly.graph_objs._figure.Figure:
    """
    TODO: add functionalities for other stimuli, e.g. MB
    """
    figure = []
    for stimulus in range(len(df_stimulus)):
        stimulus_traits = get_stimulus_traits(df_stimulus, stimulus)

        # if 'BW' in stimulus_traits['stim_name']:
        # stimulus_traits['stim_trials']= 1
        # stimulus_traits['stim_repeats']= 2*stimulus_traits['stim_repeats']
        # colors_box=['#FFFFFF']
        if 'green' in stimulus_traits['stim_name']:
            colors_box = ['#052c00', '#0c6701', '#8afe7c', '#12a101', '#0e7a01', '#052c00']
        else:
            colors_box = ['#fe7c7c', '#fafe7c', '#8afe7c', '#7cfcfe', '#7c86fe', '#fe7cfe']

        spiketrains_list = get_cell_spiketrains_per_stimulus(cell_idx, df_spikes, stimulus_traits, phase_dur,
                                                             sampling_freq)

        if 'pseudo' in stimulus_traits['stim_name']:
            spiketrains_list = reorder_spiketrains(spiketrains_list, pseudorder)
        else:
            pass

        if '4' in stimulus_traits['stim_name']:
            fig = plot_sc_aligned(cell_idx, spiketrains_list, stimulus_traits, 2 * phase_dur,
                                  stimulus_traits['stim_repeats'], colors_box)
        else:
            fig = plot_sc_aligned_new(cell_idx, spiketrains_list, stimulus_traits, phase_dur,
                                      stimulus_traits['stim_repeats'], colors_box)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)

        fig.show()

        if toappend:
            figure.append(fig)

    return figure


def calculate_psth_trace(spikes_df, to_normalise=False, normalising_array=None):
    histogram_column = spikes_df.loc[:, "PSTH"]
    histograms = histogram_column.values
    histogram_arr = np.zeros((len(spikes_df), np.shape(histograms[0])[0]))
    bins_column = spikes_df.loc[:, "PSTH_x"]
    bins = bins_column.values
    bins_x = bins[0]
    nr_cells = np.shape(histograms)[0]
    cell_indices = np.linspace(0, nr_cells - 1, nr_cells)

    for cell in range(nr_cells):
        if np.max(histograms[cell]) == 0:
            histogram_arr[cell, :] = 0
        else:
            try:
                histogram_arr[cell, :] = histograms[cell] / np.max(histograms[cell]) if not to_normalise else \
                    histograms[cell] / normalising_array[cell]
            except ValueError:
                extra_bins = np.shape(histograms[cell])[0] - np.shape(histogram_arr)[0]
                histogram_arr_temp = histograms[cell] / np.max(histograms[cell])
                try:
                    histogram_arr[cell, :] = histogram_arr_temp[:-extra_bins]
                except ValueError:
                    histogram_arr[cell, :] = 0

    return bins_x, histogram_arr


def plot_heatmap_adj(spike_df, stimulus_df, stimulus_index: int, mid_dim: int = 1.2, to_normalise=False,
                     normalising_array=None,
                     scale_to=None, save=False, savename=None):
    """
    spikes_df needs to include PSTH and PSTH_x values
    TO-DO: implement normalisation of spikes across different stimuli for same cell.
    """
    stimulus_traits = get_stimulus_traits(stimulus_df, stimulus_index)
    stimulus_trace = xyplot_stim_steps(stimulus_traits["stim_trials"], stimulus_traits["stim_phase_dur"],
                                       yrange=[0.98, 1.08])
    stimulus_trace = [stimulus_trace[0][:-1], stimulus_trace[1]]

    spikes_df = spike_df.loc[slice(None), ["Stimulus ID", stimulus_index], :]

    bins, histogram_arr = calculate_psth_trace(spikes_df, to_normalise, normalising_array)
    cell_indices = np.linspace(0, len(spikes_df) - 1, len(spikes_df))

    histogram_fig = plotly.subplots.make_subplots(
        rows=3,
        cols=1,
        row_width=[0.2, mid_dim, 0.4],
        vertical_spacing=0.01,
        shared_xaxes=True,
    )
    histogram_fig.add_trace(
        go.Scatter(
            x=bins,
            y=(np.mean(histogram_arr, axis=0) / np.max(np.mean(histogram_arr, axis=0))) * len(spikes_df),
            # y=val_scaled_rel(np.mean(histogram_arr, axis=0), traces[1]),
            mode="lines",
            name="Average PSTH",
            line=dict(color="#000000"),

        ),
        row=1,
        col=1,
    )

    histogram_fig.add_trace(
        go.Heatmap(
            x=bins,
            y=cell_indices,
            z=histogram_arr,
            zsmooth=False,
            colorscale=[
                [0, "rgb(255, 255, 255)"],  # 0
                [0.1, "rgb(200, 200, 200)"],  # 10
                [0.2, "rgb(170, 170, 170)"],  # 100
                [0.3, "rgb(150, 150, 150)"],
                [0.4, "rgb(100, 100, 100)"],
                [0.5, "rgb(100, 100, 100)"],  # 1000
                [0.6, "rgb(50, 50, 50)"],
                [0.7, "rgb(50, 50, 50)"],
                [0.8, "rgb(5, 5, 5)"],
                [0.9, "rgb(5, 5, 5)"],  # 10000
                [1.0, "rgb(0, 0, 0)"],  # 100000
            ], colorbar=dict(bgcolor='white', bordercolor='white'),
        ),
        row=2,
        col=1,
    )
    histogram_fig.update_traces(showscale=False, selector=dict(type="heatmap"), row=2, col=1)

    histogram_fig.add_trace(
        go.Scatter(x=stimulus_trace[0], y=stimulus_trace[1], mode="lines", marker_color='black'),
        row=3,
        col=1,
    )

    if stimulus_traits["stim_name"] == 'FFF':
        ##TO-DO: Create a separate function, where the templates will be stored, so you can include arbitrary colors

        colors: list = ['#FE7C7C', '#FAFE7C', '#8AFE7C', '#7CFCFE', '#7C86FE', '#FE7CFE']

        for trial in range(stimulus_traits["stim_trials"]):
            histogram_fig.add_shape(dict(type="rect",
                                         x0=0 + (trial * stimulus_traits["stim_phase_dur"]),
                                         y0=-0.5,
                                         x1=stimulus_traits["stim_phase_dur"] / stimulus_traits["stim_subphases"] +
                                            (trial * stimulus_traits["stim_phase_dur"]),
                                         y1=len(spikes_df) - 0.5,
                                         line_color=colors[trial], fillcolor=colors[trial], line_width=1, opacity=0.2,
                                         ), row=2, col=1)

    if scale_to is None:
        y_range = [-.1, len(spikes_df) + 0.8]
    else:
        y_range = [-.1, len(scale_to) + 0.8]

    histogram_fig.update_xaxes(showticklabels=False, row=1, col=1)
    histogram_fig.update_yaxes(showticklabels=False, row=3, col=1)
    histogram_fig.update_yaxes(range=y_range, showticklabels=False, row=1, col=1)
    histogram_fig.update_xaxes(showticklabels=False, row=2, col=1)
    histogram_fig.update_yaxes(range=y_range, showticklabels=False, row=2, col=1)
    histogram_fig.update_xaxes(title_text="Time in Seconds", row=3, col=1)

    histogram_fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)"},
                                height=1000,
                                width=1280,
                                showlegend=False)

    if save == True:
        histogram_fig.write_image("%s.pdf" % savename, width=1980 * 2, height=1080 * 2, )

    return histogram_fig


def xyplot_stim_steps(ntrials: int, phase_dur: int, yrange: list) -> list:
    ys = [yrange[0], yrange[1], yrange[1], yrange[0]] * (ntrials + 1)
    x_template = np.arange(phase_dur * ntrials + 2)[::int(phase_dur / 2)]
    k = 2
    listOfLists = [list(itertools.repeat(element, k)) for element in x_template]
    xs = list(itertools.chain.from_iterable(listOfLists))

    return xs, ys


def create_RGCtypes():
    RGC_categories = ['ON', 'OFF', 'ONOFF', 'unclear']
    ONs = []
    OFFs = []
    ONOFFs = []
    unclears = []
    RGCtypes = [ONs, OFFs, ONOFFs, unclears]

    return RGCtypes


def f(cell, save):
    if 'RGCtypes' not in globals():
        create_RGCtypes()
    else:
        pass

    def y(On, Off, OnOff, ONOFFswitch, unclear):
        alignment = [On, Off, OnOff, ONOFFswitch, unclear]

        if len(np.where(alignment)[0]) != 0:
            if int(cell) in RGCtypes[np.where(alignment)[0][0]]:
                print('You have previously classified this cell')
            else:
                RGCtypes[np.where(alignment)[0][0]].append(int(cell))
                print('Cell has been classified!')

        else:
            print('Cell is not classified yet')

    interact(y, On=False, Off=False, OnOff=False, ONOFFswitch=False, unclear=False, )

    figure = plot_c_spiketrain_per_stimulus_new(int(cell), arrays_selectmg2, df_spikes, df_stimulus,
                                                4, Sampling_freq, colors, toappend=save)

    if save:
        for i in range(len(figure)):
            figure[i].write_image("poster_figures/forTom/%s.svg" % figure[i].layout['title']['text'], width=900,
                                  height=600, scale=4)


def run_interact(cell_array):
    return interact(f, save=False, cell=[str(i) for i in arrays_selectmg2], )


def create_RGC_dictionary(RGC_categories, RGCtypes):
    return (dict([[RGC_categories[i], RGCtypes[i]] for i in range(len(RGCtypes))]))


def save_RGC_dictionary(RGC_dict: dict, savename: str):
    lesn = 0
    for i in range(len(RGCtypes)):
        lesn += len(RGCtypes[i])
    RGC_dict['tnumber'] = lesn

    with open(savename + '.json', 'w') as outfile:
        outfile.write(json.dumps(RGC_dict))

    print('Saving was successful')


def open_RGC_dictionary(data_directory: str, ):
    with open(data_directory) as json_file:
        data_categ = json.load(json_file)

    return data_categ


def flatten_RGC_types(RGC_dict: dict) -> list:
    print('flattened_array and individual bounds are returned')
    full_array = [RGC_dict['ON'], RGC_dict['OFF'], RGC_dict['ONOFF']]
    full_array = [item for sublist in full_array for item in sublist]

    ON_bounds = slice(0, len(RGC_dict['ON']))
    OFF_bounds = slice(ON_bounds.stop, ON_bounds.stop + len(RGC_dict['OFF']))
    ONOFF_bounds = slice(OFF_bounds.stop, len(full_array))
    bounds = [ON_bounds, OFF_bounds, ONOFF_bounds]

    return full_array, bounds


#################################################################################################
def helper_filter(val, x, sign='smaller', ):
    if sign == 'smaller':
        return val < x
    elif sign == 'bigger':
        return val > x


def calculate_PSI(x):
    """
    Calculates skew index
    """
    return 3 * (np.nanmean(x) - np.nanmedian(x)) / np.nanstd(x)


def calculate_inwindow(x, bounds: list):
    return sum((x > bounds[0]) & (x < bounds[1])) / len(x)


def proceed(f, args):
    val = input('Do you wish to proceed with defaults? (yes/no)')
    if val == 'yes':
        return f(*args)
    else:
        print('stopping here, back to user')
        return None


def create_color_book(lob: list, colours: list):
    assert len(lob) == len(colours)
    color_book = []
    for idx, l in enumerate(lob):
        color_book.append((l.stop - l.start) * [colours[idx]])

    return [item for sublist in color_book for item in sublist]


def update_plot_withdirections(plot_handle):
    plot_handle['layout'].update(
        annotations=[
            dict(
                x=-0.01, y=1.5,  # annotation point
                xref='x1',
                yref='y1',
                showarrow=True,
                arrowhead=3,
                ax=-25,
                ay=0.4,
            ),

            dict(
                x=-0.012, y=3,  # annotation point
                xref='x1',
                yref='y3',
                showarrow=True,
                arrowhead=3,
                ax=-20,
                ay=12.54,
            ),

            dict(
                x=-0.35, y=3.5,  # annotation point
                xref='x1',
                yref='y5',
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=30,
            ),

            dict(
                x=-0.69, y=3,  # annotation point
                xref='x1',
                yref='y7',
                showarrow=True,
                arrowhead=3,
                ax=20,
                ay=12.54,
            ),

            dict(
                x=-0.68, y=1.5,  # annotation point
                xref='x1',
                yref='y9',
                showarrow=True,
                arrowhead=3,
                ax=25,
                ay=0,
            ),
            dict(
                x=-0.68, y=0,  # annotation point
                xref='x1',
                yref='y11',
                showarrow=True,
                arrowhead=3,
                ax=25,
                ay=-12.54,
            ),

            dict(
                x=-0.35, y=0,  # annotation point
                xref='x1',
                yref='y13',
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-30,
            ),

            dict(
                x=-0.01, y=0.4,  # annotation point
                xref='x1',
                yref='y15',
                showarrow=True,
                arrowhead=3,
                ax=-25,
                ay=-15,
            ),
        ])

    plot_handle.for_each_xaxis(lambda x: x.update(showgrid=False))
    plot_handle.for_each_yaxis(lambda x: x.update(showgrid=False))

    return (plot_handle)
