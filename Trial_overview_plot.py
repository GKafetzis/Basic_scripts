from MEA_analysis import stimulus_and_spikes as sas
import pyspike
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go


def stimulus_paras(df_temp, stimulus_df, phase_dur=4):
    results = np.empty((len(df_temp), 1), object)
    results_fit= np.empty((len(df_temp),1), object)
    for row, idx in zip(df_temp.itertuples(), range(len(df_temp))):


        stimulus_index = row[0][1]
        recording_name = row[0][2]

        # Extract the right row from the stimulus dataframe
        trigger_complete = stimulus_df.loc[stimulus_index, recording_name][
            "Trigger_Fr_relative"
        ]
        repeat_logic = int(stimulus_df.loc[stimulus_index, recording_name][
            "Stimulus_repeat_logic"
        ])
        stimulus_repeats = int(
            (len(stimulus_df.loc[stimulus_index, recording_name][
            "Trigger_Fr_relative"
        ]) - 1)
            / repeat_logic)

        try:
            sampling_freq = stimulus_df.loc[stimulus_index, recording_name][
                "Sampling_Freq"
            ]
            phase_dur = int(np.mean(stimulus_df.loc[stimulus_index, recording_name][
                "Trigger_int"
            ])/sampling_freq)
        except KeyError:
            print('not possible')
            pass
        if idx == 0 or idx == 1000:
            print(trigger_complete)


        spiketrains_list = []
        for trial in range(repeat_logic):

            spikes_per_trial = sas.get_spikes_per_trigger_type_new(
                row.Spikes.compressed(),
                trigger_complete,
                trial,
                repeat_logic,
            )[0]

            spiketrain_list = []
            for repeat in range(stimulus_repeats):
                spikes_per_trial[repeat] = spikes_per_trial[repeat] / sampling_freq
                spiketrain_list.append(
                    pyspike.SpikeTrain(spikes_per_trial[repeat], edges=[0, phase_dur])
                )

            spiketrains_list.append(spiketrain_list)
        results[idx,0] = spiketrains_list
        results_fit[idx,0]=smooth_spiketrains(spiketrains_list, phase_dur)

    df_temp["Spiketrains"] = results[:,0]
    df_temp["PSTH_fit"]= results_fit[:,0]
    print('Finished')

    return df_temp

def psth_fit(df_temp, phase_dur=4):
    """
    Deprecated
    """
    results = np.empty((len(df_temp), 1), object)
    for row, idx in zip(df_temp.itertuples(), range(len(df_temp))):
        results[idx, 0]=smooth_spiketrains(row.Spiketrains, phase_dur)
    df_temp["PSTH_fit"]= results[:,0]

    return df_temp

#################################################################################################################################################
def investigate(overview_df, stimulus="FFF", recording=None, cell_idx=[]):
    df_temp, stimulus_df = overview_df.get_stimulus_subset(name=stimulus)
    df_temp = df_temp.loc[cell_idx,slice(None),recording]

    return(stimulus_paras(df_temp, stimulus_df))

def compute_fits(overview_df, stimulus='FFF', recording=None, cell_idx=[]):
    df_temp, stimulus_df = overview_df.get_stimulus_subset(name=stimulus)
    df_temp = df_temp.loc[cell_idx,slice(None),recording]

    return(psth_fit(df_temp))


def flatten_list(unflattened_list:list):
    return [item for sublist in unflattened_list for item in sublist]

def smooth_spiketrains(spiketrains:list, phase_dur:int):
    assert type(spiketrains)==list
    smoothed_spiketrains=[]
    for trial in range(len(spiketrains)):
        spike_list=[]
        for repeat in range(len(spiketrains[trial])):
            spike_list.append(spiketrains[trial][repeat].spikes)
        smoothed=pd.DataFrame({'hist': np.histogram(flatten_list(spike_list), bins=100, range=[0,phase_dur])[0]})
        smoothed=smoothed.ewm(com=.7).mean()
        smoothed_spiketrains.append(smoothed.values.flatten().tolist())
    return smoothed_spiketrains

def compute_bincenters(array, dur):
    _, bin_edges=np.histogram(0, bins=len(array), range=[0, dur])
    return ((bin_edges[:-1]+bin_edges[1:])/2)



def plot_raster_only_new(spiketrains, phase_dur, markersize=2, delays=None):

    raster_plot = plotly.subplots.make_subplots(
        rows=len(spiketrains), cols=2, vertical_spacing = 0.005, shared_xaxes=True,
    column_widths=[0.2,0.3])
    # First, plot spiketrain
    for single_spiketrain in range(len(spiketrains)):
        spiketrain=spiketrains[single_spiketrain]
        for repeat in range(len(spiketrain)):
            spikes_temp = spiketrain[repeat].spikes if delays is None else spiketrain[repeat].spikes-delays[single_spiketrain]
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
                ), 1+single_spiketrain, 1
            )



    raster_plot.update_xaxes(range=[-0.2, phase_dur], title_text="Time in Seconds", row=len(spiketrains), col=1)
    raster_plot.update_yaxes(range=[-0.1, len(spiketrains) +0.1],visible=False)
    # raster_axis[0].set_title('Spiketrains in Recording')
    return raster_plot

def plot_sc_aligned_new_workingon(dfr_mIdx:tuple, overview_dfr:pd.core.frame.DataFrame, phase_dur:int, colors:list, bystimulus='index', filter_val=None, )-> plotly.graph_objs._figure.Figure:
    """
    Provide first input a tuple of cell_index & recording name
    """
    figures=[]
    assert filter_val is not None



    def plot_sc_aligned_byindex(dfr, stim_index):
        stimulus_name=get_columnval_fromdfr(overview_dfr, (cell, stim_index, recording_name), 'Stimulus name')
        spiketrains_list= get_columnval_fromdfr(overview_dfr, (cell, stim_index, recording_name), 'Spiketrains')

        if type(spiketrains_list)==float:
            print('No spiketrains found for %d %s . Please ensure you have computed them'%(stim_index, stimulus_name))
            figure=np.nan
        else:
            figure= plotting_aligned(spiketrains_list, phase_dur, colors)
            figure.update_layout(title='Cell %d - Stimulus %s - Recording %s' %(cell, stimulus_name, recording_name))
            figure.show()

        return figure


    cell=dfr_mIdx[0]
    recording_name= dfr_mIdx[1]
    if bystimulus=='index':
        print('Stimulus index expected as a list')
        for stim_index in filter_val:
            figure=plot_sc_aligned_byindex(overview_dfr, stim_index)
            figures.append(figure)
    elif bystimulus=='name':
        print('Stimulus names expected as a list')
        #for stim_name in filter_val:
        dfr_temp=overview_dfr
        #dfr_temp= overview_dfr.get_stimulus_subset(name=stim_name)[0]
        for stim_index in np.unique(dfr_temp.index.get_level_values(level=1)):
            figure=plot_sc_aligned_byindex(dfr_temp, stim_index)
            figures.append(figure)

    return figures

def plotting_aligned(spiketrains_list, phase_dur, colors):
    figure=  plot_raster_only_new(spiketrains_list, phase_dur,)

    for trial in range(len(spiketrains_list)):
        figure.add_shape(dict(type="rect",
        x0=0, y0=0, x1=phase_dur/2, y1=len(spiketrains_list[0]),
        line_color=colors[trial], fillcolor=colors[trial], line_width=1,  opacity=0.2,
    layer="below",),row=trial+1, col=1)


    figure.for_each_xaxis(lambda x: x.update(showgrid=False))
    figure.for_each_yaxis(lambda x: x.update(showgrid=False))

    return figure
def get_columnval_fromdfr(dfr, mIdx:tuple, col_name:'str'):
    return dfr.loc[mIdx][col_name]
