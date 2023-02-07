from Basic_scripts import Basic
from MEA_analysis import stimulus_and_spikes as sas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings

def create_multiIndex(dimensions:list, names:list) -> pd.core.indexes.multi.MultiIndex:

    if len(dimensions)!=len(names):
        print('Dimensions do not match')
    else:
        print('Created multiIndex')

        arrays=([[i for i in range(dimensions[k])] for k in range(len(dimensions))])

        mIndex=pd.MultiIndex.from_product([arrays[k] for k in range(len(dimensions))], names=[names[k] for k in range(len(names))])

    return(mIndex)

def init_mI_Dataframe(n_columns:int, names_columns:list, mIndex) -> pd.core.frame.DataFrame:

    return(pd.DataFrame(data=np.zeros([mIndex.shape[0], n_columns]), columns=names_columns, index=mIndex))


def get_first_spike_per_phase(cell:int, dataframe_spikes:pd.core.frame.DataFrame, stimulus_traits:dict, phase_dur:int, sampling_freq:float, ON_thres:float) -> list:

    first_spikes_s_overview= np.zeros([stimulus_traits['stim_trials'], stimulus_traits['stim_repeats']])
    first_spikes_s_overview_OFF= np.zeros([stimulus_traits['stim_trials'], stimulus_traits['stim_repeats']])

    cell_df_idx= Basic.get_cell_index_stimulus_matching(cell, dataframe_spikes, stimulus_traits)
    cell_spikes= Basic.get_cell_stimulus_spikes(cell_df_idx, dataframe_spikes)

    step_dur= phase_dur/2



    for trial in range(stimulus_traits['stim_trials']):

        first_spikes_s= np.zeros(stimulus_traits['stim_repeats'], dtype= np.float64)
        first_spikes_s_OFF= np.zeros(stimulus_traits['stim_repeats'], dtype= np.float64)





        spikes_per_trial= (sas.get_spikes_per_trigger_type_new(cell_spikes, stimulus_traits['stim_rel_trig'], trial, stimulus_traits['stim_trials'])[0])


        for repeat in range(stimulus_traits['stim_repeats']):

                try:

                    spikes_per_trial[repeat][0]/sampling_freq

                    if spikes_per_trial[repeat][0]/sampling_freq<ON_thres:

                        first_spikes_s[repeat]=spikes_per_trial[repeat][0]/sampling_freq




                    else:

                        first_spikes_s[repeat]=np.nan


                except:

                    first_spikes_s[repeat]=np.nan


                try:
                    next(i for i, v in enumerate(spikes_per_trial[repeat]) if v > ON_thres*sampling_freq)

                    first_spikes_s_OFF[repeat]= spikes_per_trial[repeat][next(i for i, v in enumerate(spikes_per_trial[repeat]) if v > ON_thres*sampling_freq)]/sampling_freq - step_dur


                except:

                    first_spikes_s_OFF[repeat]=np.nan



        first_spikes_s_overview[trial]=first_spikes_s


        first_spikes_s_overview_OFF[trial]=first_spikes_s_OFF


    return [first_spikes_s_overview, first_spikes_s_overview_OFF]


def get_number_of_spikes_per_phase(cell:int, dataframe_spikes:pd.core.frame.DataFrame, stimulus_traits:dict, phase_dur:int, sampling_freq:float, ON_thres:float) -> list:

    """
    Could become problematic if light_step of too short duration
    """
    nr_of_spikes_overview=np.zeros([stimulus_traits['stim_trials'], stimulus_traits['stim_repeats']])


    nr_of_spikes_overview_OFF= np.zeros([stimulus_traits['stim_trials'], stimulus_traits['stim_repeats']])

    cell_df_idx= Basic.get_cell_index_stimulus_matching(cell, dataframe_spikes, stimulus_traits)
    cell_spikes= Basic.get_cell_stimulus_spikes(cell_df_idx, dataframe_spikes)

    step_dur= phase_dur/2



    for trial in range(stimulus_traits['stim_trials']):


        nr_of_spikes= np.zeros(stimulus_traits['stim_repeats'], dtype=np.float64)
        nr_of_spikes_OFF= np.zeros(stimulus_traits['stim_repeats'], dtype=np.float64)

        spikes_per_trial= (sas.get_spikes_per_trigger_type_new(cell_spikes, stimulus_traits['stim_rel_trig'], trial, stimulus_traits['stim_trials'])[0])



        for repeat in range(stimulus_traits['stim_repeats']):
            try:

                next(i for i, v in enumerate(spikes_per_trial[repeat]) if v > ON_thres*sampling_freq)


                nr_of_spikes[repeat]=next(k for k, v in enumerate(spikes_per_trial[repeat]) if v > ON_thres*sampling_freq)
                nr_of_spikes_OFF[repeat]= len(spikes_per_trial[repeat])-nr_of_spikes[repeat]

            except:

                nr_of_spikes_OFF[repeat]=0
                nr_of_spikes[repeat]= len(spikes_per_trial[repeat])


        nr_of_spikes_overview[trial]= nr_of_spikes


        nr_of_spikes_overview_OFF[trial]= nr_of_spikes_OFF

    return [nr_of_spikes_overview, nr_of_spikes_overview_OFF]

def apply_ON_correction(cell:int, dataframe_spikes:pd.core.frame.DataFrame, first_spike:list, nspikes:list, stimulus_traits:dict, ON_thres:int, sampling_freq:float, ) -> list:

    """
    Currently deprecated
    """

    dummy=np.arange(0,10)
    reference=np.nanmedian(first_spike[1])

    to_replace= [dummy[[first_spike[trial][repeat]-0.75*reference<0 for repeat in range(stimulus_traits['stim_repeats'])]] for trial in range(stimulus_traits['stim_trials'])]

    for trial in range(stimulus_traits['stim_trials']):

        if len(to_replace[trial])==0:
            pass

        else:

            cell_df_idx= get_cell_index_stimulus_matching(cell, dataframe_spikes, stimulus_traits)
            cell_spikes= get_cell_stimulus_spikes(cell_df_idx, dataframe_spikes)


            spikes_per_trial= (sas.get_spikes_per_trigger_type_new(cell_spikes, stimulus_traits['stim_rel_trig'], trial, stimulus_traits['stim_trials'])[0])


            for repeat in to_replace[trial]:
                try:

                    spikes_per_trial[repeat][1]/sampling_freq

                    if spikes_per_trial[repeat][1]/sampling_freq<ON_thres:
                        spike_sub=spikes_per_trial[repeat][1]/sampling_freq


                    else:
                        spike_sub=np.nan

                except:
                    spike_sub=np.nan

                print('Spike too early, replacing with', spike_sub)
                first_spike[trial][repeat]=spike_sub
                nspikes[trial][repeat]-=1

    return first_spike, nspikes, to_replace

def apply_ON_correction_spiketrain(spiketrain:list, first_spike:list, nspikes:list, stimulus_traits:dict, ON_thres:int, sampling_freq:float,) -> list:

    """
    Currently deprecated
    """

    dummy=np.arange(0,10)
    reference=np.nanmedian(first_spike[1])

    to_replace= [dummy[[first_spike[trial][repeat]-0.75*reference<0 for repeat in range(stimulus_traits['stim_repeats'])]] for trial in range(stimulus_traits['stim_trials'])]



    for trial in range(stimulus_traits['stim_trials']):

        if len(to_replace[trial])==0:
            pass


        else:


            for repeat in to_replace[trial]:



                spikes_per_trial= spiketrain[trial][repeat].spikes
                try:

                    spikes_per_trial[1]/sampling_freq

                    if spikes_per_trial[1]/sampling_freq<ON_thres:
                        spike_sub=spikes_per_trial[1]




                    else:
                        spike_sub=np.nan



                except:
                    spike_sub=np.nan

                print('Spike too early, replacing with', spike_sub)

                first_spike[trial][repeat]=spike_sub
                nspikes[trial][repeat]-=1
                spiketrain[trial][repeat].spikes= spiketrain[trial][repeat].spikes[1:]

    return first_spike, nspikes, spiketrain, to_replace


def calculate_moments(nmoments:int, stimulus_traits:dict, first_spike:list, nspikes:list) -> list:

    moments_overview=[]
    for phase in range(len(first_spike)):

        moments_overview_temp= np.zeros([nmoments, stimulus_traits['stim_trials']])

        for trial in range(stimulus_traits['stim_trials']):

            moments_overview_temp[0, trial]= (np.count_nonzero(~np.isnan(first_spike[phase][trial]))/stimulus_traits['stim_repeats'])    #reliability
            moments_overview_temp[1, trial]= (np.nanstd(first_spike[phase][trial]))#[~np.isnan(first_spike[phase][trial])]))                   #precision
            moments_overview_temp[2, trial]= (np.nanmean(first_spike[phase][trial]))#[~np.isnan(first_spike[phase][trial])]))                  #latency
            moments_overview_temp[3, trial]= (np.nanmean(nspikes[phase][trial]))                                                         #nrofspikes
            moments_overview_temp[4, trial]= np.round(np.nanvar(np.where(nspikes[phase][trial]!=0,nspikes[phase][trial],np.nan),0), 2)   #varnrofspikes

        moments_overview.append(moments_overview_temp)

    return moments_overview

def populate_df(dataframe:pd.core.frame.DataFrame, position:int, values:list) -> pd.core.frame.DataFrame:

    for row in range(values.shape[0]):
        dataframe.loc[(position, row), :]= values[row]

    return dataframe

def populate_multistimulus_df(dataframe:pd.core.frame.DataFrame, position:int, stim_position:int, values:list,  ) -> pd.core.frame.DataFrame:
    for row in range(values.shape[0]):
        dataframe.loc[(position, stim_position, row), :]= values[row]

    return dataframe


def calculate_ONOFF_polarity(ONOFF_dfr:list)->np.ndarray:

    """
    ON first in list, working currently with response to green
    """
    return ((ONOFF_dfr[0].loc(axis=0)[:,2]['Mean_nr']-ONOFF_dfr[1].loc(axis=0)[:,2]['Mean_nr'])/(ONOFF_dfr[0].loc(axis=0)[:,2]['Mean_nr']+ONOFF_dfr[1].loc(axis=0)[:,2]['Mean_nr'])).values

def get_505_spikes(cell_idx, df_spikes, stimulus_traits, into='frames', sampling_freq=None, inone=True):
    """
    TODO:add code to check for pseudorder
    """
    cell_df_idx= Basic.get_cell_index_stimulus_matching(cell_idx, df_spikes, stimulus_traits)
    cell_spikes= Basic.get_cell_stimulus_spikes(cell_df_idx, df_spikes)
    spikes_per_trial= (sas.get_spikes_per_trigger_type_new(cell_spikes, stimulus_traits['stim_rel_trig'], 2, stimulus_traits['stim_trials'])[0])
    if into=='time':
        if inone==True:
        ###TODO: right assert for sampling freq input
            return [item/sampling_freq for sublist in spikes_per_trial for item in sublist]
        else:
            return [[item/sampling_freq for item in sublist] for sublist in spikes_per_trial]
    else:
        return [item for sublist in spikes_per_trial for item in sublist]

def calculate_transcience(object, ON_edges, OFF_edges):
    assert isinstance(object, Steps_505)
    transience_array=np.zeros(len(object.cell_idces))
    centroid=[]
    #lost_cells=[]
    for cell in range(len(object.cell_idces)):
        #print('look at',cell)
        incr=0
        #create 20ms binned histogram
        take_hist=np.histogram(get_505_spikes(object.cell_idces[cell], object.df_spikes, object.stimulus_traits), bins=200, range=[0, 4*object.sampling_freq])[0]
        if np.argmax(take_hist)==0:
            transience_array[cell]=-99
            continue
        centroid.append(np.argmax(take_hist))

        if Basic.helper_filter(object.df_ON.loc(axis=0)[cell,2]['Latency'], ON_edges[0][1]*0.02, 'bigger'):
            if bool(Basic.helper_filter(object.df_ON.loc(axis=0)[cell,2]['Reliability'], .79, 'bigger')*Basic.helper_filter(object.df_ON.loc(axis=0)[cell,2]['Precision'], .18, 'smaller')):
                incr= int(np.floor((object.df_ON.loc(axis=0)[cell,2]['Latency']-ON_edges[0][0]*0.02)/0.02))
            else:
                #print(cell,'ON_is out')
                if object.df_ON.loc(axis=0)[cell,2]['Mean_nr']>object.df_OFF.loc(axis=0)[cell,2]['Mean_nr']:
                    #print(cell, 'cell is kicked out')
                    transience_array[cell]=-99
                    continue
                else:
                    ON_profile=99


        ON_binT= np.array(ON_edges[0])+incr
        ON_binS= np.array(ON_edges[1])+incr  if (np.array(ON_edges[1])+incr)[1]<100 else np.array([ON_edges[1][0]+incr ,ON_edges[1][1]])


        if Basic.helper_filter(object.df_OFF.loc(axis=0)[cell,2]['Latency'], (OFF_edges[0][1]*0.02)-2, 'bigger'):
            if bool(Basic.helper_filter(object.df_OFF.loc(axis=0)[cell,2]['Reliability'], .79, 'bigger')*Basic.helper_filter(object.df_OFF.loc(axis=0)[cell,2]['Precision'], .18, 'smaller')):
                incr= int(np.floor((object.df_OFF.loc(axis=0)[cell,2]['Latency']+2-OFF_edges[0][0]*0.02)/0.02))
                #print(incr)
            else:
                #print(cell,'OFF_is out')
                if object.df_ON.loc(axis=0)[cell,2]['Mean_nr']<object.df_OFF.loc(axis=0)[cell,2]['Mean_nr']:
                    #print(cell, 'cell is kicked out')
                    transience_array[cell]=-99
                    if 'ON_profile' in locals():
                        del ON_profile
                    continue
                else:
                    OFF_profile=-99

        OFF_binT= np.array(OFF_edges[0])+incr
        OFF_binS= np.array(OFF_edges[1])+incr

        if 'ON_profile' in locals():
            #print('ON should be out')
            if 'OFF_profile' in locals():
                transience_array[cell]=-99
                del ON_profile, OFF_profile
                continue
            else:
                ONOFF_profile=-1
        else:
            pass

        if 'OFF_profile' in locals():
            #print('OFF should be out')
            if 'ON_profile' in locals():
                transience_array[cell]=-99
                del ON_profile, OFF_profile
                continue
            else:
                ONOFF_profile=1

        else:
            pass


        #    lost_cells.append(cell)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                ON_profile= (sum(take_hist[ON_binT[0]:ON_binT[1]])/4-sum(take_hist[ON_binS[0]:ON_binS[1]])/np.diff(ON_binS))/(sum(take_hist[ON_binT[0]:ON_binT[1]])/4+sum(take_hist[ON_binS[0]:ON_binS[1]])/np.diff(ON_binS))
            except:
                ON_profile=0
            try:
                OFF_profile=(sum(take_hist[OFF_binT[0]:OFF_binT[1]])/4-sum(take_hist[OFF_binS[0]:OFF_binS[1]])/np.diff(ON_binS))/(sum(take_hist[OFF_binT[0]:OFF_binT[1]])/4+sum(take_hist[OFF_binS[0]:OFF_binS[1]])/np.diff(ON_binS))
            except:
                OFF_profile=0

            if 'ONOFF_profile' not in locals():
                #print('Will create it')
                try:
                    ONOFF_profile=  ((sum(take_hist[ON_binT[0]:ON_binT[1]])+sum(take_hist[ON_binS[0]:ON_binS[1]]))-(sum(take_hist[OFF_binT[0]:OFF_binT[1]])+sum(take_hist[OFF_binS[0]:OFF_binS[1]])))/((sum(take_hist[ON_binT[0]:ON_binT[1]])+sum(take_hist[ON_binS[0]:ON_binS[1]]))+(sum(take_hist[OFF_binT[0]:OFF_binT[1]])+sum(take_hist[OFF_binS[0]:OFF_binS[1]])))
                except:
                    #print ('This somehow is empty')
                    del ON_profile, OFF_profile,
                    transience_array[cell]=-99
                    continue

            else:
                pass
                #print('let')
                #print('ONOFF Defaultttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt:',ONOFF_profile)
        if ONOFF_profile>0:
            transience_array[cell]=ON_profile

        else:
            transience_array[cell]=OFF_profile
        #print(cell, ON_profile, OFF_profile)
        #if transience_array[cell]==0:
            #print(cell, helper_filter(df_ON.loc(axis=0)[cell,2]['Latency'], ON_edges[0][0]*0.02, 'smaller'), helper_filter(df_OFF.loc(axis=0)[cell,2]['Latency'], (OFF_edges[0][0]*0.02)-2, 'smaller'))
            #print(ON_profile, OFF_profile, ONOFF_profile, transience_array[cell])
            #print((sum(take_hist[ON_binT[0]:ON_binT[1]]), sum(take_hist[ON_binS[0]:ON_binS[1]])))
            #print((sum(take_hist[OFF_binT[0]:OFF_binT[1]]),sum(take_hist[OFF_binS[0]:OFF_binS[1]])))
        #print(ON_profile, OFF_profile, transience_array[cell])
        del ON_profile, OFF_profile, ONOFF_profile


    return transience_array, centroid,

def plot_polVStrans(plot_vars:list, lims:list, ):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x=plot_vars[0], y=plot_vars[1], c=plot_vars[2])
    plt.ylim(lims[0],lims[1])
    plt.xlim(lims[0],lims[1])
    plt.xlabel('Polarity')
    plt.ylabel('Transience')



class Steps_505():

    """
    Will need to check for stimulus order, TODO
    Works for single-exp datasets
    """


    def __init__(self, recording, json_dict:str, FFF_index, sampling_freq):

        cell_labels_dir=json_dict
        self.data_categ= Basic.open_RGC_dictionary(cell_labels_dir)
        self.cell_idces, (self.ON_bounds, self.OFF_bounds, self.ONOFF_bounds)= Basic.flatten_RGC_types(self.data_categ)

        self.df_spikes, self.df_stimulus= Basic.prepare_dataframes(recording, [FFF_index], self.cell_idces)
        self.sampling_freq= sampling_freq
        self.stimulus_traits= Basic.get_stimulus_traits(self.df_stimulus, 0)

        self.df_ON, self.df_OFF= self.compute_stats()
        self.plot_summary(self.df_ON, self.ON_bounds, 'ON')
        self.plot_summary(self.df_OFF, self.OFF_bounds, 'OFF')

        self.polarity= calculate_ONOFF_polarity([self.df_ON, self.df_OFF])
        self.transience, _= Basic.proceed(calculate_transcience, (self, [[11,15],[17,100]], [[111,115],[117,200]]))

        plot_polVStrans([self.polarity, self.transience,
        Basic.create_color_book([self.ON_bounds, self.OFF_bounds, self.ONOFF_bounds],['red', 'gray', '#eedbdb'])], [-1.04, 1.04])

    def create_stepsstats_dataframe(self):
        thindex= create_multiIndex([1, self.stimulus_traits['stim_trials']], ['Unit', 'Color'])
        column_names=['Reliability', 'Precision', 'Latency', 'Mean_nr', 'Var_nr']
        return init_mI_Dataframe(len(column_names), column_names, thindex)

    def compute_stats(self):

        self.df_ON= self.create_stepsstats_dataframe()
        self.df_OFF= self.create_stepsstats_dataframe()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cell in range(len(self.cell_idces)):

                #spiketrains_list= get_cell_spiketrains_per_stimulus(arrays_selectmg[cell], df_spikes, stimulus_traits, 4, Sampling_freq,)
                if '4' in self.stimulus_traits['stim_name']:
                    first_spike_test_cell=get_first_spike_per_phase(self.cell_idces[cell], self.df_spikes, self.stimulus_traits, 2*4, self.sampling_freq, 2*2.0)
                    nspikes_test_cell=get_number_of_spikes_per_phase(self.cell_idces[cell], self.df_spikes, self.stimulus_traits, 2*4, self.sampling_freq, 2*2.0)
                else:
                    first_spike_test_cell=get_first_spike_per_phase(self.cell_idces[cell], self.df_spikes, self.stimulus_traits, 4, self.sampling_freq, 2.0)
                    nspikes_test_cell=get_number_of_spikes_per_phase(self.cell_idces[cell], self.df_spikes, self.stimulus_traits, 4, self.sampling_freq, 2.0)



                if 'pseudo' in self.stimulus_traits['stim_name']:
                    """
                    TODO:how does pseudorder comes into existence?
                    """
                    for idx, val in enumerate(nspikes_test_cell):
                        print(idx)
                        nspikes_holder= Basic.reorder_spiketrains(nspikes_test_cell[idx], pseudorder)
                        first_spike_holder= Basic.reorder_spiketrains(first_spike_test_cell[idx], pseudorder)

                        nspikes_test_cell[idx]=np.array(nspikes_holder)
                        first_spike_test_cell[idx]=np.array(first_spike_holder)

                moments_ON_OFF= calculate_moments(len(self.df_ON.columns), self.stimulus_traits, first_spike_test_cell, nspikes_test_cell)
                self.df_ON = populate_df(self.df_ON, cell, moments_ON_OFF[0].T)
                self.df_OFF = populate_df(self.df_OFF, cell, moments_ON_OFF[1].T)

        return self.df_ON, self.df_OFF



    def plot_crude(self, moments_dat, bounds, by='Reliability', title='', ax=None):
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()
        cmap= 'inferno_r' if by=='Reliability' else 'inferno'
        label= 'Imprecision' if by=='Precision' else by
        ax_plot=ax.scatter(x=moments_dat.loc(axis=0)[bounds,2]['Latency'].values[:-1],
                    y=moments_dat.loc(axis=0)[bounds,2]['Mean_nr'].values[:-1],
                    c=moments_dat.loc(axis=0)[bounds,2][by].values[:-1],
                    s=25, cmap=plt.cm.get_cmap(cmap, 5))
        plt.colorbar(ax_plot, ticks=range(6), label=label, ax=ax)
        ax.set_xlim(0,2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(ylabel='Avg Spikes', xlabel='Latency')
        ax.set_title(title)

        #plt.show()
    def plot_lat_stats(self, moments_dat, bounds, title='',bin_edges=[0.22, 0.3], ax=None):
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()
        x0=moments_dat.loc(axis=0)[bounds,2]['Latency'].values[:-1]
        print('Skew-index:', Basic.calculate_PSI(x0))
        print('Percentage in window:', Basic.calculate_inwindow(x0, bin_edges))

        ax.hist(x0, bins=100, range=[0,2])
        ax.axvline(np.nanmean(x0), color='red')
        ax.axvline(np.nanmedian(x0), color='green')
        ax.axvspan( bin_edges[0],  bin_edges[1], alpha=0.2)
        ax.set(xlabel='Latency')
        ax.set_title(title)

    def plot_latVSjit(self, moments_dat, bounds, title='', ax=None):
        if ax is not None:
            pass
        else:
            fig, ax= plt.subplots()
        cm= 'blue' if title=='ON' else 'red'
        ax_plot=ax.scatter(x=moments_dat.loc(axis=0)[bounds,2]['Latency'].values[:-1],
                    y=moments_dat.loc(axis=0)[bounds,2]['Precision'].values[:-1],
                    s=25, c=cm, alpha=0.4)
        plt.xlim(0)
        plt.ylim(0)
        #plt.axline((0, 0), slope=1, ax=ax)
        ax.set(ylabel='Imprecision', xlabel='Latency')
        ax.set_title(title)

    def plot_summary(self, moments_dat, bounds, title='', ):
        fig, ax = plt.subplots(1, 4, figsize=(12, 5))


        self.plot_lat_stats(moments_dat, bounds, title=title, ax=ax[0], )
        self.plot_crude(moments_dat, bounds, title=title, ax=ax[1])
        self.plot_crude(moments_dat, bounds, by='Precision', title=title, ax=ax[2])
        self.plot_latVSjit(moments_dat, bounds, title=title, ax=ax[3])
        fig.tight_layout()
        plt.show()
