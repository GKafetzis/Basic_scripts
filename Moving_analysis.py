from MEA_analysis import stimulus_and_spikes as sas
from MEA_analysis import spike_plotly
from importlib import reload

import numpy as np
import pycircstat
from itertools import compress
from math import radians, degrees
import random
import pyspike

import ipywidgets as widgets
from ipywidgets import HBox, VBox
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors
from IPython.display import display

class Moving_stimulus():



    default_order= [0, 180, 45, 225, 90, 270, 135, 315]


    def __init__(self, dataframe_spikes, dataframe_stimulus, directions_degrees, stationary=False):
        self.df_spikes = dataframe_spikes
        self.df_spikes= self.df_spikes.reset_index()
        self.df_stimulus = dataframe_stimulus
        self.repeat_logic = int(self.df_stimulus['Stimulus_repeat_logic'].values[0])
        self.nr_trials = int((len(self.df_stimulus['Trigger_Fr_relative'].values[0])-1)/self.repeat_logic)
        self.dir_degrees= directions_degrees
        self.stationary= stationary

        self.check_order()
        self.ds_cutoff= float(input("DSI cutoff in per thousand integer"))/1000

        self.remove_silent()
        self.spikes_per_seg_p()
        self.norm_byarea()
        self.calculate_dos()
        self.get_vector_stats_p()


        self.figup= self.plot_polar_histogram(self.vector_lengths, 0.14, "Vector length > 0.14")
        self.figdown= self.plot_dos_histograms()

        fig_subplots=  VBox([go.FigureWidget(self.figup), go.FigureWidget(self.figdown)])
        display(fig_subplots)

        self.polarplots=self.plot_polar_pref_selection(np.arange(self.normed_spikes_per_segment.shape[0])[np.logical_or(self.masking_array, self.ds_vals>self.ds_cutoff)])
        display(self.polarplots)

        self.example_sigDS= self.calculate_dsi_significance()
        self.ds_onnull= self.plot_ds_onnull()
        DS_cell_plots=  HBox([go.FigureWidget(self.example_sigDS), go.FigureWidget(self.ds_onnull)])
        display(DS_cell_plots)

        self.plot_hidden_gems()




    def check_order(self,):
        if self.dir_degrees==self.default_order:
            print('I already have a template for reordering')
            self.re_order= self.estimate_reorder_indices(self.default_order)

            self.order_angles(self.default_order)
        else:
            import ast
            print('Please provide the order of presentation')
            self.provided_order = ast.literal_eval(input("Enter order of directions"))
            self.re_order= self.estimate_reorder_indices(self.provided_order)

            self.order_angles(self.provided_order)

    def estimate_reorder_indices(self, inpresentation_order):
        self.re_order=[]
        for i in inpresentation_order:
            self.re_order.append(np.where(np.sort(inpresentation_order)==i)[0][0])

        return self.re_order

    def reorder(self, arr, index, n):
        temp = [0] * n;
        for i in range(0,n):
            temp[index[i]] = arr[i]
        return temp


    def order_angles(self, pres_order,):

        self.angles_ordered_d= self.reorder(pres_order, self.re_order, len(pres_order))
        self.angles_ordered_d= np.append(self.angles_ordered_d, self.angles_ordered_d[0])

        self.angles_ordered_a= np.zeros(len(self.angles_ordered_d))
        for i in range(len(self.angles_ordered_d)):
            self.angles_ordered_a[i]= radians(self.angles_ordered_d[i])


    def spikes_per_seg_p(self,):


        if not self.stationary:
            self.spikes_per_direction= np.zeros([self.df_spikes.shape[0],self.repeat_logic])
            self.spikes_per_segment= np.zeros([self.nr_trials, self.df_spikes.shape[0], self.repeat_logic,])

            for idx in range(self.df_spikes.shape[0]):
                for trigger in range(self.repeat_logic):

                    spikes_per_dir=(sas.get_spikes_per_trigger_type_new(
                    self.df_spikes.loc[idx]['Spikes'].compressed(), ((self.df_stimulus['Trigger_Fr_relative'].values)[0]),
                    trigger, self.repeat_logic)[0])

                    self.spikes_per_direction[idx, trigger]=sum([len(listElem) for listElem in spikes_per_dir])/self.nr_trials

                    for trial in range(self.nr_trials):
                                self.spikes_per_segment[trial, idx, trigger]= len(spikes_per_dir[trial])


                self.spikes_per_direction[idx]= self.reorder(self.spikes_per_direction[idx], self.re_order, len(self.spikes_per_direction[idx]))

                for trial in range(self.nr_trials):
                    self.spikes_per_segment[trial, idx, :]=  self.reorder(self.spikes_per_segment[trial, idx, :], self.re_order, len(self.spikes_per_segment[trial, idx, :]))

        self.spikes_per_segment= np.append(self.spikes_per_segment, self.spikes_per_segment[:,:,None,0], axis=2)

        self.spikes_per_direction= np.insert(self.spikes_per_direction, self.spikes_per_direction.shape[1], self.spikes_per_direction[:,0], axis=1)

    def cell_hist_per_dir(self, cell, nr_bins, trial_dur, sampling_freq=17852.76785):
        self.trial_dur= trial_dur
        hist_per_dir= np.zeros((self.repeat_logic, nr_bins))
        for trigger in range(self.repeat_logic):
            spikes_per_dir=(sas.get_spikes_per_trigger_type_new(
                                self.df_spikes.loc[cell]['Spikes'].compressed(), ((self.df_stimulus['Trigger_Fr_relative'].values)[0]),
                                trigger, self.repeat_logic)[0])

            hist_per_dir[trigger, :]= np.histogram(np.sort([val/sampling_freq for sublist in spikes_per_dir for val in sublist]), bins=nr_bins, range=[0, trial_dur])[0]

        return hist_per_dir

    def find_threshold_events(self, spike_profile, t_thresh, sig_thresh):
        events=[]
        try:
            time_profile=np.histogram(np.arange(2), bins=spike_profile.shape[1], range=[0, self.trial_dur])[1][1:]
        except:
            time_profile=np.histogram(np.arange(2), bins=len(spike_profile), range=[0, self.trial_dur])[1][1:]
        if np.ndim(spike_profile)==1:


            points=get_persistent_homology(spike_profile)
            for point in points:
                if point.died==None:
                    point.died=0
                if point.born==None:
                    point.born=0
                if time_profile[point.born]>t_thresh:
                    if spike_profile[point.born]-spike_profile[point.died]>sig_thresh:
                        events.append(point)

        elif np.ndim(spike_profile)==2:

            for row in range(spike_profile.shape[0]):
                idces=[]
                points=get_persistent_homology(spike_profile[row])
                for point in points:
                    if point.died==None:
                        point.died=0
                    if point.born==None:
                        point.born=0
                    if (time_profile[point.born]>t_thresh) & (spike_profile[row][point.born]-spike_profile[row][point.died]>sig_thresh):
                        idces.append(True)
                    else:
                        idces.append(False)
                events.append(list(compress(points, idces)))
        else:
            print("Sorry, running this function for more than 2 dimensions is not currently possible")


        return events

    def remove_silent(self,):

        """
        Returns df_spikes array having removed the indices of the empty/silent neurons.
        """
        print('Assuming defaults for a 1000speed stimulus\nFirst removing the silent (no detected events) neurons\n****')
        silent_neurons=[]
        for cell_idx in np.arange(len(self.df_spikes)):
            spikes_profile=self.cell_hist_per_dir(cell_idx, nr_bins=12, trial_dur=4,)
            if sum(np.unique([len(self.find_threshold_events(spikes_profile, 1, 4)[i]) for i in range(self.repeat_logic)]))<1:
                silent_neurons.append(cell_idx)
        #print (cell_idx, ([len(test_class.find_threshold_events(spikes_profile, 1, 4)[i]) for i in range(8)]) )
        excluded = [a for a in np.arange(len(self.df_spikes)) if a not in set(silent_neurons)]
        self.df_spikes= self.df_spikes.loc[excluded].reset_index(drop=True)
        print('%d silent neurons have been detected and removed' %len(silent_neurons))





    def norm_byarea(self):

        """
        Normalization by area, default
        """
        self.normed_spikes_per_segment= np.zeros((self.spikes_per_segment.shape[1], len(self.angles_ordered_a)))
        for cell in range(self.spikes_per_segment.shape[1]):
            self.normed_spikes_per_segment[cell]= np.nanmean(self.spikes_per_segment[:, cell], axis=0)/(np.nansum(self.spikes_per_segment[:,  cell])/(self.nr_trials*self.repeat_logic))

    def norm_byarea_p(self, binned_spikes, n_trials, n_directions):
        normed_spikes_per_segment= np.zeros((binned_spikes.shape[1], binned_spikes.shape[2]))
        for cell in range(binned_spikes.shape[1]):
            normed_spikes_per_segment[cell]= np.mean(binned_spikes[:, cell], axis=0)/(np.sum(binned_spikes[:,  cell])/(n_trials*n_directions))

        return normed_spikes_per_segment


    def get_vector_length(self, cell_spikes, d=0.78539816):

        return pycircstat.resultant_vector_length(self.angles_ordered_a[:-1], cell_spikes[:-1], d=d)

    def get_vector_angle_degrees(self, cell_spikes, d=0.78539816):

        return degrees(pycircstat.mean(self.angles_ordered_a[:-1], cell_spikes[:-1], d=d))

    def get_vector_stats_p(self,):
        self.vector_lengths= np.zeros(self.normed_spikes_per_segment.shape[0])
        self.vector_angles= np.zeros(self.normed_spikes_per_segment.shape[0])
        for cell in range(self.normed_spikes_per_segment.shape[0]):
            self.vector_lengths[cell]= self.get_vector_length(self.normed_spikes_per_segment[cell])
            self.vector_angles[cell]= self.get_vector_angle_degrees(self.normed_spikes_per_segment[cell])





    def find_vector_angle_neighbors(self, vector_angle, n_neighbors=3):
        """
        Input in full circular configuration (without subtracting the 'extra' direction)
        Neighbors can be symmetric (e.g. 0 or 3),but option provided for assymetric
        (preferred and second strongest). Input of vector_angle in degrees
        """
        centre=np.argmin(np.abs(self.angles_ordered_d[:-1]-vector_angle))

        if n_neighbors==0:
            return centre
        elif n_neighbors==3:
            return [centre-1, centre, centre+1]

        elif n_neighbors==2:
            return [centre, [centre-1, centre+1] [np.argmin(np.abs(np.array(centre-1, centre+1)-vector_angle))]]


    def calculate_dos(self,):
        """
        Input in full circular configuration (without subtracting the 'extra' direction)
        Currently works with simple method (max number of spikes)
        Returns ds_val, os_val
        """
        nspikes_arrtoy=np.zeros([self.normed_spikes_per_segment.shape[0],self.normed_spikes_per_segment.shape[1]-1])
        self.os_vals= np.zeros(self.normed_spikes_per_segment.shape[0])
        self.ds_vals= np.zeros(self.normed_spikes_per_segment.shape[0])

        for cell in range(self.normed_spikes_per_segment.shape[0]):
            nspikes_arrtoy[cell]= self.normed_spikes_per_segment[cell][:-1]

        nspikes_arrtoy= np.tile(nspikes_arrtoy, 2)


        for idx, cell in enumerate(range(self.normed_spikes_per_segment.shape[0])):


            dominant_dir= np.argmax(self.normed_spikes_per_segment[cell])
            dominant_ds=nspikes_arrtoy[cell][dominant_dir]
            null_ds=nspikes_arrtoy[cell][dominant_dir+4]

            dominant_os= nspikes_arrtoy[cell][dominant_dir] + nspikes_arrtoy[cell][dominant_dir + 4]
            null_os= nspikes_arrtoy[cell][dominant_dir + 2] + nspikes_arrtoy[cell][dominant_dir + 6]

            ds= (dominant_ds-null_ds)/(dominant_ds+null_ds)
            os= (dominant_os-null_os)/(dominant_os+null_os)

            self.ds_vals[idx]=ds
            self.os_vals[idx]=os

    def calculate_dos_p(self, binned_spikes, has_9th=True):

        nr_cells= binned_spikes.shape[0]
        nr_trials= binned_spikes.shape[1]

        os_vals= np.zeros(nr_cells)
        ds_vals= np.zeros(nr_cells)

        if has_9th:
            nspikes_arrtoy=np.zeros([nr_cells,nr_trials-1])
            for cell in range(nr_cells):
                nspikes_arrtoy[cell]= binned_spikes[cell][:-1]

            nspikes_arrtoy= np.tile(nspikes_arrtoy, 2)
        else:
            nspikes_arrtoy=np.zeros([nr_cells,nr_trials-1])
            nspikes_arrtoy= np.tile(binned_spikes, 2)




        for idx, cell in enumerate(range(nr_cells)):


            dominant_dir= np.argmax(binned_spikes[cell])
            dominant_ds=nspikes_arrtoy[cell][dominant_dir]
            null_ds=nspikes_arrtoy[cell][dominant_dir+4]

            dominant_os= nspikes_arrtoy[cell][dominant_dir] + nspikes_arrtoy[cell][dominant_dir + 4]
            null_os= nspikes_arrtoy[cell][dominant_dir + 2] + nspikes_arrtoy[cell][dominant_dir + 6]

            ds= (dominant_ds-null_ds)/(dominant_ds+null_ds)
            os= (dominant_os-null_os)/(dominant_os+null_os)

            ds_vals[idx]=ds
            os_vals[idx]=os

        return ds_vals, os_vals





    def calculate_nn_ds(self, nn=3):
        """
        Input in full circular configuration (without subtracting the 'extra' direction)
        Currently works with nearest neighbors. Input: number of Neighbors
        Returns ds_vals
        """
        nspikes_arrtoy=np.zeros([self.normed_spikes_per_segment.shape[0],self.normed_spikes_per_segment.shape[1]-1])
        ds_vals= np.zeros(self.normed_spikes_per_segment.shape[0])


        for cell in range(self.normed_spikes_per_segment.shape[0]):
            nspikes_arrtoy[cell]= self.normed_spikes_per_segment[cell][:-1]

        nspikes_arrtoy= np.tile(nspikes_arrtoy, 2)


        for idx, cell in enumerate(range(self.normed_spikes_per_segment.shape[0])):
            strong_dirs=find_vector_angle_neighbors(self.vector_angles[cell], nn)

            dominant= np.sum(nspikes_arrtoy[cell][strong_dirs])
            null=  np.sum(nspikes_arrtoy[cell][np.array(strong_dirs)+4])


            ds= (dominant-null)/(dominant+null)

            ds_vals[idx]=ds

        return ds_vals

    #def run_once(self, f):
    #    def wrapper(*args, **kwargs):
    #        if not wrapper.has_run:
    #            wrapper.has_run = True
    #            return f(*args, **kwargs)
    #    wrapper.has_run = False
    #    return wrapper


    #@run_once
    #def plot_nulldsi_example(self, ):



    def calculate_dsi_significance(self, npermutations=1000, ntrials=10, ndirections=8, a=0.05, plot_example=True):

        """
        Real-neuron array of shape[ntrials, neurons, ndirections]
        """

        print('Calculating significance for each cell with %d permutations' %npermutations)
        permuted_pop= np.zeros([ntrials, npermutations, ndirections,])
        self.dsi_sig=np.full(self.normed_spikes_per_segment.shape[0], False)

        for real_neuron in range(self.normed_spikes_per_segment.shape[0]):
            assigned_bins=np.zeros(int(np.round(np.sum(self.spikes_per_direction[real_neuron][:-1]),0)), dtype=int)
            for permutation in range(npermutations):
                for trial in range(ntrials):
                    for spike in range(len(assigned_bins)):
                        assigned_bins[spike]= random.randint(0,ndirections-1)
                    permuted_pop[trial, permutation]= np.bincount(assigned_bins, minlength=ndirections)


            if sum(self.calculate_dos_p(self.norm_byarea_p(permuted_pop, ntrials, ndirections), has_9th=False)[0]>self.ds_vals[real_neuron])/npermutations<a:
                self.dsi_sig[real_neuron]=True
                if plot_example==True:
                    fig = px.histogram(self.calculate_dos_p(self.norm_byarea_p(permuted_pop, ntrials, ndirections), has_9th=False)[0],
                    nbins=20, range_x=[0, 1], histnorm='probability')
                    fig.add_vline(self.ds_vals[real_neuron], line_color="red")
                    fig.update_layout(height=500, width=400, showlegend=False,
                    title_text="Example of Significant DS cell <br> with its permuted DS distribution", title_x=0.5)

                    #display(fig)
                    plot_example=False
                else:
                    continue
        return fig

    def plot_ds_onnull(self,):

        fig= px.histogram(x=self.ds_vals, nbins=20, range_x=[0, 1], histnorm='probability',)
        fig.add_trace(go.Scatter(x=self.ds_vals[np.arange(len(self.ds_vals))[self.dsi_sig]], y=[sum(self.dsi_sig)/len(self.ds_vals)]*sum(self.dsi_sig), mode='markers', name='significant DS cells'))
        fig.add_vline(x=self.ds_cutoff , line_width=3, line_dash="dash", line_color="green",)
        fig.update_layout(title_text='DSI distribution', title_x=0.5)
        fig.update_layout(height=500, width=400, legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.45
        ))


        return fig

    def plot_hidden_gems(self,):

        self.hidden_gems= np.setdiff1d(np.arange(len(self.dsi_sig))[self.dsi_sig], np.arange(len(self.dsi_sig))[np.logical_or(self.masking_array, self.ds_vals>self.ds_cutoff)])
        if len(self.hidden_gems)==0:
            pass
        else:
            print("There are some significant DSI cells that you might have missed")
            display(self.plot_polar_pref_selection(self.hidden_gems))









    def plot_polar_histogram(self, criterion, threshold, title_name):

        masking_array= np.full(len(self.vector_angles), False)
        for cell in range(len(self.vector_angles)):
            if criterion[cell]>threshold:
                masking_array[cell]=True

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}]*2]*1,
                           subplot_titles=("cells below", "cells above"))

        fig.add_trace(go.Barpolar(
                            r = np.histogram(self.vector_angles[~masking_array], bins=10, range=[0,360], weights=criterion[~masking_array] )[0],
                            theta = np.histogram(self.vector_angles[~masking_array], bins=10, range=[0,360], weights=criterion[~masking_array] )[1][:-1],
                            width=360/10,

                            marker=dict(color='lightgray')

                            ), 1, 1)


        fig.add_trace(go.Barpolar(
                            r = np.histogram(self.vector_angles[masking_array], bins=10, range=[0,360], weights=criterion[masking_array] )[0],
                            theta = np.histogram(self.vector_angles[masking_array], bins=10, range=[0,360], weights=criterion[masking_array] )[1][:-1],
                            width=360/10,

                            marker=dict(color='lightgreen')

                            ), 1, 2)

        fig.update_polars(
                radialaxis = dict( showticklabels=False, ticks='', tickmode='array', tickvals=[ 0.5, 1, ], range=[0,1]),
                angularaxis = dict(showticklabels=False, ticks=''),
            )


        fig.update_layout(height=500, width=800,
                  title_text=title_name, title_x=0.5)

        self.masking_array= masking_array
        return fig


    def plot_dos_histograms(self,):



        figure1 = px.histogram(x=self.ds_vals, nbins=20, range_x=[0, 1], histnorm='probability')
        figure2 = px.histogram(x=self.os_vals, nbins=20, range_x=[0, 1], histnorm='probability')

        figure1_traces = []
        figure2_traces = []
        for trace in range(len(figure1["data"])):
            figure1_traces.append(figure1["data"][trace])
        for trace in range(len(figure2["data"])):
            figure2_traces.append(figure2["data"][trace])


        fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Direction", "Orientation"))

        fig.update_layout(height=500, width=800,
                  title_text="Selectivity index", title_x=0.5)

        for traces in figure1_traces:
            fig.append_trace(traces, row=1, col=1)
        for traces in figure2_traces:
            fig.append_trace(traces, row=1, col=2)
        fig.add_vline(x=self.ds_cutoff , line_width=3, line_dash="dash", line_color="green", row=1, col=1)
        fig.update_xaxes(range=[0,1])



        return fig

    def plot_polar_pref_selection(self, select,norm=True):

        nr_cols= 4
        nr_rows= int(np.ceil(len(select)/nr_cols))
        ###Do sth with the select length
        cols = colors.DEFAULT_PLOTLY_COLORS
        fig = make_subplots(rows=nr_rows, cols=nr_cols, specs=[[{'type': 'polar'}]*nr_cols]*nr_rows,
                           subplot_titles=list(map(lambda x: f'{self.df_spikes.loc[x][0]}' , select)))

        for y in range(0,nr_rows):
            for x in range(0,nr_cols):

                if y*nr_cols+x<len(select):

                    if norm==True:

                        for trial in range(self.spikes_per_segment.shape[0]):
                            fig.add_trace(go.Scatterpolar(
                            r = self.spikes_per_segment[trial, select[y*nr_cols+x]]/(np.sum(self.spikes_per_segment[:, select[y*nr_cols+x]])/80),
                            theta = self.angles_ordered_d,
                            mode='lines',
                            marker=dict(color='#d6d6d6')
                            ), y+1, x+1)

                        fig.add_trace(go.Scatterpolar(
                        r = self.normed_spikes_per_segment[select[y*nr_cols+x]],
                        theta = self.angles_ordered_d,
                        mode='lines',
                        marker=dict(color=cols[0])
                        ), y+1, x+1)


                    fig.add_trace(go.Scatterpolar(
                    r = [0, pycircstat.resultant_vector_length(self.angles_ordered_a[:-1],  self.normed_spikes_per_segment[select[y*nr_cols+x]][:-1], d=0.78539816)],
                    theta = [0, degrees(pycircstat.mean(self.angles_ordered_a[:-1],  self.normed_spikes_per_segment[select[y*nr_cols+x]][:-1], d=0.78539816))],
                    mode='lines',
                    marker=dict(color=cols[3])
                    ), y+1, x+1)

                    fig.update_annotations(yshift=25, xshift=-45)
                    #fig.layout.annotations[y*nr_cols+x+1].update(text="%d" % select[y*nr_cols+x])
                else:
                    continue

        fig.update_layout(
            autosize=False,
            width=1200,
            height=int(280*nr_rows))

        fig.update_layout(showlegend=False)
        fig.update_polars(radialaxis=dict(tickmode='array', tickvals=[ 0.5, 1, ], ticktext=['', ''], range=[0,2]))


        return fig


    def present_rawspikes(self, cell_indices:list, trial_duration=4, sampling_freq=17852):

        dat_cell_indices=[]
        for val in cell_indices:
            dat_cell_indices.append(self.df_spikes[self.df_spikes['Cell index']==val].index[0])
        for cell_idx, cell_val, in enumerate(dat_cell_indices):
            figs_list=[]
            spiketrains_list=[]
            spike_for_directions=[]
            for trigger in range(self.repeat_logic):
                spiketrain_list=[]
                spikes_per_dir=(sas.get_spikes_per_trigger_type_new(
                self.df_spikes.loc[cell_val]['Spikes'].compressed(), ((self.df_stimulus['Trigger_Fr_relative'].values)[0]),
                trigger, self.repeat_logic)[0])

                spikes_per_dir=[spikes_per_dir[i]/sampling_freq for i in range(self.nr_trials)]
                spike_for_directions.append(spikes_per_dir)

                for repeat in range(self.nr_trials):
                    spiketrain_list.append(pyspike.SpikeTrain(spikes_per_dir[repeat], edges=[0, trial_duration]))
                figs_list.append(spike_plotly.plot_raster(spiketrain_list, method='isi'))
                spiketrains_list.append(spiketrain_list)

            spiketrains_list= self.reorder(spiketrains_list, self.re_order, self.repeat_logic)

            plotted_ds= self.plot_rawspikes(spiketrains_list, self.df_spikes.loc[cell_val][0], trial_duration, self.nr_trials)
            self.update_plot_withdirections(plotted_ds)

    def plot_rawspikes(self, spiketrains_list, cell, trial_dur, nr_trials):
        reload(spike_plotly)
        return(spike_plotly.plot_ds_cell(spiketrains_list, cell, trial_dur, self.nr_trials))

    def update_plot_withdirections(self, plot_handle):
        plot_handle['layout'].update(
        annotations=[
        dict(
            x= -0.01, y=1.5, # annotation point
            xref='x1',
            yref='y1',
            showarrow=True,
            arrowhead=3,
            ax= -25,
            ay= 0.4,
        ),

        dict(
            x= -0.012, y=3, # annotation point
            xref='x1',
            yref='y3',
            showarrow=True,
            arrowhead=3,
            ax= -20,
            ay= 12.54,
        ),


        dict(
            x= -0.35, y=3.5, # annotation point
            xref='x1',
            yref='y5',
            showarrow=True,
            arrowhead=3,
            ax= 0,
            ay= 30,
        ),


        dict(
            x= -0.69, y=3, # annotation point
            xref='x1',
            yref='y7',
            showarrow=True,
            arrowhead=3,
            ax= 20,
            ay= 12.54,
        ),

        dict(
            x= -0.68, y=1.5, # annotation point
            xref='x1',
            yref='y9',
            showarrow=True,
            arrowhead=3,
            ax= 25,
            ay= 0,
        ),




        dict(
            x= -0.68, y=0, # annotation point
            xref='x1',
            yref='y11',
            showarrow=True,
            arrowhead=3,
            ax= 25,
            ay= -12.54,
        ),


        dict(
            x= -0.35, y=0, # annotation point
            xref='x1',
            yref='y13',
            showarrow=True,
            arrowhead=3,
            ax= 0,
            ay= -30,
        ),


        dict(
            x= -0.01, y=0.4, # annotation point
            xref='x1',
            yref='y15',
            showarrow=True,
            arrowhead=3,
            ax= -25,
            ay= -15,
        ),
        ])

        plot_handle.for_each_xaxis(lambda x: x.update(showgrid=False))
        plot_handle.for_each_yaxis(lambda x: x.update(showgrid=False))



        plot_handle.show()



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
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1

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
