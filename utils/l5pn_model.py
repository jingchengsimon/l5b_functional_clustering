from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import platform

from neuron import gui, h
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import os
import sys
import threading
import warnings

from utils.morphology_graph import create_directed_graph, set_graph_order
from utils.synaptic_inputs import add_background_exc_inputs, add_background_inh_inputs, add_clustered_inputs
from utils.cable_distance import distance_synapse_mark_compare, recur_dist_to_soma, recur_dist_to_root
from utils.cluster_protocol import generate_indices, generate_vecstim
from utils.random_streams import (
    cluster_assignment_rng,
    cluster_spike_rng,
    preunit_activation_rng,
    synapse_placement_rng,
)


warnings.simplefilter(action='ignore', category=(FutureWarning, RuntimeWarning))
sys.setrecursionlimit(1000000)
sys.path.insert(0, '/G/MIMOlab/Codes/NeuronWithNetworkx/mod')


def _load_local_mechanisms():
    if _mechanisms_available():
        return None

    machine = platform.machine().lower()
    repo_root = Path(__file__).resolve().parents[1]

    if machine in {'arm64', 'aarch64'}:
        candidates = [
            repo_root / 'arm64' / 'libnrnmech.dylib',
            repo_root / 'arm64' / '.libs' / 'libnrnmech.dylib',
        ]
    elif machine in {'x86_64', 'amd64'}:
        candidates = [
            repo_root / 'x86_64' / 'libnrnmech.so',
            repo_root / 'x86_64' / '.libs' / 'libnrnmech.so',
            repo_root / 'mod' / 'x86_64' / '.libs' / 'libnrnmech.so',
        ]
    else:
        candidates = []

    candidates.extend([
        repo_root / 'mod' / 'nrnmech.dll',
        repo_root / 'nrnmech.dll',
    ])

    for candidate in candidates:
        if candidate.exists():
            try:
                h.nrn_load_dll(str(candidate))
            except RuntimeError:
                if _mechanisms_available():
                    return candidate
                raise
            return candidate

    raise FileNotFoundError(
        'Could not find a compiled NEURON mechanism library. '
        'Run nrnivmodl first, or check arm64/x86_64 mechanism outputs.'
    )


def _mechanisms_available():
    probe = h.Section(name='mechanism_probe')
    try:
        probe.insert('CaDynamics_E2')
        probe.insert('NaTa_t')
    except Exception:
        return False
    finally:
        h.delete_section(sec=probe)
    return True

class L5PNModel:
    def __init__(self, swc_file, bg_exc_freq, bg_inh_freq, SIMU_DURATION, STIM_DURATION,
                 bg_syn_pos_seed, bg_spike_gen_seed, clus_spike_gen_seed=None, with_ap=False, with_global_rec=False,
                 clus_syn_pos_seed=None, max_workers_synapse=30):
        """
        Initialize cell with networkx structure.

        Args:
            swc_file: Path to SWC morphology file
            bg_exc_freq: Background excitatory frequency (Hz)
            bg_inh_freq: Background inhibitory frequency (Hz)
            SIMU_DURATION: Simulation duration (ms)
            STIM_DURATION: Stimulation duration (ms)
            bg_syn_pos_seed: Random seed for synapse positioning and background synapse weights.
                Should be fixed across simulations to maintain consistent morphology.
            bg_spike_gen_seed: Random seed for background (bg) spike generation (bg spike trains, pink noise). Used by add_background_*_inputs.
            clus_spike_gen_seed: Random seed for cluster stimulus times (generate_vecstim only).
            clus_syn_pos_seed: Random seed for cluster assignment, preunit activation order (perm),
                and invitro clustered synapse weight draw in add_clustered_inputs.
            with_ap: If True, use L5PCbiophys3withNaCa.hoc (with AP and Ca),
                    else use L5PCbiophys3.hoc (default: False)
            with_global_rec: If True, record v/ina/iNMDA per segment (electrode), save seg_* arrays,
                and after the run compute per-segment NMDA spike rate (Hz) into segment_nmda_spike_rate.npz (default: False)
        """
        h.load_file("import3d.hoc")

        _load_local_mechanisms()

        # Select biophysics file based on with_ap parameter
        biophys_file = './model/L5PCbiophys3withNaCa.hoc' if with_ap else './model/L5PCbiophys3.hoc'
        h.load_file(biophys_file)
        h.load_file('./model/L5PCtemplate.hoc')

        self.complex_cell = h.L5PCtemplate(swc_file)
        h.celsius = 37

        h.v_init = self.complex_cell.soma[0].e_pas  # -90 mV

        self.distance_matrix = None

        (self.num_syn_basal_exc, self.num_syn_apic_exc, self.num_syn_basal_inh,
         self.num_syn_basal_prox_inh, self.num_syn_apic_inh,
         self.num_syn_soma_inh) = (0, 0, 0, 0, 0, 0)

        # Seeds are resolved by the caller before L5PNModel is created.
        self.bg_syn_pos_seed = bg_syn_pos_seed
        self.clus_syn_pos_seed = clus_syn_pos_seed
        self.bg_spike_gen_seed = bg_spike_gen_seed
        self.clus_spike_gen_seed = clus_spike_gen_seed

        self.max_workers_synapse = max_workers_synapse

        if bg_exc_freq != 0:
            self.spike_interval = 1000/bg_exc_freq # interval=1000(ms)/f
        self.FREQ_EXC = bg_exc_freq  # Hz, /s
        self.FREQ_INH = bg_inh_freq  # Hz, /s
        self.SIMU_DURATION = SIMU_DURATION # 1s
        self.STIM_DURATION = STIM_DURATION # 1s

        self.syn_param_exc = [0, 0.3, 1.8] # reversal_potential, tau1, tau2, syn_weight (actually we don't use these params, delete later)
        self.syn_param_inh = [-86, 1, 8, 0.00069] #->0.00069 uS = 0.69 nS

        self.sections_soma = [i for i in map(list, list(self.complex_cell.soma))]
        self.sections_basal = [i for i in map(list, list(self.complex_cell.basal))]
        self.sections_apical = [i for i in map(list, list(self.complex_cell.apical))]
        self.all_sections = self.sections_soma + self.sections_basal + self.sections_apical
        self.all_segments = [seg for sec in h.allsec() for seg in sec]
        self.all_segments_noaxon = [seg for sec in self.all_sections for seg in sec]

        self.section_synapse_df = pd.DataFrame(columns=[
            'section_id_synapse', 'section_synapse', 'segment_synapse', 'loc', 'type',
            'distance_to_soma', 'distance_to_tuft', 'cluster_flag', 'cluster_center_flag',
            'cluster_id', 'pre_unit_id', 'region', 'branch_idx', 'syn_w', 'synapse',
            'netstim', 'netcon', 'spike_train', 'spike_train_bg'
        ], dtype=object)

        # Assigned later in assign_synapse_clusters/run_stimulation_protocol.
        for attr in (
            'basal_channel_type', 'sec_type', 'num_clusters', 'num_clusters_sampled',
            'cluster_radius', 'input_ratio_basal_apic', 'bg_exc_channel_type', 'initW',
            'num_func_group', 'inh_delay', 'num_stim', 'stim_time', 'num_conn_per_preunit',
            'num_preunit', 'unit_ids', 'indices', 'num_syn_inh_list', 'num_activated_preunit_list',
        ):
            setattr(self, attr, None)

        # Recording arrays are initialized in run_stimulation_protocol.
        for attr in (
            'soma_v_array', 'apic_v_array', 'apic_ica_array', 'trunk_v_array',
            'basal_v_array', 'tuft_v_array', 'basal_bg_i_nmda_array', 'basal_bg_i_ampa_array',
            'tuft_bg_i_nmda_array', 'tuft_bg_i_ampa_array', 'dend_v_array', 'dend_i_array',
            'dend_nmda_i_array', 'dend_ampa_i_array', 'dend_nmda_g_array', 'dend_ampa_g_array',
        ):
            setattr(self, attr, None)

        self.with_global_rec = with_global_rec
        for attr in ('seg_v_array', 'seg_ina_array', 'seg_inmda_array'):
            setattr(self, attr, None)

        self.lock = threading.Lock()

        self.section_df = pd.DataFrame(columns=[
            'parent_id', 'section_id', 'parent_name', 'section_name',
            'length', 'branch_idx', 'section_type'
        ])

        self.root_tuft_idx = self.all_sections.index(self.sections_apical[36])
        self.root_tuft_sec = self.sections_apical[36][0].sec
        # create section_df and directed morphology graphs
        self.section_df, self.DiG, self.segment_df, self.segment_DiG = create_directed_graph(
            self.all_sections,
            self.all_segments_noaxon,
            self.section_df,
            return_segment_graph=True,
        )

        # assign the order for each section
        self.class_dict_soma, self.class_dict_tuft = set_graph_order(self.DiG, self.root_tuft_idx)
        self.sec_tuft_idx = list(itertools.chain(*self.class_dict_tuft.values()))

    def initialize_synapse_layout(self, num_syn_basal_exc, num_syn_apic_exc, num_syn_basal_inh,
                                  num_syn_apic_inh, num_syn_soma_inh,
                                  basal_distal_min_um=0.0, num_syn_basal_prox_inh=0):

        if basal_distal_min_um < 0 or num_syn_basal_prox_inh < 0:
            raise ValueError('Basal distance and proximal inhibitory count must be nonnegative')
        if num_syn_basal_prox_inh and basal_distal_min_um <= 0:
            raise ValueError('Proximal basal inhibition requires a positive basal_distal_min_um')

        self.num_syn_basal_exc = num_syn_basal_exc
        self.num_syn_apic_exc = num_syn_apic_exc
        self.num_syn_basal_inh = num_syn_basal_inh
        self.num_syn_basal_prox_inh = num_syn_basal_prox_inh
        self.num_syn_apic_inh = num_syn_apic_inh
        self.num_syn_soma_inh = num_syn_soma_inh

        # add excitatory synapses
        self._sample_synapse_locations(
            num_syn_basal_exc, 'basal', 'exc', min_distance=basal_distal_min_um
        )
        self._sample_synapse_locations(num_syn_apic_exc, 'apical', 'exc')

        # add inhibitory synapses
        self._sample_synapse_locations(
            num_syn_basal_inh, 'basal', 'inh', min_distance=basal_distal_min_um
        )
        self._sample_synapse_locations(
            num_syn_basal_prox_inh, 'basal', 'inh', max_distance=basal_distal_min_um,
            seed_index_offset=num_syn_basal_inh,
        )
        self._sample_synapse_locations(num_syn_apic_inh, 'apical', 'inh')
        self._sample_synapse_locations(num_syn_soma_inh, 'soma', 'inh')

    def assign_synapse_clusters(self, basal_channel_type, sec_type, dis_to_root,
                                  num_clusters, cluster_radius, num_stim, stim_time,
                                  spat_condition, num_conn_per_preunit, num_syn_per_clus,
                                  folder_path):

        # Extract distances
        basal_distance = self.section_synapse_df[
            (self.section_synapse_df['region'] == 'basal') &
            (self.section_synapse_df['type'] == 'A')]['distance_to_soma'].values
        tuft_distance = self.section_synapse_df[
            (self.section_synapse_df['distance_to_tuft'] != -1) &
            (self.section_synapse_df['type'] == 'A')]['distance_to_tuft'].values

        # Sort the distances
        sorted_basal_distances = np.sort(basal_distance)
        sorted_tuft_distances = np.sort(tuft_distance)

        num_syn_thres = [3000 + i * 3000 for i in range(2)] if sec_type == 'basal' else [2500 + i * 2500 for i in range(2)]

        # Get the indices for the thresholds
        dist_thres_basal = [0] + [sorted_basal_distances[threshold - 1] for threshold in num_syn_thres
                                  if threshold <= len(sorted_basal_distances)] + [max(sorted_basal_distances)]

        dist_thres_tuft = [0] + [sorted_tuft_distances[threshold - 1] for threshold in num_syn_thres
                                 if threshold <= len(sorted_tuft_distances)] + [max(sorted_tuft_distances)]

        num_conn_per_preunit = min(num_conn_per_preunit, num_clusters)
        num_preunit = num_syn_per_clus * np.ceil(num_clusters / 3).astype(int)

        clus_loc_rnd = cluster_assignment_rng(self.clus_syn_pos_seed)

        if spat_condition == 'clus':
            # Number of synapses in each cluster is not fixed
            indices = generate_indices(clus_loc_rnd, num_clusters, num_conn_per_preunit, num_preunit)
            self.num_clusters_sampled = num_clusters
        elif spat_condition == 'distr':
            # num_pre*num_conn clus with 1 syn per 'cluster'
            num_clusters = num_preunit * num_conn_per_preunit
            numbers = np.repeat(np.arange(num_preunit), num_conn_per_preunit)
            clus_loc_rnd.shuffle(numbers)
            indices = [[num] for num in numbers]
            self.num_clusters_sampled = min(10, num_clusters)

        self.unit_ids = np.arange(num_preunit)
        self.indices = indices

        # Save assignment
        file_path = os.path.join(folder_path, 'preunit assignment.txt')

        with open(file_path, 'w') as f:
            for i, index_list in enumerate(indices):
                f.write(f"Cluster_id: {i}, Num_preunits: {len(index_list)}, Preunit_ids: {index_list}\n")

        self.basal_channel_type = basal_channel_type
        self.sec_type = sec_type
        self.dis_to_root = dis_to_root
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius

        self.num_stim = num_stim
        self.stim_time = stim_time
        self.num_conn_per_preunit = num_conn_per_preunit
        self.num_preunit = num_preunit

        for i in range(self.num_clusters):

            loop_count = 0

            # Unassigned background synapses for surround synapses
            sec_syn_bg_exc_df = self.section_synapse_df[(self.section_synapse_df['type'] == 'A') &
                                                        (self.section_synapse_df['cluster_flag'] == -1)]

            # Build DataFrame filter: common conditions + sec_type-specific conditions
            bg_exc_cond = (self.section_synapse_df['type'] == 'A') & (self.section_synapse_df['cluster_flag'] == -1)
            sec_specific_cond = (
                (self.section_synapse_df['region'] == 'basal') &
                (self.section_synapse_df['distance_to_soma'].between(dist_thres_basal[dis_to_root], dist_thres_basal[dis_to_root+1]))
            ) if sec_type == 'basal' else (
                (self.section_synapse_df['section_id_synapse'].isin(self.sec_tuft_idx)) &
                (self.section_synapse_df['distance_to_tuft'].between(dist_thres_tuft[dis_to_root], dist_thres_tuft[dis_to_root+1]))
            )
            sec_syn_bg_exc_ordered_df = self.section_synapse_df[bg_exc_cond & sec_specific_cond]

            index_list = indices[i]
            num_syn_per_clus = len(index_list)

            # Loop for cluster assignment
            while True:
                loop_count += 1

                # use the clus_loc_rnd for positioning
                syn_ctr = sec_syn_bg_exc_ordered_df.iloc[clus_loc_rnd.choice(len(sec_syn_bg_exc_ordered_df))]

                # Assign the surround as clustered synapse only if more than 1 syn per cluster (dispersed: 1 syn per cluster)
                if num_syn_per_clus > 1:

                    syn_ctr_sec = syn_ctr['section_synapse']
                    syn_surround_ctr = sec_syn_bg_exc_df[
                        (sec_syn_bg_exc_df['section_synapse'] == syn_ctr_sec) &
                        (sec_syn_bg_exc_df.index != syn_ctr.name)]

                    dis_syn_from_ctr = np.array(np.abs(syn_ctr['loc'] - syn_surround_ctr['loc']) * syn_ctr_sec.L)
                    # use exponential distribution to generate loc

                    max_num_syn_per_clus = max(num_syn_per_clus, 100)

                    try:
                        max_dis_mark_from_ctr = np.sort(clus_loc_rnd.exponential(cluster_radius, max_num_syn_per_clus - 1))
                    except ValueError:
                        max_dis_mark_from_ctr = np.sort(clus_loc_rnd.exponential(cluster_radius, 0))

                    # not enough synapses on the same section
                    syn_ctr_sec_id = syn_ctr['section_id_synapse']
                    syn_suc_sec_id = syn_ctr_sec_id
                    syn_pre_sec_id = syn_ctr_sec_id

                    exceed_flag = False

                    while len(dis_syn_from_ctr) < max_num_syn_per_clus - 1:

                        # Check and empty syn_pre_surround_ctr and syn_suc_surround_ctr if they exist
                        if 'syn_pre_surround_ctr' in locals():
                            syn_pre_surround_ctr = syn_pre_surround_ctr.iloc[0:0]

                        if 'syn_suc_surround_ctr' in locals():
                            syn_suc_surround_ctr = syn_suc_surround_ctr.iloc[0:0]

                        # Check and empty dis_syn_pre_from_ctr and dis_syn_suc_from_ctr if they exist
                        if 'dis_syn_pre_from_ctr' in locals():
                            dis_syn_pre_from_ctr = np.array([])

                        if 'dis_syn_suc_from_ctr' in locals():
                            dis_syn_suc_from_ctr = np.array([])

                        # the children section of the center section
                        if list(self.DiG.successors(syn_suc_sec_id)):
                            # iterate
                            syn_suc_sec_id = clus_loc_rnd.choice(list(self.DiG.successors(syn_suc_sec_id)))
                            try:
                                syn_suc_sec = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_suc_sec_id]['section_synapse'].values[0]
                                syn_suc_surround_ctr = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_suc_sec_id]
                                dis_syn_suc_from_ctr = np.array((1 - syn_ctr['loc']) * syn_ctr_sec.L + syn_suc_surround_ctr['loc'] * syn_suc_sec.L)
                            except IndexError:
                                pass

                        # the parent section of the center section
                        # there is no dendritic section on the soma, so we should not choose soma as the parent section
                        # also don't choose the apical nexus section as the parent section
                        if list(self.DiG.predecessors(syn_pre_sec_id)) not in ([], [0], [121]):
                            syn_pre_sec_id = clus_loc_rnd.choice(list(self.DiG.predecessors(syn_pre_sec_id)))
                            try:
                                syn_pre_sec = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_pre_sec_id]['section_synapse'].values[0]
                                syn_pre_surround_ctr = sec_syn_bg_exc_df[sec_syn_bg_exc_df['section_id_synapse'] == syn_pre_sec_id]
                                dis_syn_pre_from_ctr = np.array(syn_ctr['loc'] * syn_ctr_sec.L + (1 - syn_pre_surround_ctr['loc']) * syn_pre_sec.L)
                            except IndexError:
                                # print(f"IndexError: syn_pre_sec_id: {syn_pre_sec_id}")
                                pass


                        arr_to_concat, df_to_concat = [], []

                        # Combine conditions and append to lists
                        for dis_syn, syn_surround in [
                            ('dis_syn_from_ctr', 'syn_surround_ctr'),
                            ('dis_syn_suc_from_ctr', 'syn_suc_surround_ctr'),
                            ('dis_syn_pre_from_ctr', 'syn_pre_surround_ctr')
                        ]:
                            if dis_syn in locals() and syn_surround in locals():
                                arr_to_concat.append(locals()[dis_syn])
                                df_to_concat.append(locals()[syn_surround])

                        # Concatenate arrays and dataframes if not empty
                        if arr_to_concat:
                            dis_syn_from_ctr = np.concatenate(arr_to_concat)

                        if df_to_concat:
                            syn_surround_ctr = pd.concat(df_to_concat)

                        # after the loop, if the pre of pre and suc of suc exceed the sec_id of the sec_syn_bg_exc_ordered_df but the len(dis_syn_from_ctr) still does not reach the standard,
                        # break the loop and re-choose the syn_ctr (the chosen one be reset to type 'A')
                        suc_exceed_flag = (list(self.DiG.successors(syn_suc_sec_id)) == []) or (not any(sec_id in np.unique(sec_syn_bg_exc_ordered_df['section_id_synapse'])
                                                                                                        for sec_id in list(self.DiG.successors(syn_suc_sec_id))))
                        pre_exceed_flag = not any(sec_id in np.unique(sec_syn_bg_exc_ordered_df['section_id_synapse'])
                                                  for sec_id in list(self.DiG.predecessors(syn_pre_sec_id)))
                        exceed_flag = suc_exceed_flag and pre_exceed_flag and (len(dis_syn_from_ctr) < max_num_syn_per_clus - 1)

                        if exceed_flag:
                            break

                    if exceed_flag:
                        continue

                    max_clus_mem_idx = distance_synapse_mark_compare(dis_syn_from_ctr, max_dis_mark_from_ctr)
                    clus_mem_idx = clus_loc_rnd.choice(max_clus_mem_idx, num_syn_per_clus - 1, replace=False)

                    # assign the surround as clustered synapse
                    self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index, 'cluster_flag'] = 1
                    self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index, 'cluster_center_flag'] = 0
                    self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index, 'cluster_id'] = i
                    for j in range(len(clus_mem_idx)):
                        try:
                            self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index[j], 'pre_unit_id'] = index_list[j+1]
                        except IndexError:
                            self.section_synapse_df.loc[syn_surround_ctr.iloc[clus_mem_idx].index[j], 'pre_unit_id'] = -1
                break

            # assign the center as clustered synapse
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_flag'] = 1
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_center_flag'] = 1
            self.section_synapse_df.loc[syn_ctr.name, 'cluster_id'] = i
            try:
                self.section_synapse_df.loc[syn_ctr.name, 'pre_unit_id'] = index_list[0]
            except IndexError:
                self.section_synapse_df.loc[syn_ctr.name, 'pre_unit_id'] = -1

            if i < 10:
                if num_syn_per_clus > 1:
                    print(np.unique(syn_surround_ctr['section_id_synapse']))
                else:
                    print(np.unique(syn_ctr['section_id_synapse']))

    def run_stimulation_protocol(self, folder_path, simu_condition, input_ratio_basal_apic, bg_exc_channel_type,
                   initW, num_func_group, inh_delay, num_trials,
                   use_fixedW=False, fixedW=0.0004, expected=False,
                   aff_mode='linear', aff_list=None, iter_step=2):

        self.input_ratio_basal_apic = input_ratio_basal_apic
        self.bg_exc_channel_type = bg_exc_channel_type
        self.initW = initW
        self.num_func_group = num_func_group
        self.inh_delay = inh_delay
        self.use_fixedW = use_fixedW
        self.fixedW = fixedW

        # Determine spat_condition and num_clus_condition based on folder_path
        if 'distr' in folder_path:
            spat_condition, num_clus_condition = 'distr', 'multi' if 'multiclus' in folder_path else 'single'
            section_synapse_df_clus = pd.read_csv(os.path.join(folder_path.replace('distr', 'clus'), 'section_synapse_df.csv'))
        elif 'multiclus' in folder_path:
            spat_condition, num_clus_condition = 'clus', 'multi'
            folder_path_clus = folder_path.replace('multiclus_3', 'singclus').replace('/G/results/simulation_multiclus_Oct25/', '/mnt/mimo_1/simu_results_sjc/simulation_singclus_Aug25/')
            parts = list(Path(folder_path_clus).parts)
            parts[-2] = '1'
            section_synapse_df_clus = pd.read_csv(os.path.join(Path(*parts).as_posix(), 'section_synapse_df.csv'))
        else:
            spat_condition, num_clus_condition, section_synapse_df_clus = 'clus', 'single', self.section_synapse_df

        # Cluster stimulus times: clus_spike_gen_seed; preunit activation order (perm): clus_syn_pos_seed
        clus_spk_rnd = cluster_spike_rng(self.clus_spike_gen_seed)
        clus_pos_rnd = preunit_activation_rng(self.clus_syn_pos_seed)

        spt_unit_array_list = []
        stim_time_var = 5
        for num_stim in range(1, self.num_stim + 1):
            spt_unit_array = generate_vecstim(clus_spk_rnd, self.unit_ids, num_stim, self.stim_time, stim_time_var)
            spt_unit_array_list.append(spt_unit_array)

        perm = clus_pos_rnd.permutation(self.num_preunit)

        ## Rearrange the perm to always start with the first syn of the first cluster
        pre_unit_id_first_syn = self.section_synapse_df[(self.section_synapse_df['cluster_center_flag'] == 1) &
                                                                (self.section_synapse_df['cluster_id'] == 0)]['pre_unit_id'].values[0]
        perm_list = perm.tolist()
        if pre_unit_id_first_syn in perm_list:
            perm_list.remove(pre_unit_id_first_syn)
            perm_list = [pre_unit_id_first_syn] + perm_list
        perm = np.array(perm_list)

        self.num_syn_inh_list = [self.num_syn_basal_inh, self.num_syn_apic_inh, self.num_syn_soma_inh]

        # create an ndarray to store the voltage of each cluster of each trial
        num_time_points = 1 + 40 * self.SIMU_DURATION

        if iter_step <= 0:
            raise ValueError('iter_step must be positive')

        effective_iter_step = 1 if expected else iter_step

        def include_zero_aff_axis(aff_values):
            return list(dict.fromkeys([0] + list(aff_values)))

        if aff_mode == 'linear':
            self.num_activated_preunit_list = list(range(0, self.num_preunit + 1, effective_iter_step))
        elif aff_mode == 'curve':
            # Dense first, then sparse increments, useful for resolving the low-input regime.
            dense_part = list(range(0, effective_iter_step + 1))
            sparse_part = list(range(effective_iter_step, self.num_preunit + 1, effective_iter_step))
            self.num_activated_preunit_list = sorted(list(set(dense_part + sparse_part)))
        elif aff_mode == 'full':
            self.num_activated_preunit_list = [0, self.num_preunit]
        elif aff_mode == 'custom':
            if aff_list is None or len(aff_list) == 0:
                raise ValueError('aff_list must be provided when aff_mode is custom')
            invalid_aff = [n for n in aff_list if n < 0 or n > self.num_preunit]
            if invalid_aff:
                raise ValueError(
                    f'aff_list contains values outside valid range 0..{self.num_preunit}: {invalid_aff}'
                )
            self.num_activated_preunit_list = list(aff_list)
        else:
            raise ValueError(f'Unknown aff_mode: {aff_mode}')

        self.num_activated_preunit_list = include_zero_aff_axis(self.num_activated_preunit_list)
        num_aff_fibers = len(self.num_activated_preunit_list)

        # Initialize arrays with common shape
        common_shape = (num_time_points, self.num_stim, num_aff_fibers, num_trials)
        dend_shape = (self.num_clusters_sampled, *common_shape)

        # Initialize arrays concisely using dictionary and setattr
        voltage_arrays = ['soma_v', 'apic_v', 'apic_ica', 'soma_i', 'trunk_v', 'basal_v', 'tuft_v']
        bg_current_arrays = ['basal_bg_i_nmda', 'basal_bg_i_ampa', 'tuft_bg_i_nmda', 'tuft_bg_i_ampa']
        dend_arrays = ['dend_v', 'dend_i', 'dend_nmda_i', 'dend_ampa_i', 'dend_nmda_g', 'dend_ampa_g']
        for arr_name in voltage_arrays + bg_current_arrays:
            setattr(self, f'{arr_name}_array', np.zeros(common_shape))
        for arr_name in dend_arrays:
            setattr(self, f'{arr_name}_array', np.zeros(dend_shape))

        if self.with_global_rec:
            num_segments_noaxon = len(self.all_segments_noaxon)
            seg_global_shape = (num_segments_noaxon, num_time_points, self.num_stim, num_aff_fibers, num_trials)
            self.seg_v_array = np.zeros(seg_global_shape)
            self.seg_ina_array = np.zeros(seg_global_shape)
            self.seg_inmda_array = np.zeros(seg_global_shape)

        if simu_condition == 'invivo':
            add_background_exc_inputs(self.section_synapse_df, self.syn_param_exc, self.SIMU_DURATION, self.FREQ_EXC,
                                    self.input_ratio_basal_apic, self.bg_exc_channel_type, self.initW, self.num_func_group,
                                    self.bg_syn_pos_seed, self.bg_spike_gen_seed, spat_condition, num_clus_condition, section_synapse_df_clus,
                                    self.max_workers_synapse,
                                    use_fixedW=self.use_fixedW, fixedW=self.fixedW)

        for num_activated_preunit in self.num_activated_preunit_list:
            for num_stim in range(self.num_stim):
                for num_trial in range(num_trials): # 1

                    spt_unit_array = spt_unit_array_list[num_stim]
                    spt_unit_array_truncated = spt_unit_array[perm[:num_activated_preunit]]

                    if expected and num_activated_preunit > 0:
                        # For expected input
                        spt_unit_array_truncated = spt_unit_array[perm[:num_activated_preunit][-1]]

                    add_clustered_inputs(self.section_synapse_df, self.num_clusters, self.basal_channel_type,
                                         self.initW, spt_unit_array_truncated, self.clus_syn_pos_seed, self.num_preunit,
                                         use_fixedW=self.use_fixedW, fixedW=self.fixedW)


                    # Add background inputs for in vivo-like condition
                    if simu_condition == 'invivo':
                        num_activated_preunit_idx = self.num_activated_preunit_list.index(num_activated_preunit)
                        add_background_inh_inputs(self.section_synapse_df, self.syn_param_inh, self.SIMU_DURATION, self.FREQ_INH,
                                                self.inh_delay, self.bg_spike_gen_seed, spat_condition, num_clus_condition,
                                                section_synapse_df_clus, num_activated_preunit_idx,
                                                self.max_workers_synapse)

                for num_trial in range(num_trials):
                    num_aff_idx = self.num_activated_preunit_list.index(num_activated_preunit)

                    self._run_single_trial(num_stim, num_aff_idx, num_trial, folder_path)

        # Save arrays efficiently
        arrays_to_save = {
            'soma_v_array': self.soma_v_array, 'apic_v_array': self.apic_v_array, 'apic_ica_array': self.apic_ica_array,
            'soma_i_array': self.soma_i_array, 'trunk_v_array': self.trunk_v_array, 'basal_v_array': self.basal_v_array,
            'tuft_v_array': self.tuft_v_array, 'basal_bg_i_nmda_array': self.basal_bg_i_nmda_array,
            'basal_bg_i_ampa_array': self.basal_bg_i_ampa_array, 'tuft_bg_i_nmda_array': self.tuft_bg_i_nmda_array,
            'tuft_bg_i_ampa_array': self.tuft_bg_i_ampa_array, 'dend_v_array': self.dend_v_array,
            'dend_i_array': self.dend_i_array, 'dend_nmda_i_array': self.dend_nmda_i_array,
            'dend_ampa_i_array': self.dend_ampa_i_array, 'dend_nmda_g_array': self.dend_nmda_g_array,
            'dend_ampa_g_array': self.dend_ampa_g_array
        }
        if self.with_global_rec:
            if self.seg_v_array is not None:
                arrays_to_save['seg_v_array'] = self.seg_v_array
            if self.seg_ina_array is not None:
                arrays_to_save['seg_ina_array'] = self.seg_ina_array
            if self.seg_inmda_array is not None:
                arrays_to_save['seg_inmda_array'] = self.seg_inmda_array

        for name, array in arrays_to_save.items():
            np.save(os.path.join(folder_path, f'{name}.npy'), array)

        self.section_synapse_df.to_csv(os.path.join(folder_path, 'section_synapse_df.csv'), index=False)

    def _sample_synapse_locations(self, num_syn, region, sim_type, *, min_distance=0.0,
                                  max_distance=None, seed_index_offset=0):

        type = 'A' if sim_type == 'exc' else 'B'

        region_mapping = {
            'basal': (self.sections_basal, 'dend'),
            'apical': (self.sections_apical, 'apic'),
            'soma': (self.sections_soma, 'soma')
        }
        sections, section_type = region_mapping[region]
        # Cast section lengths to float before normalization. `section_df` keeps mixed dtypes
        # (object-heavy rows), and np.random.Generator.choice requires numeric probabilities.
        section_length = (
            self.section_df.loc[self.section_df['section_type'] == section_type, 'length']
            .astype(float)
            .to_numpy(dtype=np.float64)
        )
        weights = section_length / float(section_length.sum())

        def generate_synapse(i):
            syn_rnd = synapse_placement_rng(
                self.bg_syn_pos_seed, region, sim_type, i + seed_index_offset
            )
            for _ in range(10000):
                section = sections[syn_rnd.choice(len(sections), p=weights)][0].sec
                loc = float(syn_rnd.uniform())
                distance_to_soma = recur_dist_to_soma(section, loc)
                if distance_to_soma >= min_distance and (
                    max_distance is None or distance_to_soma < max_distance
                ):
                    break
            else:
                raise RuntimeError(
                    f'Could not sample {region} {sim_type} synapse in distance interval '
                    f'[{min_distance}, {max_distance})'
                )

            section_name = section.psection()['name']
            section_id_synapse = self.section_df.loc[self.section_df['section_name'] == section_name, 'section_id'].iat[0]
            branch_idx = self.section_df.loc[self.section_df['section_name'] == section_name, 'branch_idx'].iat[0]
            segment_synapse = section(loc)
            distance_to_tuft = recur_dist_to_root(section, loc, self.root_tuft_sec) if section_id_synapse in self.sec_tuft_idx else -1

            return {
                'section_id_synapse': section_id_synapse, 'section_synapse': section, 'segment_synapse': segment_synapse,
                'loc': loc, 'type': type, 'distance_to_soma': distance_to_soma, 'distance_to_tuft': distance_to_tuft,
                'cluster_flag': -1, 'cluster_center_flag': -1, 'cluster_id': -1, 'pre_unit_id': -1,
                'region': region, 'branch_idx': branch_idx, 'syn_w': None, 'synapse': None,
                'netstim': None, 'netcon': None, 'spike_train': [], 'spike_train_bg': []
            }

        with ThreadPoolExecutor(max_workers=self.max_workers_synapse) as executor:
            rows = list(tqdm(executor.map(generate_synapse, range(num_syn)), total=num_syn))
        if rows:
            self.section_synapse_df = pd.concat(
                [self.section_synapse_df, pd.DataFrame(rows, dtype=object)],
                ignore_index=True,
            )

    def _build_segment_metadata(self):
        """
        Per-segment metadata in the same order as all_segments_noaxon / seg_v_array axis 0.
        region: 'soma', 'basal' (dend), or 'apical' (apic); distance_to_tuft is -1 if not on tuft subtree.
        """
        n = len(self.all_segments_noaxon)
        segment_index = np.arange(n, dtype=np.int32)
        distance_to_soma = np.zeros(n, dtype=np.float64)
        distance_to_tuft = np.full(n, -1.0, dtype=np.float64)
        region = np.empty(n, dtype=object)

        sec_to_sid = {}
        for sid, segs in enumerate(self.all_sections):
            if len(segs):
                sec_to_sid[id(segs[0].sec)] = sid

        type_to_region = {'dend': 'basal', 'apic': 'apical', 'soma': 'soma'}

        for i, seg in enumerate(self.all_segments_noaxon):
            sid = sec_to_sid.get(id(seg.sec))
            if sid is None:
                raise ValueError('segment section not found in all_sections')
            st = self.section_df.iloc[sid]['section_type']
            region[i] = type_to_region.get(st, str(st))
            distance_to_soma[i] = recur_dist_to_soma(seg.sec, seg.x)
            if sid in self.sec_tuft_idx:
                distance_to_tuft[i] = recur_dist_to_root(seg.sec, seg.x, self.root_tuft_sec)

        return {
            'segment_index': segment_index,
            'distance_to_soma': distance_to_soma,
            'distance_to_tuft': distance_to_tuft,
            'region': region,
        }

    def _run_single_trial(self, num_stim, num_aff_fiber, num_trial, folder_path):

        soma_v = h.Vector().record(self.complex_cell.soma[0](0.5)._ref_v)
        apic_v = h.Vector().record(self.complex_cell.apic[121-85](1)._ref_v)
        apic_ica = h.Vector().record(self.complex_cell.apic[121-85](1)._ref_ica)

        trunk_v = h.Vector().record(self.complex_cell.apic[3](0)._ref_v)
        basal_v = h.Vector().record(self.complex_cell.apic[71-1](0.8)._ref_v)
        tuft_v = h.Vector().record(self.complex_cell.apic[152-85](0.5)._ref_v) # the 152th dendrite (tip), L: 192.8, order: 3, distance to root: 565.0

        # EPSC record (VClamp)
        vc = h.SEClamp(self.complex_cell.soma[0](0.5))
        # vc.dur1 = 1000  # Long duration to hold the voltage
        # vc.amp1 = 60   # Holding voltage at 60 mV
        soma_i = h.Vector().record(vc._ref_i)

        try:
            # Record summed local background NMDA current at the basal tip branch
            exc_syn_on_basal_sec = self.section_synapse_df[(self.section_synapse_df['section_id_synapse'] == 71) &
                                                        (self.section_synapse_df['type'] == 'A')]['synapse']
            basal_bg_i_nmda_list = []
            basal_bg_i_ampa_list = []

            for exc_syn in exc_syn_on_basal_sec:

                try:
                    basal_bg_i_nmda = h.Vector().record(exc_syn._ref_i_NMDA)
                except AttributeError:
                    basal_bg_i_nmda = h.Vector().record(exc_syn._ref_i_AMPA)

                basal_bg_i_ampa = h.Vector().record(exc_syn._ref_i_AMPA)

                basal_bg_i_nmda_list.append(basal_bg_i_nmda)
                basal_bg_i_ampa_list.append(basal_bg_i_ampa)

            # Record summed local background NMDA current at the tuft tip branch
            exc_syn_on_tuft_sec = self.section_synapse_df[(self.section_synapse_df['section_id_synapse'] == 152) &
                                                        (self.section_synapse_df['type'] == 'A')]['synapse']
            tuft_bg_i_nmda_list = []
            tuft_bg_i_ampa_list = []

            for exc_syn in exc_syn_on_tuft_sec:

                try:
                    tuft_bg_i_nmda = h.Vector().record(exc_syn._ref_i_NMDA)
                except AttributeError:
                    tuft_bg_i_nmda = h.Vector().record(exc_syn._ref_i_AMPA)

                tuft_bg_i_ampa = h.Vector().record(exc_syn._ref_i_AMPA)

                tuft_bg_i_nmda_list.append(tuft_bg_i_nmda)
                tuft_bg_i_ampa_list.append(tuft_bg_i_ampa)

        except AttributeError:
            pass

        # Record center synapse voltage and current at each cluster
        dend_v_list = []
        dend_i_list_list = []
        dend_i_nmda_list_list = []
        dend_i_ampa_list_list = []
        dend_g_nmda_list_list = []
        dend_g_ampa_list_list = []

        for cluster_id in range(self.num_clusters_sampled):

            # choose the center synapse of each cluster (spatial condition: clus)
            cluster_ctr = self.section_synapse_df[(self.section_synapse_df['cluster_id'] == cluster_id) &
                                                (self.section_synapse_df['cluster_center_flag'] == 1)]['segment_synapse'].values[0]

            dend_v = h.Vector().record(cluster_ctr._ref_v)

            clustered_sec = np.unique(self.section_synapse_df[self.section_synapse_df['cluster_id'] == cluster_id]['section_synapse'])
            exc_syn_on_clus_sec = self.section_synapse_df[(self.section_synapse_df['section_synapse'].isin(clustered_sec)) &
                                                            (self.section_synapse_df['type'].isin(['A']))]['synapse']
            exc_syn_on_clus_sec_filt = list(filter(None, exc_syn_on_clus_sec)) # Not work: exc_syn_on_clus_sec[exc_syn_on_clus_sec!=None]

            dend_i_list = []
            dend_i_nmda_list = []
            dend_i_ampa_list = []
            dend_g_nmda_list = []
            dend_g_ampa_list = []

            for exc_syn in exc_syn_on_clus_sec_filt:

                dend_i = h.Vector().record(exc_syn._ref_i)

                try:
                    dend_i_nmda = h.Vector().record(exc_syn._ref_i_NMDA)
                    dend_g_nmda = h.Vector().record(exc_syn._ref_g_NMDA)
                except AttributeError:
                    dend_i_nmda = h.Vector().record(exc_syn._ref_i_AMPA)
                    dend_g_nmda = h.Vector().record(exc_syn._ref_g_AMPA)

                dend_i_ampa = h.Vector().record(exc_syn._ref_i_AMPA)
                dend_g_ampa = h.Vector().record(exc_syn._ref_g_AMPA)

                dend_i_list.append(dend_i)
                dend_i_nmda_list.append(dend_i_nmda)
                dend_i_ampa_list.append(dend_i_ampa)
                dend_g_nmda_list.append(dend_g_nmda)
                dend_g_ampa_list.append(dend_g_ampa)

            dend_v_list.append(dend_v)
            dend_i_list_list.append(dend_i_list)
            dend_i_nmda_list_list.append(dend_i_nmda_list)
            dend_i_ampa_list_list.append(dend_i_ampa_list)
            dend_g_nmda_list_list.append(dend_g_nmda_list)
            dend_g_ampa_list_list.append(dend_g_ampa_list)


        # Repertoire of seg_v (voltage), seg_ina (Na current, built-in) and per-segment iNMDA; only when global recording is enabled
        seg_v = None
        seg_ina = None
        seg_inmda_vectors = None
        if self.with_global_rec:
            seg_v = [h.Vector().record(seg._ref_v) for seg in self.all_segments_noaxon]
            seg_ina = [h.Vector().record(seg._ref_ina) for seg in self.all_segments_noaxon]
            seg_to_syns = defaultdict(list)
            for _, row in self.section_synapse_df[self.section_synapse_df['type'] == 'A'].iterrows():
                seg_syn = row['segment_synapse']
                syn = row['synapse']
                if syn is not None and seg_syn is not None:
                    sec = seg_syn.sec
                    nseg = sec.nseg
                    seg_idx = min(int(seg_syn.x * nseg), nseg - 1)
                    seg_to_syns[(id(sec), seg_idx)].append(syn)
            seg_inmda_vectors = []
            for seg in self.all_segments_noaxon:
                seg_idx = min(int(seg.x * seg.sec.nseg), seg.sec.nseg - 1)
                key = (id(seg.sec), seg_idx)
                syns_on_seg = seg_to_syns.get(key, [])
                vecs = []
                for exc_syn in syns_on_seg:
                    try:
                        vecs.append(h.Vector().record(exc_syn._ref_i_NMDA))
                    except AttributeError:
                        vecs.append(h.Vector().record(exc_syn._ref_i_AMPA))
                seg_inmda_vectors.append(vecs)

        # Simulate the full neuron for 1 seconds
        h.tstop = self.SIMU_DURATION
        h.run()

        seg_inmda = None
        if self.with_global_rec and seg_inmda_vectors is not None:
            n_t = int(soma_v.size())
            seg_inmda = []
            for vecs in seg_inmda_vectors:
                if vecs:
                    seg_inmda.append(np.sum([np.array(v) for v in vecs], axis=0))
                else:
                    seg_inmda.append(np.zeros(n_t))

        with self.lock:

            self.soma_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(soma_v)
            self.apic_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(apic_v)
            self.apic_ica_array[:, num_stim, num_aff_fiber, num_trial] = np.array(apic_ica)

            self.soma_i_array[:, num_stim, num_aff_fiber, num_trial] = np.array(soma_i)

            self.trunk_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(trunk_v)
            self.basal_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(basal_v)
            self.tuft_v_array[:, num_stim, num_aff_fiber, num_trial] = np.array(tuft_v)

            try:
                self.basal_bg_i_nmda_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(basal_bg_i_nmda_list), axis=0)
                self.basal_bg_i_ampa_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(basal_bg_i_ampa_list), axis=0)
                self.tuft_bg_i_ampa_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(tuft_bg_i_ampa_list), axis=0)
                self.tuft_bg_i_nmda_array[:, num_stim, num_aff_fiber, num_trial] = np.average(np.array(tuft_bg_i_nmda_list), axis=0)

            except UnboundLocalError:
                pass

            for cluster_id in range(self.num_clusters_sampled):
                self.dend_v_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.array(dend_v_list[cluster_id])
                self.dend_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_i_list_list[cluster_id]), axis=0) # sum, not average
                self.dend_nmda_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_i_nmda_list_list[cluster_id]), axis=0)
                self.dend_nmda_g_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_g_nmda_list_list[cluster_id]), axis=0)

                self.dend_ampa_i_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_i_ampa_list_list[cluster_id]), axis=0)
                self.dend_ampa_g_array[cluster_id, :, num_stim, num_aff_fiber, num_trial] = np.sum(np.array(dend_g_ampa_list_list[cluster_id]), axis=0)

            if self.with_global_rec and seg_v is not None and seg_ina is not None and seg_inmda is not None:
                self.seg_v_array[:, :, num_stim, num_aff_fiber, num_trial] = np.array([list(v) for v in seg_v])
                self.seg_ina_array[:, :, num_stim, num_aff_fiber, num_trial] = np.array([list(v) for v in seg_ina])
                self.seg_inmda_array[:, :, num_stim, num_aff_fiber, num_trial] = np.array(seg_inmda)

        return True
