import argparse
import itertools
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.random_streams import SEED_FIELDS, resolve_workflow_seeds

# main function
swc_file_path = './model/cell1.asc'

def create_parser():
    """Create and configure argument parser with default values from utils"""
    parser = argparse.ArgumentParser(description='Neuron simulation parameters')
    
    # Synapse numbers
    parser.add_argument('--num_syn_basal_exc', type=int, default=10042,
                        help='Number of basal excitatory synapses (default: 10042)')
    parser.add_argument('--num_syn_apic_exc', type=int, default=16070,
                        help='Number of apical excitatory synapses (default: 16070)')
    parser.add_argument('--num_syn_basal_inh', type=int, default=1023,
                        help='Number of basal inhibitory synapses (default: 1023)')
    parser.add_argument('--num_syn_apic_inh', type=int, default=1637,
                        help='Number of apical inhibitory synapses (default: 1637)')
    parser.add_argument('--num_syn_soma_inh', type=int, default=150,
                        help='Number of soma inhibitory synapses (default: 150)')
    parser.add_argument('--basal_distal_min_um', type=float, default=0.0,
                        help='Minimum soma cable distance for the main basal excitatory and inhibitory pools (default: 0)')
    parser.add_argument('--num_syn_basal_prox_inh', type=int, default=0,
                        help='Additional basal inhibitory synapses below basal_distal_min_um (default: 0)')
    
    # Simulation duration
    parser.add_argument('--simu_duration', type=int, default=1000,
                        help='Simulation duration in ms (default: 1000)')
    parser.add_argument('--stim_duration', type=int, default=1000,
                        help='Stimulation duration in ms (default: 1000)')
    parser.add_argument('--stim_time', type=int, default=500,
                        help='Time point of stimulation in ms (default: 500)')
    parser.add_argument('--num_stim', type=int, default=1,
                        help='Number of stimuli (default: 1)')
    
    # Channel types
    parser.add_argument('--basal_channel_type', type=str, default='AMPANMDA',
                        choices=['AMPANMDA', 'AMPA'],
                        help='Basal channel type (default: AMPANMDA)')
    parser.add_argument('--bg_exc_channel_type', type=str, default='AMPANMDA',
                        choices=['AMPANMDA', 'AMPA'],
                        help='Background excitatory channel type (default: AMPANMDA)')
    parser.add_argument('--channel_suffix', type=str, default='singclus',
                        help='Base channel suffix for simulation folder name (e.g. singclus, singclus_AMPA). '
                             'With --with_ap / --with_global_rec, _ap / _globrec are appended (default: singclus)')
    
    # Cluster parameters
    parser.add_argument('--cluster_radius', type=float, default=5.0,
                        help='Cluster radius in um (default: 5.0)')
    parser.add_argument('--num_clusters', type=int, default=1,
                        help='Number of clusters (default: 1)')
    parser.add_argument('--num_syn_per_clus', type=int, default=72,
                        help='Number of synapses per cluster (default: 72)')
    parser.add_argument('--num_conn_per_preunit', type=int, default=3,
                        help='Number of connections per preunit (default: 3)')
    
    # Simulation parameters
    parser.add_argument('--simu_cond', type=str, nargs='+', default=['invivo'],
                        choices=['invivo', 'invitro'],
                        help='Simulation condition (default: invivo)')
    parser.add_argument('--spat_cond', type=str, nargs='+', default=['clus'],
                        choices=['clus', 'distr'],
                        help='Spatial condition: clus (clustered) or distr (distributed). '
                             'When both are given, they run serially in the given order. '
                             '(default: clus)')
    parser.add_argument('--sec_type', type=str, nargs='+', default=['basal'],
                        choices=['basal', 'apical'],
                        help='Section type (default: basal)')
    parser.add_argument('--dis_to_root', type=int, nargs='+', default=[0],
                        help='Distance from clusters to root (default: 0)')

    # Simulation modes
    parser.add_argument('--expected', action='store_true', default=False,
                        help='Use expected/linear-sum cluster stimulus logic. '
                             'If set, iter_step is forced to 1 in run_stimulation_protocol (default: False)')
    parser.add_argument('--aff_mode', type=str, default='linear',
                        choices=['linear', 'curve', 'full', 'custom'],
                        help='Activation mode across preunits: linear uses range(0, N+1, iter_step); '
                             'curve uses dense first then sparse increments; full runs 0 and N preunits; '
                             'custom uses aff_list as the within-run aff axis and auto-adds 0 if omitted '
                             '(default: linear)')
    parser.add_argument('--aff_list', type=int, nargs='+', default=None,
                        help='Custom activated-preunit counts for aff_mode=custom, e.g. --aff_list 4 24 48 72. '
                             'This list is used within each run and does not expand parameter combinations.')
    parser.add_argument('--iter_step', type=int, default=2,
                        help='Step size for aff_mode linear/curve. Ignored by full; forced to 1 when --expected is set (default: 2)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to run (default: 1)')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Starting epoch index for epoch execution (default: 1)')
    parser.add_argument('--max_workers_epoch', type=int, default=20,
                        help='Max parallel build_cell processes per spat_cond (default: 20)')
    parser.add_argument('--max_workers_synapse', type=int, default=30,
                        help='Max threads per process for synapse/input prep (default: 30)')
    
    # Background input parameters
    parser.add_argument('--bg_exc_freq', type=float, default=1.0,
                        help='Background excitatory frequency in Hz (default: 1.0)')
    parser.add_argument('--bg_inh_freq', type=float, default=4.0,
                        help='Background inhibitory frequency in Hz (default: 4.0)')
    parser.add_argument('--input_ratio_basal_apic', type=float, default=1.0,
                        help='Input ratio of basal to apical (default: 1.0)')
    
    # Synaptic weight parameters
    parser.add_argument(
        '--initW',
        type=float,
        default=0.0004,
        help='When use_fixedW is off: mean of log-normal excitatory weights (sigma=1, '
             'mu=log(initW)-0.5*sigma^2 so E[weight]=initW). When use_fixedW is on: not used for '
             'those weights (they are fixedW). Units: uS (default: 0.0004).',
    )
    parser.add_argument(
        '--use_fixedW',
        action='store_true',
        default=False,
        help='If set, all excitatory synapse weights that would be log-normal draws use fixedW '
             'instead; initW does not affect those weights. Default: off (log-normal from initW).',
    )
    parser.add_argument(
        '--fixedW',
        type=float,
        default=0.0004,
        help='Excitatory synapse weight (uS) when --use_fixedW is set. Does not change the '
             'log-normal distribution when use_fixedW is off (default: 0.0004).',
    )
    parser.add_argument('--num_func_group', type=int, default=10,
                        help='Number of functional groups (default: 10)')
    parser.add_argument('--inh_delay', type=float, default=4.0,
                        help='Delay of inhibitory inputs in ms (default: 4.0)')
    
    # Other parameters
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials (default: 1)')
    parser.add_argument('--folder_tag', type=str, default='1',
                        help='Folder tag for output (default: 1)')
    parser.add_argument(
        '--results_root',
        type=str,
        default='/G/results/simulation_singclus_supple_Jun26',
        help='Root directory for simulation outputs. Each run writes to '
             '{results_root}/{sec_type}_range{N}_{spat}_{simu}{suffix}/[{folder_tag}/]{epoch}/ '
             '(default: /G/results/simulation_singclus_supple_Jun26)',
    )
    # Random seeds - default to epoch value
    parser.add_argument('--bg_syn_pos_seed', type=int, nargs='+', default=None,
                        help='Random seed for background synapse positioning and background synapse weights. '
                             'Accepts one or more values; multiple values expand into separate runs. '
                             'If None, uses syn_pos_seed when provided, otherwise epoch value (default: None)')
    parser.add_argument('--clus_syn_pos_seed', type=int, nargs='+', default=None,
                        help='Random seed for cluster assignment, preunit activation order, and invitro clustered weights. '
                             'Accepts one or more values; multiple values expand into separate runs. '
                             'If None, uses syn_pos_seed when provided, otherwise epoch value (default: None)')
    parser.add_argument('--syn_pos_seed', type=int, default=None,
                        help='Deprecated shared synapse-position seed. Used as fallback for bg_syn_pos_seed '
                             'and clus_syn_pos_seed when those are not set (default: None)')
    parser.add_argument('--bg_spike_gen_seed', type=int, nargs='+', default=None,
                        help='Random seed for background (bg) spike generation (bg spike trains, pink noise). '
                             'Accepts one or more values; multiple values expand into separate runs. '
                             'If None, uses epoch value (default: None)')
    parser.add_argument('--clus_spike_gen_seed', type=int, nargs='+', default=None,
                        help='Random seed for cluster stimulus spike generation (stim times in generate_vecstim, '
                             'preunit permutation). Accepts one or more values; multiple values expand into '
                             'separate runs. If None, uses epoch value. Distinct from bg_spike_gen_seed (bg only) '
                             '(default: None)')
    # Biophysics model selection
    # Default is False (use L5PCbiophys3.hoc without AP and Ca)
    # Use --with_ap to enable L5PCbiophys3withNaCa.hoc (with AP and Ca dynamics)
    parser.add_argument('--with_ap', action='store_true', default=False,
                        help='Use L5PCbiophys3withNaCa.hoc (with AP and Ca dynamics). '
                             'Default is False (uses L5PCbiophys3.hoc without AP and Ca)')
    # Global segment recording: seg_ina and seg_inmda for all segments, saved as npy
    parser.add_argument('--with_global_rec', action='store_true', default=False,
                        help='Record seg_ina and seg_inmda for all segments and save to npy. Default is False')
    
    return parser


def build_cell(args):
    """Build and simulate cell with parameters from argparse"""
    # Lazy imports: NEURON must NOT be loaded in the main process before fork
    from analysis.nmda_spike_detection import save_segment_nmda_spike_rate_npz
    from utils.l5pn_model import L5PNModel

    epoch = args.epoch
    print('\n' + '='*80)
    print(f'EPOCH {epoch}')
    print('='*80 + '\n')

    seeds = resolve_workflow_seeds(
        epoch=epoch,
        bg_syn_pos_seed=args.bg_syn_pos_seed,
        clus_syn_pos_seed=args.clus_syn_pos_seed,
        bg_spike_gen_seed=args.bg_spike_gen_seed,
        clus_spike_gen_seed=args.clus_spike_gen_seed,
        legacy_syn_pos_seed=args.syn_pos_seed,
    )
    bg_syn_pos_seed = seeds.bg_syn_pos
    clus_syn_pos_seed = seeds.clus_syn_pos
    bg_spike_gen_seed = seeds.bg_spike_gen
    clus_spike_gen_seed = seeds.clus_spike_gen

    # Build channel_suffix: ensure leading underscore, then append conditional suffixes
    channel_suffix = args.channel_suffix.strip()
    channel_suffix = ('_' + channel_suffix) if channel_suffix and not channel_suffix.startswith('_') else channel_suffix
    seed_suffix_tag = getattr(args, 'seed_suffix_tag', '')
    if seed_suffix_tag:
        seed_suffix_tag = ('_' + seed_suffix_tag) if not seed_suffix_tag.startswith('_') else seed_suffix_tag
        channel_suffix += seed_suffix_tag
    channel_suffix += ''.join(['_ap' if args.with_ap else '', '_globrec' if args.with_global_rec else ''])
    simu_folder = f'{args.sec_type}_range{args.dis_to_root}_{args.spat_cond}_{args.simu_cond}{channel_suffix}'
    if args.use_fixedW:
        w_tag = format(args.fixedW, '.10g').replace('-', 'neg')
        simu_folder = f'{simu_folder}_fixedW{w_tag}'

    # Normalize folder tag
    folder_tag = str(int(args.folder_tag) % 100) if int(args.folder_tag) % 100 != 0 else '100'
    expected_suffix = '_expected' if args.expected else ''
    results_root = args.results_root.rstrip('/')
    folder_path = f'{results_root}/{simu_folder}{expected_suffix}/{folder_tag}/{epoch}'

    simulation_params = {
        'cell model': 'L5PN',
        'NUM_SYN_BASAL_EXC': args.num_syn_basal_exc, 'NUM_SYN_APIC_EXC': args.num_syn_apic_exc,
        'NUM_SYN_BASAL_INH': args.num_syn_basal_inh, 'NUM_SYN_APIC_INH': args.num_syn_apic_inh,
        'NUM_SYN_BASAL_PROX_INH': args.num_syn_basal_prox_inh,
        'BASAL_DISTAL_MIN_UM': args.basal_distal_min_um,
        'NUM_SYN_SOMA_INH': args.num_syn_soma_inh, 'SIMU DURATION': args.simu_duration,
        'STIM DURATION': args.stim_duration, 'simulation condition': args.simu_cond,
        'synaptic spatial condition': args.spat_cond, 'basal channel type': args.basal_channel_type,
        'channel_suffix': args.channel_suffix, 'seed_suffix_tag': seed_suffix_tag.lstrip('_'),
        'effective_channel_suffix': channel_suffix.lstrip('_'), 'results_root': results_root,
        'section type': args.sec_type,
        'distance from clusters to root': args.dis_to_root, 'number of clusters': args.num_clusters,
        'cluster radius': args.cluster_radius, 'background excitatory frequency': args.bg_exc_freq,
        'background inhibitory frequency': args.bg_inh_freq, 'input ratio of basal to apical': args.input_ratio_basal_apic,
        'background excitatory channel type': args.bg_exc_channel_type, 'initial weight of AMPANMDA synapses': args.initW,
        'use_fixedW': args.use_fixedW, 'fixedW': args.fixedW,
        'number of functional groups': args.num_func_group, 'delay of inhibitory inputs': args.inh_delay,
        'number of stimuli': args.num_stim, 'time point of stimulation': args.stim_time,
        'number of connection per preunit': args.num_conn_per_preunit, 'number of synapses per cluster': args.num_syn_per_clus,
        'number of trials': args.num_trials, 'syn_pos_seed': bg_syn_pos_seed,
        'bg_syn_pos_seed': bg_syn_pos_seed, 'clus_syn_pos_seed': clus_syn_pos_seed,
        'bg_spike_gen_seed': bg_spike_gen_seed, 'clus_spike_gen_seed': clus_spike_gen_seed,
        'expected': args.expected, 'aff_mode': args.aff_mode, 'aff_list': args.aff_list, 'iter_step': args.iter_step,
        'effective_iter_step': 1 if args.expected else args.iter_step,
        'num_epochs': args.num_epochs,
        'start_epoch': args.start_epoch,
        'max_workers_epoch': args.max_workers_epoch,
        'max_workers_synapse': args.max_workers_synapse,
        'with_ap': args.with_ap, 'with_global_rec': args.with_global_rec,
        'segment_nmda_spike_rate_npz': 'segment_nmda_spike_rate.npz' if args.with_global_rec else None,
    }

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    json_filename = os.path.join(folder_path, "simulation_params.json")
    with open(json_filename, 'w') as json_file:
        json.dump(simulation_params, json_file, indent=4)

    cell1 = L5PNModel(swc_file_path, args.bg_exc_freq, args.bg_inh_freq, args.simu_duration, args.stim_duration,
                     bg_syn_pos_seed, bg_spike_gen_seed, clus_spike_gen_seed, args.with_ap, args.with_global_rec,
                     clus_syn_pos_seed=clus_syn_pos_seed, max_workers_synapse=args.max_workers_synapse)
    cell1.initialize_synapse_layout(args.num_syn_basal_exc, args.num_syn_apic_exc, args.num_syn_basal_inh,
                                    args.num_syn_apic_inh, args.num_syn_soma_inh,
                                    basal_distal_min_um=args.basal_distal_min_um,
                                    num_syn_basal_prox_inh=args.num_syn_basal_prox_inh)
    
    cell1.assign_synapse_clusters(args.basal_channel_type, args.sec_type, args.dis_to_root,
                                  args.num_clusters, args.cluster_radius, args.num_stim, args.stim_time,
                                  args.spat_cond, args.num_conn_per_preunit, args.num_syn_per_clus, folder_path)

    cell1.run_stimulation_protocol(folder_path, args.simu_cond, args.input_ratio_basal_apic,
                                   args.bg_exc_channel_type, args.initW, args.num_func_group, args.inh_delay, args.num_trials,
                                   use_fixedW=args.use_fixedW, fixedW=args.fixedW,
                                   expected=args.expected, aff_mode=args.aff_mode, aff_list=args.aff_list, iter_step=args.iter_step)

    if args.with_global_rec and cell1.seg_v_array is not None:
        save_segment_nmda_spike_rate_npz(cell1, folder_path)


def _run_tasks(combinations, max_workers):
    """Run build_cell over combinations, isolating per-epoch failures.

    Unlike executor.map (fail-fast), one failing epoch is logged and skipped
    instead of aborting the whole job; missing epochs are recovered later by
    the rerun_missing scripts. Returns the list of failed combinations.
    """
    def label(c):
        return (f'epoch={c.epoch} simu_cond={c.simu_cond} spat_cond={c.spat_cond} '
                f'sec_type={c.sec_type} dis_to_root={c.dis_to_root}')

    def record(combo, exc):
        print(f'[FAILED] {label(combo)}: {exc!r}', flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        return combo

    failures = []
    if max_workers == 1:
        for combo in combinations:
            try:
                build_cell(combo)
            except Exception as exc:  # noqa: BLE001 - isolate per-epoch failures
                failures.append(record(combo, exc))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(build_cell, c): c for c in combinations}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - isolate per-epoch failures
                    failures.append(record(futures[future], exc))

    ok = len(combinations) - len(failures)
    print(f'{ok}/{len(combinations)} tasks succeeded, {len(failures)} failed.', flush=True)
    return failures


def run_combination(args):
    """Expand CLI arguments and run requested parameter combinations."""
    if args.num_epochs <= 0:
        raise ValueError('num_epochs must be positive')
    if args.max_workers_epoch <= 0:
        raise ValueError('max_workers_epoch must be positive')
    if args.max_workers_synapse <= 0:
        raise ValueError('max_workers_synapse must be positive')

    epoch_range = range(args.start_epoch, args.start_epoch + args.num_epochs)
    seed_values = {
        field: (list(getattr(args, field)) if getattr(args, field) is not None else [None])
        for field, _ in SEED_FIELDS
    }
    multi_seed_fields = {
        field
        for field, values in seed_values.items()
        if len(values) > 1
    }
    seed_products = list(itertools.product(*(seed_values[field] for field, _ in SEED_FIELDS)))

    for spat_cond in args.spat_cond:
        combinations = []
        for seed_combo in seed_products:
            seed_kwargs = {
                field: seed_combo[idx]
                for idx, (field, _) in enumerate(SEED_FIELDS)
            }
            seed_tag_parts = [
                f'{abbr}{seed_kwargs[field]}'
                for field, abbr in SEED_FIELDS
                if field in multi_seed_fields and seed_kwargs[field] is not None
            ]
            seed_suffix_tag = '_'.join(seed_tag_parts)

            for epoch in epoch_range:
                for simu_cond, sec_type, dis_to_root in itertools.product(
                    args.simu_cond, args.sec_type, args.dis_to_root
                ):
                    run_args = argparse.Namespace(**vars(args))
                    run_args.epoch = epoch
                    run_args.simu_cond = simu_cond
                    run_args.spat_cond = spat_cond
                    run_args.sec_type = sec_type
                    run_args.dis_to_root = dis_to_root
                    for field, value in seed_kwargs.items():
                        setattr(run_args, field, value)
                    run_args.seed_suffix_tag = seed_suffix_tag
                    combinations.append(run_args)

        seed_summary = ', '.join(
            f'{field}={seed_values[field]}'
            for field, _ in SEED_FIELDS
            if seed_values[field] != [None]
        )
        seed_summary = seed_summary or 'epoch-default seeds'

        print(
            f'Running epochs={epoch_range.start}..{epoch_range.stop - 1} '
            f'spat_cond={spat_cond} seeds=({seed_summary}) '
            f'({len(combinations)} tasks, max_workers_epoch={args.max_workers_epoch})'
        )
        _run_tasks(combinations, args.max_workers_epoch)


if __name__ == "__main__":
    parser = create_parser()
    run_combination(parser.parse_args())
