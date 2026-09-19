"""Cell3-only apical-tree selection and range partitioning.

This module deliberately contains no NEURON imports so the policy can be
validated independently.  Cell1 and cell2 never instantiate this policy.
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np


class Cell3ApicalTreePolicy:
    """Select one cell3 apical tree per epoch and split its excitatory pool."""

    def __init__(self, epoch, root_names):
        if int(epoch) < 1:
            raise ValueError('epoch must be >= 1')
        if len(root_names) != 4:
            raise ValueError(f'cell3 requires four apical roots, got {root_names!r}')
        self.epoch = int(epoch)
        self.root_names = tuple(root_names)
        self.selected_root_name = self.root_names[(self.epoch - 1) % len(self.root_names)]
        match = re.fullmatch(r'apic\[(\d+)\]', self.selected_root_name)
        if match is None:
            raise ValueError(f'Invalid cell3 apical root name: {self.selected_root_name!r}')
        self.selected_apic_index = int(match.group(1))
        self.selected_root_id = None
        self.selected_section_ids = frozenset()

    def bind(self, name_to_id, graph):
        missing = [name for name in self.root_names if name not in name_to_id]
        if missing:
            raise ValueError(f'Cell3 apical roots not found in morphology graph: {missing}')
        self.selected_root_id = int(name_to_id[self.selected_root_name])
        self.selected_section_ids = frozenset(
            int(section_id) for section_id in nx.descendants(graph, self.selected_root_id)
        )
        if not self.selected_section_ids:
            raise ValueError(f'Cell3 root {self.selected_root_name} has no descendants')

    @property
    def recording_site(self):
        return {'section_type': 'apic', 'index': self.selected_apic_index, 'x': 1.0}

    def _eligible_exc(self, section_synapse_df):
        if self.selected_root_id is None:
            raise RuntimeError('Cell3 policy must be bound before use')
        return section_synapse_df[
            (section_synapse_df['type'] == 'A')
            & section_synapse_df['section_id_synapse'].isin(self.selected_section_ids)
            & (section_synapse_df['distance_to_tuft'] >= 0)
        ].sort_values('distance_to_tuft', kind='stable')

    def range_indices(self, section_synapse_df):
        eligible = self._eligible_exc(section_synapse_df)
        if len(eligible) < 3:
            raise ValueError(
                f'Cell3 root {self.selected_root_name} has only {len(eligible)} eligible excitatory synapses'
            )
        return tuple(np.asarray(chunk.index, dtype=int) for chunk in np.array_split(eligible, 3))

    def metadata(self, section_synapse_df=None):
        payload = {
            'policy': 'cell3_epoch_balanced_single_apical_tree_v1',
            'epoch': self.epoch,
            'selected_root': self.selected_root_name,
            'selected_root_section_id': self.selected_root_id,
            'recording_site': self.recording_site,
        }
        if section_synapse_df is None or self.selected_root_id is None:
            return payload

        eligible = self._eligible_exc(section_synapse_df)
        chunks = self.range_indices(section_synapse_df)
        distances = eligible['distance_to_tuft'].to_numpy(float)
        boundaries = [
            float(distances[len(chunks[0]) - 1]),
            float(distances[len(chunks[0]) + len(chunks[1]) - 1]),
        ]
        selected_inh = section_synapse_df[
            (section_synapse_df['type'] == 'B')
            & section_synapse_df['section_id_synapse'].isin(self.selected_section_ids)
            & (section_synapse_df['distance_to_tuft'] >= 0)
        ]['distance_to_tuft'].to_numpy(float)
        inh_counts = [
            int(np.count_nonzero(selected_inh <= boundaries[0])),
            int(np.count_nonzero((selected_inh > boundaries[0]) & (selected_inh <= boundaries[1]))),
            int(np.count_nonzero(selected_inh > boundaries[1])),
        ]
        payload.update({
            'eligible_exc_count': int(len(eligible)),
            'eligible_inh_count': int(len(selected_inh)),
            'range_boundaries_um': boundaries,
            'exc_counts_by_range': [int(len(chunk)) for chunk in chunks],
            'inh_counts_by_range': inh_counts,
        })
        return payload
