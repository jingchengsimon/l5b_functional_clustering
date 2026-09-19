"""Small regression checks for the cell3-only apical policy."""

import unittest

import networkx as nx
import pandas as pd

from utils.cell3_apical_policy import Cell3ApicalTreePolicy


class Cell3ApicalTreePolicyTest(unittest.TestCase):
    roots = ['apic[40]', 'apic[72]', 'apic[77]', 'apic[102]']

    def test_epoch_rotation(self):
        selected = [Cell3ApicalTreePolicy(epoch, self.roots).selected_root_name for epoch in range(1, 9)]
        self.assertEqual(selected, self.roots * 2)

    def test_selected_tree_and_three_way_split(self):
        graph = nx.DiGraph([(40, 41), (41, 42), (72, 73), (77, 78), (102, 103)])
        policy = Cell3ApicalTreePolicy(1, self.roots)
        policy.bind({name: root for name, root in zip(self.roots, (40, 72, 77, 102))}, graph)
        frame = pd.DataFrame({
            'type': ['A'] * 8 + ['B'] * 2,
            'section_id_synapse': [41, 41, 42, 42, 41, 42, 41, 73, 41, 42],
            'distance_to_tuft': [1, 2, 3, 4, 5, 6, 7, 1, 2.5, 6.5],
        })
        chunks = policy.range_indices(frame)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 3])
        metadata = policy.metadata(frame)
        self.assertEqual(metadata['eligible_exc_count'], 7)
        self.assertEqual(metadata['exc_counts_by_range'], [2, 2, 3])
        self.assertEqual(metadata['inh_counts_by_range'], [0, 1, 1])


if __name__ == '__main__':
    unittest.main()
