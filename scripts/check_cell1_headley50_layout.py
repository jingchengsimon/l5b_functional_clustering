"""Build one layout and assert the Headley-50 synapse contract."""

import json

from utils.l5pn_model import L5PNModel


cell = L5PNModel("./model/cell1.asc", 1.0, 4.0, 1000, 1000, 1, 1, 1, False, False,
                 clus_syn_pos_seed=1, max_workers_synapse=4)
cell.initialize_synapse_layout(10042, 16070, 1023, 1637, 150,
                               basal_distal_min_um=50.0, num_syn_basal_prox_inh=107)
df = cell.section_synapse_df
basal_exc = df[(df.region == "basal") & (df.type == "A")]
basal_inh = df[(df.region == "basal") & (df.type == "B")]
apical_exc = df[(df.region == "apical") & (df.type == "A")]
apical_inh = df[(df.region == "apical") & (df.type == "B")]
soma_inh = df[(df.region == "soma") & (df.type == "B")]
summary = {
    "basal_exc_distal": int((basal_exc.distance_to_soma >= 50).sum()),
    "basal_exc_proximal": int((basal_exc.distance_to_soma < 50).sum()),
    "basal_inh_distal": int((basal_inh.distance_to_soma >= 50).sum()),
    "basal_inh_proximal": int((basal_inh.distance_to_soma < 50).sum()),
    "apical_exc": len(apical_exc),
    "apical_inh": len(apical_inh),
    "soma_inh": len(soma_inh),
    "perisomatic_inh": int((basal_inh.distance_to_soma < 50).sum()) + len(soma_inh),
}
assert summary == {
    "basal_exc_distal": 10042,
    "basal_exc_proximal": 0,
    "basal_inh_distal": 1023,
    "basal_inh_proximal": 107,
    "apical_exc": 16070,
    "apical_inh": 1637,
    "soma_inh": 150,
    "perisomatic_inh": 257,
}, summary
print(json.dumps(summary, indent=2))
