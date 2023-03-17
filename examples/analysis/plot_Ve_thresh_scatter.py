# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:54:51 2023

@author: Danny
"""

import sys
sys.path.append("D:/Program Files/ascent-master/ascent")
import os

import matplotlib.pyplot as plt
import numpy as np

from src.core.query import Query
from src.utils import Object

sample = 0
model = 0
sim = 0
n_sim = 0
inner = 0
fibers = range(0,12)

base_n_sim = os.path.join('samples', str(sample), 'models', str(model), 'sims', str(sim), 'n_sims')
plt.figure()
for fiber in fibers:
    # load potentials, which are inputs to the n_sim
    pve1 = os.path.join(base_n_sim, str(n_sim), 'data', 'inputs', f'inner{inner}_fiber{fiber}.dat') #  '0_0_0_0',
    thresh1 = os.path.join(base_n_sim, str(n_sim), 'data', 'outputs', f'thresh_inner{inner}_fiber{fiber}.dat')
    dpve1 = np.loadtxt(pve1)
    pp_ve = max(dpve1) - min(dpve1)
    dthresh1 = np.loadtxt(thresh1)

    # load the corresponding fiber coordinates
    sim_object = Query.get_object(Object.SIMULATION, [sample, model, sim])
    for t, (p_i, _) in enumerate(sim_object.master_product_indices):
        if t == n_sim:
            potentials_ind = p_i
            break

    active_src_ind, fiberset_ind = sim_object.potentials_product[potentials_ind]
    master_fiber_ind = sim_object.indices_n_to_fib(fiberset_index=fiberset_ind, inner_index=inner, local_fiber_index=fiber)

    fiber_coords_path = os.path.join(
        'samples',
        str(sample),
        'models',
        str(model),
        'sims',
        str(sim),
        'fibersets',
        str(fiberset_ind),
        f'{master_fiber_ind}.dat',
    )
    z_coords = np.loadtxt(fiber_coords_path, skiprows=1)[:, 2]
    #print(dpve1)
    print(dthresh1)
    plt.scatter(pp_ve, dthresh1, label=f'Fiber:{fiber}')
plt.ylabel('Threshold Amplitude (mA)')
plt.xlabel('Ve (V)')
plt.title('Threshold as a Function of Ve and Fiber')
plt.legend()
plt.show()

print('done')
