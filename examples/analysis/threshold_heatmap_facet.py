#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent

Generate heatmaps of activation thresholds, using Seaborn's FacetGrid.
RUN THIS FROM REPOSITORY ROOT
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src.core.plotter import heatmaps
from src.core.query import Query

q = Query(
    {
        'partial_matches': True,
        'include_downstream': True,
        'indices': {'sample': [0], 'model': [0, 1], 'sim': [1]},
    }
).run()

# Get threshold data
fdata = q.threshold_data()

# Specify keyword arguments to pass to heatmaps()
heatmap_kws = {'min_max_ticks': False}

# Build heatmap grid
g = sns.FacetGrid(fdata, row='model', col='nsim', sharex=False, sharey=False)
g.map(heatmaps, *fdata.columns, **heatmap_kws)

# Title and clear axis labels
plt.suptitle('Grid of activation threshold heatmaps')
for ax in g.axes.ravel():
    ax.set_xlabel('')
    ax.set_ylabel('')
