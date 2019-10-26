from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def weights_hist(
		weights: np.ndarray, n_bins: int = 10, range: Tuple[float, float] = (0., 1.), labels_y_shift: float = 3,
		labels_x_shift: float = -0.03):
	"""
	Plot an histogram of the weights.

	Parameters
	----------
	weights: ndarray
		The weights.
	n_bins: int
		The number of bins in the histogram.
	range: tuple
		The minimum and maximum values on the horizontal axis.
	labels_y_shift: float
		Vertical shift of the labels above the bars.
	labels_x_shift: float
		Horizontal shift for the labels above the bars.

	Returns
	-------
	out: tuple
		Figure and axes.

	"""

	# for the sake of convenience/efficiency
	n = len(weights)

	fig = plt.figure()
	axes = plt.axes()

	values, bins_edges, _ = axes.hist(weights, n_bins, range)

	# for the sake of convenience, `values` are cast as `int`s
	values = values.astype(int)

	for left_edge, right_edge, val in zip(bins_edges[:-1], bins_edges[1:], values):

		# if the value is different from 0...
		if val:
			axes.text(
				(left_edge + right_edge) / 2 + labels_x_shift, val + labels_y_shift, str(val),
				color='blue', fontweight='bold')

	# the y limits are extracted (as a list)...
	ylim = list(axes.get_ylim())

	# ...and modified to account for the added text's...
	ylim[1] += labels_y_shift

	# they are re-set afterwards
	axes.set_ylim(ylim)

	# a vertical line at the ideal weight, i.e., when all the weights are equal
	axes.axvline(x=1. / n, color='red')

	# print(f'maximum is {weights.max()}')

	# the non-zero weights...
	nonzero_weights = weights[np.nonzero(weights)[0]]

	# ...are used to compute the entropy...
	entropy = -(nonzero_weights * np.log(nonzero_weights)).sum()

	# ...and compare it with the maximum
	max_entropy = np.log2(n)

	# print(entropy, max_entropy)

	axes.set_title(f'entropy = {entropy:.2f} (max. is {max_entropy:.2f}); maximum weight = {weights.max():.4f}')

	return fig, axes