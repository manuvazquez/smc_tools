import warnings

import numpy as np

from . import util


def normalize_from_logs(logs: np.ndarray) -> np.ndarray:
	"""
	Normalizes a sequence of numbers (so that they add up to 1) given their individual logarithms.

	Parameters
	----------
	logs : array_like
		The (individual) logarithms of a sequence of numbers.

	Returns
	-------
	out: ndarray
		The normalized sequence.
	"""

	log_sum = util.log_sum_from_individual_logs(logs)

	return np.exp(logs - log_sum)


def normalize_or_flatten(weights: np.ndarray) -> np.ndarray:
	"""
	Normalizes the weights so that they add up to 1. If all of them are zero (for a computer, anyway), then a uniform
	distribution is returned.

	Parameters
	----------
	weights: ndarray
		The weights to be normalized.

	Returns
	-------
	out: ndarray
		The normalized weights.

	"""

	sum = weights.sum()

	# if the sum of all weights is zero (normalization cannot be carried out)...
	if np.isclose(sum, 0.0):

		# the name of weights
		n = len(weights)

		warnings.warn('All the weights add up to 0...just flattening them', UserWarning)

		# the weights are reset
		return np.full(n, 1. / n)

	# otherwise (the weights can be normalized)...
	else:

		# the weights divided by their sum so that, afterwards, they add up to 1
		return weights / sum