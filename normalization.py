import warnings

import numpy as np
import scipy.special

# exponent of number "e" that yields "eps" (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html)
negep_for_e = np.finfo(np.float64).negep * np.log(2)


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

	return np.exp(logs - scipy.special.logsumexp(logs))


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


def normalize_or_flatten_logs(log_weights: np.ndarray) -> np.ndarray:
	"""
	Normalizes the *logarithms* of the weights so that the latter up to 1. If all of them are zero (within machine
	limits), then (the log of) a uniform distribution is returned.

	Parameters
	----------
	log_weights: ndarray
		The logarithms of the weights to be normalized.

	Returns
	-------
	out: ndarray
		The logarithms of the normalized weights.

	"""

	# the logarithm of the sum
	log_sum = scipy.special.logsumexp(log_weights)

	# if the sum of all weights is zero (normalization cannot be carried out)...
	if log_sum <= negep_for_e:

		# the name of weights
		n = len(log_weights)

		warnings.warn('All the weights add up to 0...just flattening them', UserWarning)

		# the weights are reset
		return np.full(n, -np.log(n))

	# otherwise (the weights can be normalized)...
	else:

		# the sum of the weights' logs is subtracted from the latter
		return log_weights - log_sum
