import numpy as np


def log_sum_from_individual_logs(logs):
	"""
	Returns the logarithm of the sum of a sequence of numbers given their individual logarithms.

	Parameters
	----------
	logs : array_like
		The (individual) logarithms of a sequence of numbers.

	Returns
	-------
	out: float
		The logarithm of the sum.
	"""

	descending_sort = np.sort(logs)[::-1]

	return descending_sort[0] + np.log1p(np.exp(descending_sort[1:] - descending_sort[0]).sum())


def normalize_from_logs(logs):
	"""
	Returns the sum of a sequence of numbers given their individual logarithms.

	Parameters
	----------
	logs : array_like
		The (individual) logarithms of a sequence of numbers.

	Returns
	-------
	out: float
		The (natural) sum.
	"""

	log_sum = log_sum_from_individual_logs(logs)

	return np.exp(logs - log_sum)
