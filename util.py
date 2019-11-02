from typing import Sequence

import scipy.special


def log_sum_from_individual_logs(logs: Sequence) -> float:
	"""
	Returns the logarithm of the sum of a sequence of numbers given their individual logarithms.

	TODO: this function is here for backwards compatibility reasons

	Parameters
	----------
	logs : array_like
		The (individual) logarithms of a sequence of numbers.

	Returns
	-------
	out: float
		The logarithm of the sum.
	"""

	return scipy.special.logsumexp(logs)
