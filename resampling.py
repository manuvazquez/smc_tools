import abc

import numpy as np


class ResamplingAlgorithm(metaclass=abc.ABCMeta):
	"""
	Abstract class for a resampling algorithm.
	"""
	
	def __init__(self) -> None:

		pass

	@abc.abstractmethod
	def get_indexes(self, weights: np.ndarray, n: int) -> np.ndarray:
		"""
		Returns the indexes of the particles that should be kept after resampling.
		
		Notice that it doesn't perform any "real" resampling": the real work must be performed somewhere else.

		Parameters
		----------
		weights : ndarray
			The weights of the particles.
		n : int, optional
			The number of indexes requested

		Returns
		-------
		out : ndarray
			The indices of the selected particles
		"""
		pass


class MultinomialResamplingAlgorithm(ResamplingAlgorithm):
	"""
	Class for multinomial resampling.
	"""
	
	def __init__(self, prng=np.random.RandomState()) -> None:
		"""
		Creates a "multinomial" resampling object.

		Parameters
		----------
		prng : RandomState instance or None, optional (default=None)
			If RandomState instance, `prng` is the random number generator;
			If None, a new random number generator instance is built
		"""
		
		self._prng = prng
		
	def get_indexes(self, weights: np.ndarray, n: int = None) -> np.ndarray:
		
		if not n:

			n = weights.size
		
		return self._prng.choice(range(weights.size), n, p=weights)


class ResamplingCriterion(metaclass=abc.ABCMeta):
	"""
	Abstract class for a resampling criterion.
	"""

	@abc.abstractmethod
	def is_resampling_needed(self, weights: np.ndarray) -> bool:
		"""
		Determines whether resampling is needed.

		Parameters
		----------
		weights : ndarray
			The particles' weights

		Returns
		-------
		out: bool
			Returns True if resampling must be carried out according to the criterion.
		"""
		
		pass


class EffectiveSampleSizeBasedResamplingCriterion(ResamplingCriterion):
	"""
	Class for resampling based on sample size.
	"""
	
	def __init__(self, resampling_ratio: float) -> None:
		"""
		Creates an effective sample size-based resampling criterion object.

		Parameters
		----------
		resampling_ratio : float
			The requested effective sample size.
		"""
		
		self._resampling_ratio = resampling_ratio
		
	def is_resampling_needed(self, weights: np.ndarray) -> bool:
		
		# a division by zero may occur...
		try:
			
			n_effective_particles = 1/np.dot(weights, weights)
			
		except ZeroDivisionError:
			
			raise Exception('all the weights are zero!!')
			
		return n_effective_particles < (self._resampling_ratio * weights.size)


class AlwaysResamplingCriterion(ResamplingCriterion):
	"""
	Class implementing an always-resampling policy.
	"""
	
	def is_resampling_needed(self, weights: np.ndarray) -> bool:
		
		return True
