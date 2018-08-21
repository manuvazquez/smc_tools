import abc

import numpy as np


class ResamplingAlgorithm(metaclass=abc.ABCMeta):
	
	def __init__(self):
		"""
			Abstract class for a resampling algorithm
		"""
		
		pass

	@abc.abstractmethod
	def get_indexes(self, weights, n):
		"""
		Returns the indexes of the particles that should be kept after resampling.
		
		Notice that it doesn't perform any "real" resampling": the real work must be performed somewhere else.

		Parameters
		----------
		weights : array_like
			The weights of the particles.
		n : int, optional
			The number of indexes requested

		Returns
		-------
		out : array_like
			The indices of the selected particles
		"""
		pass


class MultinomialResamplingAlgorithm(ResamplingAlgorithm):
	
	def __init__(self, PRNG=np.random.RandomState()):
		"""
		Creates a "multinomial" resampling object.

		Parameters
		----------
		PRNG : RandomState instance or None, optional (default=None)
			If RandomState instance, PRNG is the random number generator;
			If None, a new random number generator instance is built
		"""
		
		self._PRNG = PRNG
		
	def get_indexes(self, weights, n=None):
		
		if not n:

			n = weights.size
		
		return self._PRNG.choice(range(weights.size), n, p=weights)


class ResamplingCriterion(metaclass=abc.ABCMeta):

	@abc.abstractmethod
	def is_resampling_needed(self, weights):
		"""
		Determines whether resampling is needed.

		Parameters
		----------
		weights : array_like
			The particles' weights

		Returns
		-------
		out: bool
			Returns True if resampling must be carried out according to the criterion.
		"""
		
		pass


class EffectiveSampleSizeBasedResamplingCriterion(ResamplingCriterion):
	
	def __init__(self, resampling_ratio):
		"""
		Creates an effective sample size-based resampling criterion object.

		Parameters
		----------
		resampling_ratio : float
			The requested effective sample size.
		"""
		
		self._resampling_ratio = resampling_ratio
		
	def is_resampling_needed(self, weights):
		
		# a division by zero may occur...
		try:
			
			n_effective_particles = 1/np.dot(weights, weights)
			
		except ZeroDivisionError:
			
			raise Exception('all the weights are zero!!')
			
		return n_effective_particles < (self._resampling_ratio * weights.size)


class AlwaysResamplingCriterion(ResamplingCriterion):
	
	def is_resampling_needed(self, weights):
		
		return True
