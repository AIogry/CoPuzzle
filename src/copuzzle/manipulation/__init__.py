"""Versioned direct-contact task suites, native collections and continuous data.

Imports are lightweight; simulation is required only when collecting/replaying.
The existing abstract and physical trajectory contracts remain independent.
"""
from .suite import generate_suite, load_suite, save_suite, validate_suite, suite_sha256
from .data import ContinuousDataset, load_transitions

__all__ = ['generate_suite', 'load_suite', 'save_suite', 'validate_suite',
           'suite_sha256', 'ContinuousDataset', 'load_transitions']
