"""Tests for the metrics functions."""

import unittest

from maldi_learn.metrics import specificity_score
from maldi_learn.metrics import sensitivity_score
from maldi_learn.metrics import specificity_sensitivity_curve

from sklearn.metrics import precision_recall_curve

import numpy as np


MOCK_PREDICTIONS = [0, 1, 0, 1, 0]
MOCK_SCORES = [0.1, 0.4, 0.35, 0.8, 0.2]
MOCK_LABELS = [0, 0, 1, 1, 1]

MOCK_PREDICTIONS_FALSE = [1, 1, 0, 1, 0]
MOCK_LABELS_FALSE = [0, 0, 1, 0, 1]

MOCK_PREDICTIONS_TRUE = [1, 1, 0, 1, 0]
MOCK_LABELS_TRUE = [1, 1, 0, 1, 0]

class TestMetrics(unittest.TestCase):
    
    def test_specificity(self):
        self.assertEqual(
            specificity_score(
                MOCK_LABELS, 
                MOCK_PREDICTIONS,
            ),
            0.5,
        )

    def test_sensitivity(self):
        self.assertEqual(
            sensitivity_score(
                MOCK_LABELS, 
                MOCK_PREDICTIONS,
            ),
            1/3.,
        )

    def test_specificity_true(self):
        self.assertEqual(
            specificity_score(
                MOCK_LABELS_TRUE, 
                MOCK_PREDICTIONS_TRUE,
            ),
            1,
        )

    def test_sensitivity_true(self):
        self.assertEqual(
            sensitivity_score(
                MOCK_LABELS_TRUE, 
                MOCK_PREDICTIONS_TRUE,
            ),
            1,
        )

    def test_specificity_false(self):
        self.assertEqual(
            specificity_score(
                MOCK_LABELS_FALSE, 
                MOCK_PREDICTIONS_FALSE,
            ),
            0,
        )

    def test_sensitivity_false(self):
        self.assertEqual(
            sensitivity_score(
                MOCK_LABELS_FALSE, 
                MOCK_PREDICTIONS_FALSE,
            ),
            0,
        )



class TestCurve(unittest.TestCase):

    def test_curve_specificity(self):
        specificity, sensitivity, thresholds = specificity_sensitivity_curve(
                                                            MOCK_LABELS,
                                                            MOCK_SCORES)
        
        spec_true = np.array([0, 0.5, 0.5, 0.5, 1, 1])
        print('thresh', thresholds)
        print('spec', specificity, spec_true)

        for i, spec in enumerate(specificity):
            self.assertAlmostEqual(
                spec,
                spec_true[i]
            )

    def test_curve_sensitivity(self):
        specificity, sensitivity, thresholds = specificity_sensitivity_curve(
                                                            MOCK_LABELS,
                                                            MOCK_SCORES)
        sen_true = np.array([1, 1, 0.666666666, 0.33333333, 0.333333333, 0])
        print('sen', sensitivity, sen_true)

        for i, sen in enumerate(sensitivity):
            self.assertAlmostEqual(
                sen,
                sen_true[i],
            )

#    def test_equal_lengths(self):
#        self._clf_smoke_test(
#            MOCK_DATA,
#            MOCK_LABELS,
#            2.1993352
#        )
#
#    def test_different_lengths(self):
#        self._clf_smoke_test(
#            MOCK_DATA_DIFFERENT_LENGTHS,
#            MOCK_LABELS,
#            2.1993352
#        )
