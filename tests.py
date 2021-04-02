from collections import namedtuple
import numpy.testing as np
import unittest

from replay_buffer import to_experience
from sum_tree import SumTree

experience = ()


class SumTreeTest(unittest.TestCase):
    def test_length(self):
        self.assertEqual(len(SumTree(2).nodes), 3)
        self.assertEqual(len(SumTree(3).nodes), 6)
        self.assertEqual(len(SumTree(4).nodes), 7)
        self.assertEqual(len(SumTree(7).nodes), 14)
        self.assertEqual(len(SumTree(8).nodes), 15)
        self.assertEqual(len(SumTree(9).nodes), 24)

    def test_append(self):
        tree = SumTree(3)

        self.assertEquals(len(tree), 0)

        tree.append(experience)
        tree.update(0, 5)
        np.assert_array_equal(tree.nodes, [5, 5, 0, 5, 0, 0])
        self.assertEquals(len(tree), 1)

        tree.append(experience)
        tree.update(1, 3)
        np.assert_array_equal(tree.nodes, [8, 8, 0, 5, 3, 0])
        self.assertEquals(len(tree), 2)

        tree.append(experience)
        tree.update(2, 2)
        np.assert_array_equal(tree.nodes, [10, 8, 2, 5, 3, 2])
        self.assertEquals(len(tree), 3)

        tree.append(experience)
        tree.update(0, 7)
        np.assert_array_equal(tree.nodes, [12, 10, 2, 7, 3, 2])
        self.assertEquals(len(tree), 3)

        tree.append(experience)
        tree.update(1, 6)
        np.assert_array_equal(tree.nodes, [15, 13, 2, 7, 6, 2])
        self.assertEquals(len(tree), 3)

    def test_sample(self):
        tree = SumTree(4)
        tree.append(experience)
        tree.update(0, 1)
        tree.append(experience)
        tree.update(1, 2)
        tree.append(experience)
        tree.update(2, 3)
        tree.append(experience)
        tree.update(3, 4)

        self.assertEqual(tree.sample(1)[1], 2)
        self.assertEqual(tree.sample(2)[1], 2)
        self.assertEqual(tree.sample(3)[1], 3)
        self.assertEqual(tree.sample(4)[1], 3)

SumTreeTest().test_sample()