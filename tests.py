from collections import namedtuple
import numpy.testing as np
import unittest

from replay_buffer import to_experience
from sum_tree import SumTree

to_experience = namedtuple('Experience', field_names=['priority'])

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

        tree.append(to_experience(5))
        np.assert_array_equal(tree.nodes, [5, 5, 0, 5, 0, 0])
        self.assertEquals(len(tree), 1)

        tree.append(to_experience(3))
        np.assert_array_equal(tree.nodes, [8, 8, 0, 5, 3, 0])
        self.assertEquals(len(tree), 2)

        tree.append(to_experience(2))
        np.assert_array_equal(tree.nodes, [10, 8, 2, 5, 3, 2])
        self.assertEquals(len(tree), 3)

        tree.append(to_experience(7))
        np.assert_array_equal(tree.nodes, [12, 10, 2, 7, 3, 2])
        self.assertEquals(len(tree), 3)

        tree.append(to_experience(6))
        np.assert_array_equal(tree.nodes, [15, 13, 2, 7, 6, 2])
        self.assertEquals(len(tree), 3)

    def test_sample(self):
        tree = SumTree(4)
        tree.append(to_experience(1))
        tree.append(to_experience(2))
        tree.append(to_experience(3))
        tree.append(to_experience(4))

        self.assertTupleEqual(tree.sample(1)[1], to_experience(2))
        self.assertTupleEqual(tree.sample(2)[1], to_experience(2))
        self.assertTupleEqual(tree.sample(3)[1], to_experience(3))
        self.assertTupleEqual(tree.sample(4)[1], to_experience(3))
