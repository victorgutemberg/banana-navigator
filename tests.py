from collections import namedtuple
import numpy.testing as np
import unittest

from array_local import Array
from replay_buffer import RankBasedReplayBuffer
from sum_tree import SumTree

experience = ()


class ArrayTest(unittest.TestCase):
    def test_append(self):
        array = Array(4)

        transitions = [1, 2, 3, 4]
        array.append(transitions[0])
        array.append(transitions[1])
        array.append(transitions[2])
        array.append(transitions[3])

        expected_transition = transitions[::-1]
        for i in range(len(transitions)):
            # check that all transitions have the max priority assigned
            # and if the transitions are added in reverse order.
            self.assertEntry(array[i], array.max_priority, expected_transition[i])

    def test_batch_update(self):
        array = Array(4)
        array.append(1)
        array.append(2)
        array.append(3)
        array.append(4)

        array.batch_update(range(4), [6, 5, 7, 4])

        self.assertEntry(array[0], 7, 2)
        self.assertEntry(array[1], 6, 4)
        self.assertEntry(array[2], 5, 3)
        self.assertEntry(array[3], 4, 1)

    def assertEntry(self, entry, priority, transition):
        self.assertEqual(entry[0], priority)
        self.assertEqual(entry[1], transition)


class SumTreeTest(unittest.TestCase):
    def test_length(self):
        self.assertEqual(len(SumTree(2).nodes), 3)
        self.assertEqual(len(SumTree(3).nodes), 6)
        self.assertEqual(len(SumTree(4).nodes), 7)
        self.assertEqual(len(SumTree(7).nodes), 14)
        self.assertEqual(len(SumTree(8).nodes), 15)
        self.assertEqual(len(SumTree(9).nodes), 24)

    def test_append(self):
        tree = SumTree(3, e=0)

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
        tree = SumTree(4, e=0)
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

    def test_sample_right_index_limit(self):
        '''Test requesting a very high priority to check if it doesn't run off right limit.'''
        tree = SumTree(3, e=0)
        tree.append(experience)
        tree.update(0, 1)

        tree.append(experience)
        tree.update(1, 2)

        self.assertEqual(tree.sample(5)[1], 2)


class RankBasedReplayBufferTest(unittest.TestCase):
    def test_get_sample_ranges(self):
        batch_size = 3
        buffer_size = 10
        alpha = 1
        buffer = RankBasedReplayBuffer(None, batch_size, buffer_size, 'array', alpha, None, None, None)
        buffer.memory = range(10) # small hack to initialize the memory with some values.

        # start by testing the ranges with batch size 3.
        segments = buffer._get_sample_ranges()
        self.assertEqual(len(segments), batch_size)
        self.assertTupleEqual(segments[0], (0, 0))
        self.assertTupleEqual(segments[1], (1, 3))
        self.assertTupleEqual(segments[2], (4, 9))

        # increase batch size to 5 and repeat test.
        buffer.batch_size = batch_size = 5
        segments = buffer._get_sample_ranges()
        self.assertEqual(len(segments), batch_size)
        self.assertTupleEqual(segments[0], (0, 0))
        self.assertTupleEqual(segments[1], (1, 1))
        self.assertTupleEqual(segments[2], (2, 2))
        self.assertTupleEqual(segments[3], (3, 5))
        self.assertTupleEqual(segments[4], (6, 9))
