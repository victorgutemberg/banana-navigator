import numpy as np

class SumTree:
    def __init__(self, capacity):
        # the depth of the complete binary tree.
        depth = int(np.ceil(np.log2(capacity)) + 1)
        leaves_capacity = 2 ** (depth - 1)

        redundant_leaves = leaves_capacity - capacity

        self.capacity = capacity
        self.nodes = np.zeros(2 ** depth - 1 - redundant_leaves)
        self.transitions = np.zeros(self.capacity, dtype=tuple)
        self.max = 0
        self.cur_index = 0 # holds the next position to insert an item.

        self._leaves_offset = 2 ** (depth - 1) - 1 # offset to reach the leaves of the tree.
        self._len = 0

    @property
    def sum(self):
        return self.nodes[0]

    def append(self, transition):
        priority = transition.priority

        self.max = max(self.max, priority)
        self.update(self.cur_index, priority)
        self.transitions[self.cur_index] = transition

        self.cur_index = (self.cur_index + 1) % self.capacity # calculate next index.
        if len(self) < self.capacity:
            self._len += 1

    def get_left_index(self, index):
        return 2 * index + 1

    def get_right_index(self, index):
        return 2 * index + 2

    def get_parent_index(self, index):
        return (index - 1) // 2

    def update(self, index, priority):
        if self.transitions[index]:
            change = priority - self.transitions[index].priority
        else:
            change = priority

        self._update(index + self._leaves_offset, change)

    def sample(self, priority):
        index = self._sample(0, priority)
        transition = self.transitions[index]

        return (index, transition)

    def _sample(self, index, priority):
        # if reached a leaf index, stop and return transition index.
        while index < self._leaves_offset:
            left_index = self.get_left_index(index)
            right_index = self.get_right_index(index)

            if priority < self.nodes[left_index]:
                index = left_index
            else:
                index = right_index
                priority -= self.nodes[left_index]

        return index - self._leaves_offset

    def _update(self, index, change):
        self.nodes[index] += change

        if index:
            parent_index = self.get_parent_index(index)
            self._update(parent_index, change)

    def __len__(self):
        return self._len