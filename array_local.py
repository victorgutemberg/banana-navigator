from collections import deque


class Array:
    def __init__(self, capacity):
        self.capacity = capacity
        self._init_transitions()
        self.max_priority = 1

    def append(self, transition):
        # since the new transitions are assigned the max priority, they are
        # guaranteed to be placed at the begining of the sorted array.
        self.transitions.appendleft([self.max_priority, transition])

    def batch_update(self, indexes, priorities):
        for index, priority in zip(indexes, priorities):
            self.max_priority = max(priority, self.max_priority)
            self.transitions[index][0] = priority
        self.sort()

    def sort(self):
        # save temporary copy of transitions as a list and clear the queue
        transitions = list(self.transitions)
        self.transitions.clear()

        transitions.sort(key=lambda entry: entry[0], reverse=True)
        self._init_transitions(transitions)

    def _init_transitions(self, transitions=[]):
        self.transitions = deque(transitions, maxlen=self.capacity)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, index):
        return self.transitions[index]
