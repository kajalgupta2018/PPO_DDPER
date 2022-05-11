import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        #print("priority value :", s) 
        #print("tree :", self.tree) 
        #print("length tree :", len(self.tree)) 

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def reach_max(self):
        if self.n_entries < self.capacity:
           return False
        else:
           return True 
            


    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

        return idx    

    def get_node(self, p):
        idx =0
        
        for x in self.tree[1:]:

            #print("x:", x)
            left = 2 * idx
            left_sum = x[left]
            if p < left_sum:
                idx = left
            else:
                idx = left + 1
                p-= left_sum

        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
        




    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)


    def get_data(self):
        return self.data

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
