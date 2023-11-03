from typing import List


class CCLArguments(object):
    def __init__(self,
                 difficulty_order: List[List] = [
                     [6, 5], [6, 5, 7, 4, 1], [6, 5, 7, 4, 1, 3, 2]],
                 n_epochs: List = [5, 5, 10]):
        self.difficulty_order = difficulty_order
        self.n_epochs = n_epochs
