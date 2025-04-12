from torch.utils.data import Sampler
from itertools import chain

import random

class CategorySampler(Sampler):
    def __init__(self, grouped_data, shuffle_categories=False):
        self.indices_by_category = []
        for category, samples in grouped_data.items():
            #if category == "person":
            self.indices_by_category.append(list(range(len(samples))))
        if shuffle_categories:
            random.shuffle(self.indices_by_category)
        
        self.category_order = list(chain(*self.indices_by_category))

    def __iter__(self):
        return iter(self.category_order)

    def __len__(self):
        return len(self.category_order)
