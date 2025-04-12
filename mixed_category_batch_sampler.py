from torch.utils.data import Sampler
import random
import itertools

class MixedCategoryBatchSampler(Sampler):
    def __init__(self, flattened_data, batch_size, shuffle_categories=False):
        """
        Args:
            flattened_data: List of (category, img_id, captions).
            batch_size: Number of samples per batch.
            shuffle_categories: Whether to shuffle categories.
        """
        self.flattened_data = flattened_data
        self.batch_size = batch_size

        # Organize indices by category
        self.indices_by_category = {}
        for idx, (category, _, _) in enumerate(flattened_data):
            if category not in self.indices_by_category:
                self.indices_by_category[category] = []
            self.indices_by_category[category].append(idx)

        # Shuffle categories if specified
        self.categories = list(self.indices_by_category.keys())
        if shuffle_categories:
            random.shuffle(self.categories)

    def __iter__(self):
        # Prepare iterators for each category
        category_iters = {
            category: iter(self._infinite_cycle(self.indices_by_category[category]))
            for category in self.categories
        }

        while True:
            batch = []
            for category in self.categories:
                try:
                    batch.append(next(category_iters[category]))
                except StopIteration:
                    continue

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            
            if not any(category_iters.values()):
                break

    def __len__(self):
        return len(self.flattened_data) // self.batch_size

    @staticmethod
    def _infinite_cycle(iterable):
        """Helper function to cycle indefinitely through a list."""
        while True:
            for item in iterable:
                yield item
