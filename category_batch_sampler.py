from torch.utils.data import Sampler
import random

class CategoryBatchSampler(Sampler):
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
        # Generate batches for each category
        for category in self.categories:
            indices = self.indices_by_category[category]
            if len(indices) > self.batch_size:
                # Shuffle indices for the category
                random.shuffle(indices)
            # Create batches
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return sum(len(indices) // self.batch_size + (1 if len(indices) % self.batch_size != 0 else 0)
                   for indices in self.indices_by_category.values())


'''
class CategoryBatchSampler(Sampler):
    def __init__(self, flattened_data, batch_size, shuffle_categories=False):
        """
        Args:
            flattened_data: List of (category, img_id, captions).
            batch_size: Number of samples per batch.
            shuffle_categories: Whether to shuffle categories.
        """
        self.flattened_data = flattened_data
        self.batch_size = batch_size
        self.shuffle_categories = shuffle_categories

        # Organize indices by category
        self.indices_by_category = {}
        for idx, (category, _, _) in enumerate(flattened_data):
            if category not in self.indices_by_category:
                self.indices_by_category[category] = []
            self.indices_by_category[category].append(idx)

        # Precompute batches for all categories
        self.precomputed_batches = self._precompute_batches()

    def _precompute_batches(self):
        """Precompute batches for all categories."""
        precomputed_batches = []
        categories = list(self.indices_by_category.keys())
        
        # Shuffle categories if specified
        if self.shuffle_categories:
            random.shuffle(categories)

        for category in categories:
            indices = self.indices_by_category[category]
            # Shuffle indices for the category
            random.shuffle(indices)
            # Create batches
            for i in range(0, len(indices), self.batch_size):
                precomputed_batches.append(indices[i:i + self.batch_size])

        return precomputed_batches

    def __iter__(self):
        """Yield precomputed batches."""
        # Shuffle batches for each epoch if desired
        if self.shuffle_categories:
            random.shuffle(self.precomputed_batches)
        yield from self.precomputed_batches

    def __len__(self):
        """Return the total number of precomputed batches."""
        return len(self.precomputed_batches)
'''
