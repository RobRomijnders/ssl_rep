import numpy as np
from sklearn.model_selection import train_test_split


def generate_data(num_samples):
    """
    Generate random data from the two half moons.

    Each sample get perturbed with a small Gaussian noise of variance 0.1^2
    :param num_samples:
    :return:
    """
    radius = 1  # Radius of the half moons
    angle = np.random.rand(num_samples) * 2 * np.pi  # Sample a random angle with the half moons

    # Transform points from the real line to the unit circle
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    # Generate two half moons by translating the lower half
    lbl = np.greater_equal(angle, np.pi).astype(np.int32)
    x += lbl  # Translate the x coordinate by 1 for the positive labels
    y += 0.5 * lbl

    # Collect the data
    data = np.stack((x, y), axis=-1)

    # Add some noise to the data to spice up the problem
    data += 0.1 * np.random.randn(*data.shape)

    return data, lbl


class DataLoader:
    """
    Small wrapper to contain the data
    """
    def __init__(self, X_labeled, y_labeled, X_unlabeled, batch_size=32):
        data = dict()

        # Either supply some data to be split or manually define the trainings data
        if False:
            data['X_train'], data['X_val'], data['y_train'], data['y_val'] = train_test_split(X_labeled, y_labeled)
        else:
            data['X_train'], data['y_train'] = np.array([[1.0, 0.0], [0.1, 0.0]]), np.array([0, 1])
        data['X_val'], data['y_val'] = X_labeled, y_labeled
        data['X_unlabeled'] = X_unlabeled

        self.data = data
        self.batch_size = batch_size

        # Print data sizes
        print(f'Data set sizes\n'
              f'Training: {len(data["X_train"])} samples\n'
              f'Validation: {len(data["X_val"])} samples\n'
              f'Unlabeled: {len(data["X_unlabeled"])} samples')

    def sample(self, data_split, do_ssl=False):
        """
        Sample a batch of data from the internal data.
        :param data_split: which split of the data you want. Can be 'train' or 'val'
        :param do_ssl: if True, then half the batch will be unlabeled, if False, all data will be labeled
        :return: batch_data and batch_labels
        """
        assert data_split in ['train', 'val']

        # Sample random indices and slice the data sets
        idx = np.random.randint(0, len(self.data['y_' + data_split]), size=self.batch_size)
        X = self.data['X_' + data_split][idx]
        y = self.data['y_' + data_split][idx]

        if do_ssl:
            # If we are doing SSL, then append another batch of batch_size with unlabeled samples.
            # The labels get filled with the dummy -1
            idx = np.random.randint(0, len(self.data['X_unlabeled']), size=self.batch_size)
            X_unlabeled = self.data['X_unlabeled'][idx]

            X = np.concatenate((X, X_unlabeled), axis=0)
            y = np.concatenate((y, np.full(self.batch_size, fill_value=-1)), axis=0)
        return X, y



