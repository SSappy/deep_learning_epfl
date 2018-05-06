class Normalize(object):
    """
    Data pre-processing class to normalize data so the values are in the range [new_min, new_max].
    """

    def __init__(self, min_, max_, new_min=0, new_max=1):
        """
        Initializer.
        :param min_: Min of the un-normalized data.
        :param max_: Max of the un-normalized data.
        :param new_min: Min of the new data.
        :param new_max: Max of the new data.
        """
        self.min = min_
        self.max = max_
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, data):
        """
        Normalize a given data point.
        :param data: Data point to normalize.
        :return: Normalized data.
        """
        data = (self.new_max - self.new_min)*(data - self.min)/(self.max - self.min) + self.new_min

        return data


class Standardize(object):
    """
    Data pre-processing class to standardize data so the values have a fixed mean and standard deviation.
    """

    def __init__(self, mean, std):
        """
        Initializer.
        :param mean: Mean of the un-standardized data.
        :param std: Std of the un-standardized data.
        """
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Standardize a given data point.
        :param data: Data point to standardize.
        :return: Standardized data.
        """
        data = (data - self.mean)/self.std

        return data

