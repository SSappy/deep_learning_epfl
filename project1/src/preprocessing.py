class Normalize(object):
    """
    Normalize data so the values are in the range [new_min, new_max]
    """

    def __init__(self, min_, max_, new_min=0, new_max=1):
        """
        :param min_: Min of the un-normalized data
        :param max_: Max of the un-normalized data
        :param new_min: Min of the new data
        :param new_max: Max of the new data
        """
        self.min = min_
        self.max = max_
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, data):
        data = (self.new_max - self.new_min)*(data - self.min)/(self.max - self.min) + self.new_min

        return data


class Standardize(object):
    """
    Standardize data so the values have a fixed mean and standard deviation
    """

    def __init__(self, mean, std, new_mean=0, new_std=1):
        """
        :param mean: Mean of the un-standardized data
        :param std: Std of the un-standardized data
        :param new_mean: Mean of the new data
        :param new_std: Std of the new data
        """
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = (data - self.mean)/self.std

        return data
