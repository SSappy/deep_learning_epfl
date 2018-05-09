import random
from torch import FloatTensor

def down_sample(x, regular=True):
    """
    :param x: input tensor of shape (316, 28, 500)
    :param regular: boolean that indicates whether we sample at regular intervals
    """
    if regular:
        k = random.randint(0, 10)
        idx = [k+i for i in range(50)]
        return x[:, :, idx]
    if not regular:
        ks = [random.randint(0, 10) for _ in range(50)]
        idx = [k+i for k, i in zip(ks, list(range(0, 500, 10)))]
        return x[:, :, idx]