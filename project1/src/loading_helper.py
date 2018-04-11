import dlc_bci as bci


def load_data(train=True, one_khz=False):
    return bci.load(root='../data', train=train, one_khz=one_khz)
