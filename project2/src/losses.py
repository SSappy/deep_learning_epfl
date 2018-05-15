from torch import FloatTensor

class MSE:
    def __init__(self):
        pass

    @staticmethod
    def compute_loss(predictions, targets):
        """
        :param predictions: predictions (torch Tensor of shape (num_samples, 1)
        :param targets: targets (torch Tensor of shape (num_samples, 1)
        :return: tensor of shape (num_samples, 1) containing the squared errors (MSE is the mean of this tensor).
        """
        # make sure shapes are the same
        assert predictions.shape == targets.shape
        # make sure we have a torch Tensor
        pred = FloatTensor(predictions)
        return ((targets-pred)**2).mean(dim=1)

    @staticmethod
    def compute_grad(predictions, targets):
        """
        Compute gradient of loss with respect to predictions.
        :param predictions:
        :param targets:
        :return:
        """
        assert predictions.shape == targets.shape
        return (predictions - targets).mean(0)