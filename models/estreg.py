from torch import nn
from .estimator import Estimator
from .regressor import Regressor


class EstReg(nn.Module):
    super(Estimator, self).__init__()
    self.estimator = Estimator()
    self.regressor = Regressor()

    def forward(self, input):
        return regressor(imput, estimator(input)[-1])