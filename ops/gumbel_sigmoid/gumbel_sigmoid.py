import torch
from torch import nn


class GumbelSigmoid(nn.Module):
    def __init__(self, max_T, decay_alpha, decay_method='exp', start_iter=0):
        super(GumbelSigmoid, self).__init__()

        self.max_T = max_T
        self.cur_T = max_T
        self.decay_alpha = decay_alpha
        self.decay_method = decay_method
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8
        # self.cur_T = (self.decay_alpha ** start_iter) * self.cur_T

        assert self.decay_method in ['exp', 'step', 'cosine']

    def forward(self, x):
        # Shape <x> : [N, C, H, W]
        # Shape <r> : [N, C, H, W]
        r = 1 - x
        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        # Generate Noise
        x_N = torch.rand_like(x)
        r_N = torch.rand_like(r)
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        # Get Final Distribution
        x = x + x_N
        x = x / (self.cur_T + self.p_value)
        r = r + r_N
        r = r / (self.cur_T + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)
        x = x[:, [0], :, :]

        self.cur_T = self.cur_T * self.decay_alpha

        return x


if __name__ == '__main__':
    pass
    # ToDo: Test Code Here.
    # _test_T = 0.6
    # Block = GumbelSigmoid(_test_T, 1.0)
