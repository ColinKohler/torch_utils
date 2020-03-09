import torch
import torch.nn as nn
from torch.autograd import Varaible

class ConvLSTM(nn.Module):
  def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
    super(ConvLSTM, self).__init__()

    self.height, self.width = input_size
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.kernel_size = kernel_size
    self.padding = kernel_size[0] // 2, kernel_size[1] // 2
    self.bias = bias

    self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                          out_channels=4 * self.hidden_dim,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=self.bias)

  def forward(self, inp, state):
    h_state, c_state = state

    combined = torch.cat([inp, h_state], dim=1)
    conv = self.conv(combined)
    cc_i, cc_f, cc_o, cc_g = torch.splot(conv, self.hidden_dim, dim=1)

    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.tanh(cc_g)

    c_next = f * c_state + i * g
    h_next = o * torch.tanh(c_next)

    return h_next, c_next

  def init_hidden(self, batch_size):
    return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=self.conv.weight.device)),
            Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=self.conv.weight.device)))

class ConvLSTMBlock(nn.Module):
  def __init__(self, input_size, input_dim, hidden_dim, kernel_size, stride, bias):
    super(ConvLSTMBlock, self).__init__()

    self.conv_lstm = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, bias)
    self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, bias=bias)
    self.relu = nn.LeakyReLU(0.1, inplace=True)

  def forward(self, inp):
    # inp = (b, t, c, h,  w)

    batch_size = inp.size(0)
    seq_len  inp.size(1)
    h, c = self.conv_lstm.init_hidden(batch_size)

    for t in range(seq_len):
      h, c = self.conv_lstm(inp[:,t], state=[h,c])
    out = self.relu(self.conv(self.relu(h)))

    return out
