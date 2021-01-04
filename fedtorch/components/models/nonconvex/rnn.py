# -*- coding: utf-8 -*-
import torch.nn as nn


__all__ = ['rnn']

class RNN(nn.Module):
    def __init__(self, dataset, input_size, hidden_size, output_size, batch_size, n_layers=1):
        super(RNN, self).__init__()
        self.dataset = dataset
        if self.dataset not in ['shakespeare']:
            raise NotImplementedError
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        

        self.init_hidden(self.batch_size)
    
    def forward(self, input):
        if self.hidden.size(1) != input.size(0):
            self.init_hidden(input.size(0))
        if self.hidden.device != input.device:
            self.hidden = self.hidden.to(input.device)
        input = self.encoder(input)
        output, h = self.gru(input, self.hidden.detach())
        self.hidden.data = h.data
        output = self.decoder(output)
        return output.permute(0,2,1)
    
    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size
        weight = next(self.parameters())
        self.hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_size)
        return

def rnn(args):
    return RNN(
        dataset=args.data, input_size=args.vocab_size,
        hidden_size=args.rnn_hidden_size, output_size=args.vocab_size, batch_size=args.batch_size)