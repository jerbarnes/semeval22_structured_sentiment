import torch
from awd.weight_drop import WeightDrop
from awd.locked_dropout import LockedDropout


class EnhancedLSTM(torch.nn.Module):
    """
    A wrapper for different recurrent dropout implementations, which
    pytorch currently doesn't support nativly.
    
    Uses multilayer, bidirectional lstms with dropout between layers
    and time steps in a variational manner.

    "allen" reimplements a lstm with hidden to hidden dropout, thus disabling
    CUDNN. Can only be used in bidirectional mode.
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`

    "drop_connect" uses default implemetation, but monkey patches the hidden to hidden
    weight matrices instead.
    `Regularizing and Optimizing LSTM Language Models
        <https://arxiv.org/abs/1708.02182>`
    
    "native" ignores dropout and uses the default implementation.
    """

    def __init__(self,
                 lstm_type,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.lstm_type = lstm_type

        if lstm_type == "allen":
            from AllenNLPCode.custom_stacked_bidirectional_lstm import CustomStackedBidirectionalLstm
            self.provider = CustomStackedBidirectionalLstm(
                input_size, hidden_size, num_layers, ff_dropout,
                recurrent_dropout)
        elif lstm_type == "drop_connect":
            self.provider = WeightDropLSTM(
                input_size,
                hidden_size,
                num_layers,
                ff_dropout,
                recurrent_dropout,
                bidirectional=bidirectional)
        elif lstm_type == "native":
            self.provider = torch.nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True)
        else:
            raise Exception(lstm_type + " is an invalid lstm type")

    # Expects unpacked inputs in format (batch, seq, features)
    def forward(self, inputs, hidden, lengths):
        if self.lstm_type in ["allen", "native"]:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True)

            output, _ = self.provider(packed, hidden)

            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

            return output
        elif self.lstm_type == "drop_connect":
            return self.provider(inputs, lengths)


class WeightDropLSTM(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.locked_dropout = LockedDropout()
        self.lstms = [
            torch.nn.LSTM(
                input_size
                if l == 0 else hidden_size * (1 + int(bidirectional)),
                hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True) for l in range(num_layers)
        ]
        if recurrent_dropout:
            self.lstms = [
                WeightDrop(lstm, ['weight_hh_l0'], dropout=recurrent_dropout)
                for lstm in self.lstms
            ]

        self.lstms = torch.nn.ModuleList(self.lstms)
        self.ff_dropout = ff_dropout
        self.num_layers = num_layers

    def forward(self, input, lengths):
        """Expects input in format (batch, seq, features)"""
        output = input
        for lstm in self.lstms:
            output = self.locked_dropout(
                output, batch_first=True, p=self.ff_dropout)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                output, lengths, batch_first=True)
            output, _ = lstm(packed, None)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        return output
