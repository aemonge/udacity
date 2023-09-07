"""My main Seq2Seq model, with encoder and decoder layers."""

import random
from typing import Any, Tuple

import torch
from torch import Tensor, nn

train_on_gpu = torch.cuda.is_available()

class Encoder(nn.Module):
    """My Encoder layer."""

    def __init__(self, input_size: int, hidden_size: int, embedding_dim: int) -> None:
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_dim)  # , hidden_size?
        self.lstm_1 = nn.LSTM(  # type: ignore
            embedding_dim, self.hidden_size * 2, 4, dropout=0.5, batch_first=True
        )
        self.lstm_2 = nn.LSTM(  # type: ignore
            self.hidden_size * 2, self.hidden_size * 4, 4, dropout=0.2
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Any, Any]:
        """
        Forward method for the tensor network.

        Todo
        ----
        * Define the Any's at the return type

        Parameters
        ----------
            x : Tensor
                the src vector

        Returns
        -------
            x : Tensor
                the encoder outputs
            h :
                the hidden state
            c :
                the cell state
        """
        # x = self.embedding(x)
        # x, (h, c) = self.lstm_1(x)
        # x, (h, c) = self.lstm_2(x, (h, c))
        #
        # return x, h, c
        # Add this line to check the shape of the input tensor
        print("Input shape:", x.shape)
        # Add this line to check the values of the input tensor
        print("Input values:", x)
        x = self.embedding(x)
        # Add this line to check the shape of the embedded tensor
        print("Embedded shape:", x.shape)
        # Add this line to check the values of the embedded tensor
        print("Embedded values:", x)
        x, (h, c) = self.lstm_1(x)
        # Add this line to check the shape of the LSTM 1 output
        print("LSTM 1 output shape:", x.shape)
        # Add this line to check the values of the LSTM 1 output
        print("LSTM 1 output values:", x)
        x, (h, c) = self.lstm_2(x, (h, c))
        # Add this line to check the shape of the LSTM 2 output
        print("LSTM 2 output shape:", x.shape)
        # Add this line to check the values of the LSTM 2 output
        print("LSTM 2 output values:", x)
        return x, h, c


class Decoder(nn.Module):
    """My decoder with a LSTM layer."""

    def __init__(self, hidden_size: int, output_size: int, embedding_dim: int) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, embedding_dim)
        self.lstm_1 = nn.LSTM(  # type: ignore
            embedding_dim, hidden_size, num_layers=2, dropout=0.3, batch_first=True
        )

        self.output = nn.Linear(hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor]) -> Tuple[Any, Any]:
        """
        Forward method for the tensor network.

        Notes
        -----
            LSTMs, or Long Short-Term Memory units, are a type of recurrent neural
            network (RNN) that have feedback connections. This means they can process
            sequences of data, retaining a 'memory' of the previous states of the data
            as they process each new timestep. This makes them ideal for tasks
            involving sequential data, like time series prediction, natural language
            processing, and more.

        Parameters
        ----------
            x : Tensor
                the src vector
            h : Tuple[Tensor, Tensor]
                The hidden statek cell state

        Returns
        -------
            x :
                The prediction
            h :
                the hidden state
        """
        x = self.embedding(x)
        x, h = self.lstm_1(x, h)

        x = self.output(x)
        x = self.softmax(x)

        return x, h


class Seq2Seq(nn.Module):
    """The main NN architecture."""

    def __init__(
        self,
        encoder_input_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        decoder_output_size: int,
    ) -> None:
        super(Seq2Seq, self).__init__()
        self.hidden_dim = 10
        self.encoder = Encoder(encoder_input_size, encoder_hidden_size, self.hidden_dim)
        self.decoder = Decoder(decoder_hidden_size, decoder_output_size, self.hidden_dim)

    def forward(
        self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5
    ) -> Tensor:
        """
        Forward method for the tensor network.

        Todo
        ----
        * Define the Any's at the return type

        Parameters
        ----------
            src : Tensor
                the src vector
            trg : Tensor
                the trg vector
            teacher_forcing_ratio : float
                the teacher forcing ratio

        Returns
        -------
            x : Tensor
                the prediction
            h :
                the hidden state
            c :
                the cell state
        """
        # Initialize an empty tensor to store decoder outputs
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.decoder.output_size, device=trg.device)

        # First, the source sequence (src) is passed through the encoder
        _, hidden, cell = self.encoder(src)

        # The initial decoder input is the <sos> token, i.e., the first token of the target sequence
        decoder_input = trg[:, 0].unsqueeze(1)

        # Iteratively decode each time step
        for t in range(1, trg.shape[1]):
            # Pass the decoder input, hidden, and cell states to the decoder
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            # Store the decoder output in the outputs tensor
            outputs[:, t] = decoder_output.squeeze(1)

            # Decide if we will use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the most probable token
            top1 = decoder_output.argmax(2)

            # If teacher forcing, use the actual next token as next input. If not, use the predicted token
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def init_hidden(self, batch_size: int) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # Initialize hidden states for Encoder
        # Encoder has 2 LSTMs with 4 layers each
        encoder_h = weight.new(8, batch_size, self.encoder.hidden_size * 4).zero_()
        encoder_c = weight.new(8, batch_size, self.encoder.hidden_size * 4).zero_()

        # Initialize hidden states for Decoder
        # Decoder has 1 LSTM with 2 layers
        decoder_h = weight.new(2, batch_size, self.decoder.hidden_size).zero_()
        decoder_c = weight.new(2, batch_size, self.decoder.hidden_size).zero_()

        if train_on_gpu:
            encoder_h, encoder_c = encoder_h.cuda(), encoder_c.cuda()
            decoder_h, decoder_c = decoder_h.cuda(), decoder_c.cuda()

        return ((encoder_h, encoder_c), (decoder_h, decoder_c))
