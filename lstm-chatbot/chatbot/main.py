"""
My main Chat-Bot file, potentially to be converted to the main CLI file.
"""

import contextlib
import os
from collections import Counter

import torch
# Disable all progress bars to avoid the Udacity lack up Jupyter Updates
import tqdm
from datasets import disable_progress_bar, load_dataset
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab

from .model import Seq2Seq
from .vocabulary import Vocabulary

train_on_gpu = torch.cuda.is_available()


class ChatBot:
    """
    A ChatBot utilizing a Sequence to Sequence model.

    Attributes
    ----------
    debug : bool
        Whether to print debugging information.
    vocabulary : Vocabulary
        The vocabulary object.
    model : Seq2Seq
        The sequence-to-sequence model.
    dataset : Dataset
        The dataset to be used.
    """

    def __init__(self, debug=False):
        """
        Initialize the ChatBot.

        Parameters
        ----------
        debug : bool, optional
            Whether to print debugging information. Defaults to False.
        """
        self.debug = debug
        self.vocabulary = Vocabulary(debug=self.debug)
        self.model = None
        self.dataset = None

    @staticmethod
    def get_batches(arr, batch_size: int, seq_length: int):
        """Create a generator that returns batches of size
        batch_size x seq_length from arr.

        Arguments
        ---------
            arr : dict
                Dictionary containing data you want to make batches from.
            batch_size :  integer
                Batch size, the number of sequences per batch.
            seq_length : integer
                Number of encoded chars in a sequence.
        """

        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr)  # Convert to PyTorch tensor

        # Determine the number of batches we can make
        total = batch_size * seq_length
        n_batches = len(arr) // total

        # Keep only enough characters to make full batches
        arr = arr[: n_batches * total]

        # Reshape into batch_size rows
        arr = arr.reshape((batch_size, -1))

        # Iterate over the batches using a window of size seq_length
        for n in range(0, arr.shape[1], seq_length):
            x = arr[:, n : n + seq_length]
            y = torch.zeros_like(x)
            print(f"x slice shape: {x.shape}")
            print(f"y slice shape: {y.shape}")

            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y

    def load_hyperparameters(self):
        """
        Load the hyperparameters for the model.
        """
        # Set your hyperparameters here. For instance:
        self.embedding_dim = 256
        self.hidden_size = 512
        self.encoder_input_size = len(self.vocabulary.word2index)
        self.decoder_output_size = len(self.vocabulary.word2index)
        self.learning_rate = 0.001
        self.epochs = 10
        self.batch_size = 128
        self.clip = 5.0

    def initialize_model(self):
        """
        Initialize the Seq2Seq model with the given hyperparameters.
        """
        self.model = Seq2Seq(
            encoder_input_size=self.encoder_input_size,
            encoder_hidden_size=self.hidden_size,
            decoder_hidden_size=self.hidden_size,
            decoder_output_size=self.decoder_output_size,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        """
        Train the model on the dataset.
        """
        # Define your training loop here
        if not self.dataset:
            raise ValueError("Dataset has not been loaded.")

        if train_on_gpu:
            self.model.cuda()

        counter = 0
        n_chars = len(self.vocabulary.word2index)
        if self.debug:
            print("Vocabulary Size (n_chars):", n_chars)
        for epoch in range(self.epochs):
            if self.debug:
                print(f"Training epoch: {epoch+1}/{self.epochs}")

            # Initialize hidden state
            h = self.model.init_hidden(batch_size=self.batch_size)

            # for x, y in self.get_batches(
            #     self.train_data, batch_size=self.batch_size, seq_length=n_chars
            # ):
            # for (attention_mask, input_ids, label, token_type_ids) in self.dataloader:
            for _, x, y, __ in self.dataloader:
                counter += 1

                if self.debug:
                    print(f"Shapes: x, y: {(x.shape, y.shape)}", end="")

                x = torch.nn.functional.one_hot(x, num_classes=n_chars)
                if self.debug:
                    print(f"One-hot encoded x shape: {x.shape}")

                x, y = torch.from_numpy(x), torch.from_numpy(y)
                if train_on_gpu:
                    x, y = x.cuda(), y.cuda()

                if self.debug:
                    print(f"x shape: {x.shape}, y shape: {y.shape}")

                self.model.zero_grad()
                if self.debug:
                    print("Zero Graded....", end="")

                output, h = self.model(x, h)
                if self.debug:
                    print(f"output shape: {output.shape}", end="")

                loss = self.criterion(output, y)
                if self.debug:
                    print("\rGot Loss....", end="")

                loss.backward()
                if self.debug:
                    print("\rGot Backward....", end="")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                if self.debug:
                    print("\rGot Optimized....", end="")

                if counter % 100 == 0:
                    if self.debug:
                        print("\r", end="")
                    print(f"Loss: {loss.item()}")

    @contextlib.contextmanager
    def suppress_output(self):
        """
        Udacity has NOT update it's jupyter notebook, use this to suppress

        ImportError: FloatProgress not found. Please update jupyter and ipywidgets.
        See https://ipywidgets.readthedocs.io/en/stable/user_install.html
        """
        with open(os.devnull, "w") as fnull:
            old_out = os.dup(1)
            old_err = os.dup(2)
            os.dup2(fnull.fileno(), 1)
            os.dup2(fnull.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(old_out, 1)
                os.dup2(old_err, 2)
                os.close(old_out)
                os.close(old_err)

    def load_dataset(self, dataset_name="glue"):
        """
        Load a dataset using huggingface datasets.
        Parameters
        ----------
        dataset_name : str, optional
        The name of the dataset. Defaults to "glue".
        """

        def encode_sentence(s, vocab):
            if type(s) == list:
                s = " ".join(s)
            tokenizer = get_tokenizer("basic_english")
            tokenized_sentence = tokenizer(s)
            index_tensor = [vocab[token] for token in tokenized_sentence]
            return {"encoded_sentence": torch.tensor(index_tensor)}

        counter = Counter()

        with self.suppress_output():
            self.dataset = load_dataset(dataset_name, "mrpc", split="train")

            for example in self.dataset["sentence1"]:
                tokenizer = get_tokenizer("basic_english")
                counter.update(tokenizer(example))
            vocab = Vocab(counter)

            self.dataset = self.dataset.map(
                lambda e: encode_sentence(e["sentence1"], vocab),
                batched=True,
            )

        if train_on_gpu:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"],
                device="cuda",
            )
        else:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"],
            )

        self.dataloader = DataLoader(self.dataset)

    def assert_seq2seq(self):
        """
        Assert the Seq2Seq model to ensure its correctness.
        """
        # Add your assertion code here
        pass

    def use_pretrained_embeddings(self, embeddings):
        """
        Optionally use pretrained word embeddings in the model.

        Parameters
        ----------
        embeddings : Any
            The pre-trained embeddings.
        """
        # If you decide to use pre-trained embeddings, implement this method.
        pass

    def evaluate(self):
        """
        Evaluate the model's performance.
        """
        # You can implement methods to evaluate your model's performance here.
        pass

    def interact(self):
        """
        Interact with the chatbot.
        """
        # Here you'll write code to interact with the model in a dialogue manner.
        pass


def main():
    """
    Main function to run the chatbot.
    """
    bot = ChatBot(debug=True)

    disable_progress_bar()
    tqdm.tqdm.disable = True
    bot.load_dataset()
    bot.load_hyperparameters()
    bot.initialize_model()

    bot.train()

    bot.interact()


if __name__ == "__main__":
    main()
