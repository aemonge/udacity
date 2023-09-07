"""My Vocabulary class."""

from typing import List

class Vocabulary:
    """A vocabulary."""

    def __init__(self, debug=False) -> None:
        """
        Vocabulary.

        Attributes
        ----------
        word2index : dict
            A dictionary mapping words to indices.
        index2word : dict
            A dictionary mapping indices to words.
        n_words : int
            The number of words in the vocabulary.

        Parameters
        ----------
        debug : bool
            If true, will print useful debug information.
        """
        self.word2index = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4  # Starting count considering the special tokens
        self.debug = debug

    def add_sentence(self, sentence: List[str]) -> None:
        """
        Add a sentence to the vocabulary.

        Parameters
        ----------
        sentence : list
            A list of words.
        """
        for word in sentence:
            self.add_word(word)

    def add_word(self, word: str) -> None:
        """
        Add a word to the vocabulary.

        Parameters
        ----------
        word : str
            A word.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
            if self.debug:
                print(f"Added word: {word} with index: {self.n_words}")

    def to_index(self, word: str) -> int:
        """
        Convert a word to its index.

        Parameters
        ----------
        word : str
            A word.

        Returns
        -------
        int
            The index of the word or index of "<UNK>" for unknown words.
        """
        return self.word2index.get(word, self.word2index["<UNK>"])

    def to_word(self, index: int) -> str:
        """
        Convert an index to its word.

        Parameters
        ----------
        index : int
            An index.

        Returns
        -------
        str
            The word or "<UNK>" for unknown indices.
        """
        return self.index2word.get(index, "<UNK>")
