import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab

CLASS_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


class TweetsDataset(Dataset):

    def __init__(self, data, tokens_vocab=None, vocab_path=None, load_from_file=False):
        self.texts = []
        tokens = []
        self.token_ids = []
        self.labels = []

        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        # for validation and test sets during training
        if tokens_vocab:
            self.tokens_vocab = tokens_vocab
            for item in data:
                self.texts.append(item["text"])
                self.labels.append(item["label"])
                tokens_i = tokenizer(item["text"])
                tokens.append(tokens_i)

        # for tests from pretrained model
        elif load_from_file:
            self.tokens_vocab = read_vocab(vocab_path)
            for item in data:
                self.texts.append(item["text"])
                self.labels.append(item["label"])
                tokens_i = tokenizer(item["text"])
                tokens.append(tokens_i)

        # for training
        else:
            counter = Counter()
            for item in data:
                self.texts.append(item["text"])
                self.labels.append(item["label"])
                tokens_i = tokenizer(item["text"])
                tokens.append(tokens_i)
                counter.update(tokens_i)
            self.tokens_vocab = vocab(
                counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=2)
            self.tokens_vocab.set_default_index(self.tokens_vocab['<unk>'])
            save_vocab(self.tokens_vocab, vocab_path)

        self.PAD_IDX = self.tokens_vocab['<pad>']
        self.EOS_IDX = self.tokens_vocab['<eos>']
        self.BOS_IDX = self.tokens_vocab['<bos>']

        for tokens_i in tokens:
            self.token_ids.append([self.tokens_vocab[t] for t in tokens_i])

        del tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.LongTensor([self.BOS_IDX] + self.token_ids[idx] + [self.EOS_IDX])
        label = self.labels[idx]
        return x, label


def collate_texts(batch, pad_idx):
    tokens_batch = [b[0] for b in batch]
    labels_batch = torch.LongTensor([b[1] for b in batch])
    tokens_len = torch.LongTensor([len(t) for t in tokens_batch])
    tokens_padded = pad_sequence(tokens_batch, padding_value=pad_idx, batch_first=True)
    return {
        "tokens": tokens_padded,
        "tokens_len": tokens_len,
        "labels": labels_batch
    }


def read_vocab(path):
    import pickle
    pkl_file = open(path, 'rb')
    vocab_ = pickle.load(pkl_file)
    pkl_file.close()
    return vocab_


def save_vocab(vocab_, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab_, output)
    output.close()
