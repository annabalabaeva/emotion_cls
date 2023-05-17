import torch
import torch.nn as nn


class LstmClassifier(nn.Module):

    def __init__(self, vocab_size, model_cfg):
        super(LstmClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, model_cfg["embedding_dim"])

        self.lstm = nn.LSTM(
            model_cfg["embedding_dim"],
            model_cfg["hidden_dim"],
            num_layers=model_cfg["n_layers"],
            bidirectional=model_cfg["bidirectional"],
            dropout=model_cfg["dropout"],
            batch_first=True
        )

        self.fc = nn.Linear(model_cfg["hidden_dim"] * 2, model_cfg["n_classes"])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden_state, cell_state) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        logits = self.fc(hidden)
        outputs = self.softmax(logits)

        return outputs
