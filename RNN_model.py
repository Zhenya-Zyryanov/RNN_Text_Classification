import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(s, lang='english'):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'<.*?>', ' ', s)
    if lang == 'english':
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
    else:
        s = re.sub(r'[^а-я0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize(s, lang='english'):
    if lang == 'russian':
        STOP = set(stopwords.words('russian'))
    else:
        STOP = set(stopwords.words('english'))
    toks = word_tokenize(s)
    toks = [t for t in toks if t.isalpha() and t not in STOP]
    return toks

def text_to_seq(tokens, word2idx, max_len):
    PAD = 0
    UNK = 1
    seq = [word2idx.get(t, UNK) for t in tokens]
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [PAD] * (max_len - len(seq))

def seq_len(x, max_len, padding_idx):
    return max_len - (x == padding_idx).sum(dim=-1)

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, label_encoder, max_len, lang='english'):
        self.word2idx = word2idx
        self.max_len = max_len
        self.lang = lang

        cleaned = [clean_text(t, lang=lang) for t in texts]
        tokenized = [tokenize(t, lang=lang) for t in cleaned]

        sequences = [text_to_seq(toks, self.word2idx, max_len=self.max_len) for toks in tokenized]
        self.X = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(label_encoder.transform(labels), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNN(nn.Module):
    def __init__(self, vocab_size, embd_dim, rnn_hidden_size, num_classes, num_layers, rnn_type="LSTM",
                 padding_idx=0, pretrained_embedding=None, freeze_embedding=True, dropout=0.0, bidirectional=False):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.embd_dim = embd_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=0)

        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(pretrained_embedding)
            if freeze_embedding:
                self.embedding.weight.requires_grad = False

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embd_dim, rnn_hidden_size, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embd_dim, rnn_hidden_size, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(embd_dim, rnn_hidden_size, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)

        linear_input_size = rnn_hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Linear(linear_input_size, linear_input_size // 2)
        self.norm = nn.LayerNorm(linear_input_size // 2)
        self.af = nn.LeakyReLU()
        self.fc2 = nn.Linear(linear_input_size // 2, num_classes)

    def forward(self, x):
        batch_size, max_len = x.size()
        lens = seq_len(x, max_len, self.padding_idx)

        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        
        if self.bidirectional:
            forward_out = rnn_out[:, :, :self.rnn_hidden_size]
            backward_out = rnn_out[:, :, self.rnn_hidden_size:]
            
            forward_final = forward_out[torch.arange(batch_size), lens - 1, :]
            backward_final = backward_out[torch.arange(batch_size), lens - 1, :]
            
            final = torch.cat((forward_final, backward_final), dim=1)
        else:
            final = rnn_out[torch.arange(batch_size), lens - 1, :]

        out = self.fc(final)
        out = self.af(out)
        out = self.norm(out)
        out = self.fc2(out)
        return out