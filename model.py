"""Device-independent version of the server's encoder -> linear -> LIF model."""

import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer


class LogSNN(nn.Module):
    def __init__(self, model_name, backbone="t5"):
        super().__init__()
        if backbone == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
        self.dense = nn.Linear(self.encoder.config.hidden_size, 2)
        self.lif = snn.Leaky(
            beta=1.0, threshold=1.0,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=True, output=True,
        )

    def forward(self, sentences):
        self.lif.reset_hidden()
        device = self.dense.weight.device
        tokens = self.tokenizer(
            sentences, add_special_tokens=True, padding=True,
            truncation=True, max_length=512, return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.encoder(**tokens).last_hidden_state[:, 0, :]
        _, membrane = self.lif(self.dense(hidden))
        return membrane
