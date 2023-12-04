from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from lightfake.datas.audio import extract_feature
from lightfake.utils.common import load_module


class LightningADD:
    def __init__(self, filepath: str, device: str):
        super().__init__()
        self.device = device
        self.network = self._load_checkpoint(filepath, device)

    def _load_checkpoint(self, filepath: str, device: str):
        checkpoint = torch.load(filepath, map_location="cpu")
        hparams = checkpoint["hyper_parameters"]
        weights = checkpoint["state_dict"]
        network = load_module(hparams["network"], weights["network"], device)
        return network

    @torch.inference_mode()
    def __call__(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> List[str]:
        features, lengths = self._preprocess(speeches, sample_rates)
        scores, *_ = self.network(features, lengths)
        return scores

    def _preprocess(
        self, speeches: List[torch.Tensor], sample_rates: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(speeches) == len(sample_rates), "The batch is mismatch."

        batches = [
            extract_feature(speech, sample_rates[i])
            for i, speech in enumerate(speeches)
        ]

        xs = [x.t() for x in batches]
        x_lens = [x.size(0) for x in xs]

        xs = pad_sequence(xs, batch_first=True).to(self.device)
        x_lens = torch.tensor(x_lens, device=self.device)

        return xs, x_lens
