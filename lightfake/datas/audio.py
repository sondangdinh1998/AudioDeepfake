from typing import Tuple, List, Any

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torchaudio


def extract_feature(
    waveform: torch.Tensor, sample_rate: int, n_lfcc: int = 40
) -> torch.Tensor:
    extractor = torchaudio.transforms.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": 512,
            "win_length": int(0.02 * sample_rate),
            "hop_length": int(0.01 * sample_rate),
        },
        log_lf=True,
    )

    lfcc = extractor(waveform).squeeze(0)
    delta = torchaudio.functional.compute_deltas(lfcc)
    delta_delta = torchaudio.functional.compute_deltas(delta)

    feature = torch.cat((lfcc, delta, delta_delta))
    mean = feature.mean(dim=1, keepdim=True)
    std = feature.std(dim=1, keepdim=True)
    feature = (feature - mean) / (std + 1e-12)

    return feature


def build_augmentation(config: DictConfig) -> Tuple[List[Any], List[Any]]:
    augment_config = config.get("audio_augment", {})
    audio_augments = [instantiate(cfg) for cfg in augment_config.values()]

    augment_config = config.get("feature_augment", {})
    feature_augments = [instantiate(cfg) for cfg in augment_config.values()]

    return audio_augments, feature_augments
