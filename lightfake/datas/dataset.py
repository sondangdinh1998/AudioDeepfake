from typing import Tuple, List, Union, Any, Optional

from omegaconf import DictConfig

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from lightfake.datas.audio import extract_feature, build_augmentation
from lightfake.utils.common import build_dataset


def collate_add_data(batch: List[Any]) -> Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    lengths = [len(feat) for feat in features]
    features = pad_sequence(features, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)

    targets = [b[1] for b in batch]
    targets = torch.tensor(targets, dtype=torch.long)

    return features, lengths, targets


class AudioDeepfakeDetectionDataset(Dataset):
    def __init__(
        self,
        labels: List[str],
        filepaths: Union[str, List[str]],
        augmentation: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()

        self.labels = labels
        self.dataset = build_dataset(filepaths)

        self.audio_augment, self.feature_augment = [], []
        if augmentation is not None:
            augments = build_augmentation(augmentation)
            self.audio_augment, self.feature_augment = augments

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        data = self.dataset[index]
        filepath = data["filepath"]
        label = data["label"]

        signal, sr = torchaudio.load(filepath)  # type: ignore
        for augment in self.audio_augment:
            signal = augment.apply(signal, sr)

        feature = extract_feature(signal, sr)
        for augment in self.feature_augment:
            feature = augment.apply(feature)

        label = self.labels.index(label)

        return feature.T, label

    def __len__(self) -> int:
        return len(self.dataset)
