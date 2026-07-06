"""
data/dataset.py                                                      [HGCAN]
Loads cached AssemblyPairData (.pt) for training. One item = one assembly
(carrying all its candidate pairs). Batch size 1 is the simple, robust default
for the base version; PyG batching can be added later via the __inc__ rules.
"""
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset


def split_ids(assembly_dir, val_frac=0.2, seed=42):
    ids = sorted(p.stem for p in Path(assembly_dir).glob("*.pt"))
    random.Random(seed).shuffle(ids)
    cut = int((1 - val_frac) * len(ids))
    return ids[:cut], ids[cut:]


def official_splits(assembly_dir, split_json, val_frac=0.15, seed=42):
    """Three-way split using the dataset's official train_test.json.

    The official file lists ALL 8,251 assemblies as {"train":[...], "test":[...]}.
    We intersect with what's actually cached (only joint-bearing assemblies were
    built), hold out the official TEST untouched, and carve a val set FROM the
    official train for model selection.

    Returns (train_ids, val_ids, test_ids), each a list of cache stems.
    """
    import json
    cached = {p.stem for p in Path(assembly_dir).glob("*.pt")}
    off = json.loads(Path(split_json).read_text(encoding="utf-8"))
    off_train = [i for i in off["train"] if i in cached]
    test_ids = [i for i in off["test"] if i in cached]

    rng = random.Random(seed)
    rng.shuffle(off_train)
    cut = int((1 - val_frac) * len(off_train))
    train_ids, val_ids = off_train[:cut], off_train[cut:]

    print(f"[split] official: train {len(off['train'])}, test {len(off['test'])} "
          f"(full dataset)")
    print(f"[split] cached & usable: train {len(off_train)} -> "
          f"(fit {len(train_ids)} + val {len(val_ids)}), test {len(test_ids)}")
    return train_ids, val_ids, test_ids


class HGCANCache(Dataset):
    def __init__(self, assembly_dir, ids):
        self.paths = [Path(assembly_dir) / f"{i}.pt" for i in ids]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # weights_only=False: our cache stores a custom AssemblyPairData (a pickled
        # class), which PyTorch 2.6's default weights_only=True refuses. The cache is
        # our own trusted artifact, so False is correct here.
        return torch.load(self.paths[i], weights_only=False)
