import logging
import torch.utils.data


def collate_tokens(
    values,
    pad_idx=1,
    left_pad=False,
    pad_to_length=None,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


class NumericalPairedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        src,
        tgt,
        shuffle=True,
    ):
        super().__init__()
        self.src = src
        self.tgt = tgt
        self.shuffle = shuffle
    
    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index] if self.tgt is not None else None
        example = {
            "id": index, 
            "source": src_item,
            "target": tgt_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        id = torch.LongTensor([s["id"] for s in samples])
        
        src_tokens = collate_tokens(
            [s["source"] for s in samples]
        )
        target = collate_tokens(
            [s["target"] for s in samples]
        )    
        batch = {
            "id": id,
            "nsamples": len(samples),
            "net_input": {
                "source": src_tokens,
            },
            "target": target,
        }
        return batch
    
