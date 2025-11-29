import torch
import numpy as np

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    """
    Simplified collate for fixed-size tensors.
    Now that all samples have exactly 36 points and same num_frames,
    we can just stack them directly!
    """
    # Check if all tensors have the same shape (they should!)
    shapes = [b.shape for b in batch]
    if len(set(shapes)) == 1:
        # All same shape - simple stack!
        return torch.stack(batch, dim=0)
    else:
        # Fallback to padding (videos with different frame counts)
        dims = batch[0].dim()
        max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
        size = (len(batch),) + tuple(max_size)
        canvas = batch[0].new_zeros(size=size)
        for i, b in enumerate(batch):
            sub_tensor = canvas[i]
            for d in range(dims):
                sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
            sub_tensor.add_(b)
        return canvas


def collate(batch):
    """
    Simplified collate function for fixed-size point grids.
    All samples now have exactly 36 points (6x6 grid) and same num_frames.
    No padding or truncation needed!
    """
    notnone_batches = [b for b in batch if b is not None]
    
    # Skip batch if all samples are None
    if len(notnone_batches) == 0:
        return None
    
    # Extract data - all samples have same shape now!
    databatch = [b['inp'] for b in notnone_batches]
    labelbatch = [b['target'] for b in notnone_batches]
    lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]  # All should be same (num_frames)

    # Stack tensors (much simpler now!)
    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)  # All True since all same length


    out_batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
             # "y_action_names": actionlabelbatchTensor}
    if 'clip_image' in notnone_batches[0]:
        clip_image_batch = [torch.as_tensor(b['clip_image']) for b in notnone_batches]
        out_batch.update({'clip_images': collate_tensors(clip_image_batch)})

    if 'clip_text' in notnone_batches[0]:
        textbatch = [b['clip_text'] for b in notnone_batches]
        out_batch.update({'clip_text': textbatch})

    if 'clip_path' in notnone_batches[0]:
        textbatch = [b['clip_path'] for b in notnone_batches]
        out_batch.update({'clip_path': textbatch})

    if 'all_categories' in notnone_batches[0]:
        textbatch = [b['all_categories'] for b in notnone_batches]
        out_batch.update({'all_categories': textbatch})

    return out_batch
