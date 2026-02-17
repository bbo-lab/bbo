import numpy as np

def segment(mask):
    mask = np.asarray(mask, dtype=bool)

    # Pad with False at both ends to catch edge segments
    padded = np.pad(mask.astype(int), (1, 1), mode='constant')

    # Find changes (0→1 = start, 1→0 = end)
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return np.column_stack((starts, ends))
