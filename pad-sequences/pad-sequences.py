import numpy as np

def pad_sequences(sequences: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if len(sequences) == 0:
        return np.empty((0, 0), dtype=int)
    
    if max_len is None:
        max_len = 0

        # calculating max_len
        max_len = max(len(sequence) for sequence in sequences)

        
        for sequence in sequences:
            curr_seq_len = len(sequence)
            
            if curr_seq_len < max_len:
                padding = [pad_value] * (max_len - curr_seq_len)
                sequence.extend(padding)

                
    # max_len is specified as input parameter
    else:
        for sequence in sequences:
            curr_seq_len = len(sequence)
            
            if curr_seq_len < max_len:
                padding = [pad_value] * (max_len - curr_seq_len)
                sequence.extend(padding)

            elif curr_seq_len > max_len:
                sequence[:] = sequence[0:max_len] 
                # re-assignment vs in-place mutation
                # sequence = sequence[0:max_len] --> this only changes the local variable sequence
                # that iteration. It does not update the list stored inside the sequences list

    
    return np.array(sequences, dtype=int)

# As numpy arrays are fixed in length, if we would have directly worked with numpy arrays in above for loops,
# it would have assigned a new memory location for numpy arrays each time, making it slower for larger data
# hence, at last we convert the python list into numpy array (after all the pre-processing)