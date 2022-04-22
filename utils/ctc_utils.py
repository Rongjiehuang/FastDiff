# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Di Wu)

import numpy as np
import torch
from textgrid import TextGrid, IntervalTier


def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label, 1)
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label


def forced_align(ctc_probs: torch.Tensor,
                 y: torch.Tensor,
                 blank_id=0) -> list:
    """ctc forced alignment.
    Args:
        torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
        torch.Tensor y: id sequence tensor 1d tensor (L)
        int blank_id: blank symbol index
    Returns:
        torch.Tensor: alignment result
    """
    y_insert_blank = insert_blank(y, blank_id)

    log_alpha = torch.zeros((ctc_probs.size(0), len(y_insert_blank)))
    log_alpha = log_alpha - float('inf')  # log of zero
    state_path = (torch.zeros(
        (ctc_probs.size(0), len(y_insert_blank)), dtype=torch.int16) - 1
                  )  # state path

    # init start state
    log_alpha[0, 0] = ctc_probs[0][y_insert_blank[0]]
    log_alpha[0, 1] = ctc_probs[0][y_insert_blank[1]]

    for t in range(1, ctc_probs.size(0)):
        for s in range(len(y_insert_blank)):
            if y_insert_blank[s] == blank_id or s < 2 or y_insert_blank[
                s] == y_insert_blank[s - 2]:
                candidates = torch.tensor(
                    [log_alpha[t - 1, s], log_alpha[t - 1, s - 1]])
                prev_state = [s, s - 1]
            else:
                candidates = torch.tensor([
                    log_alpha[t - 1, s],
                    log_alpha[t - 1, s - 1],
                    log_alpha[t - 1, s - 2],
                ])
                prev_state = [s, s - 1, s - 2]
            log_alpha[t, s] = torch.max(candidates) + ctc_probs[t][y_insert_blank[s]]
            state_path[t, s] = prev_state[torch.argmax(candidates)]

    state_seq = -1 * torch.ones((ctc_probs.size(0), 1), dtype=torch.int16)

    candidates = torch.tensor([
        log_alpha[-1, len(y_insert_blank) - 1],
        log_alpha[-1, len(y_insert_blank) - 2]
    ])
    prev_state = [len(y_insert_blank) - 1, len(y_insert_blank) - 2]
    state_seq[-1] = prev_state[torch.argmax(candidates)]
    for t in range(ctc_probs.size(0) - 2, -1, -1):
        state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

    output_alignment = []
    for t in range(0, ctc_probs.size(0)):
        output_alignment.append(y_insert_blank[state_seq[t, 0]])

    return output_alignment


def generator_textgrid(maxtime, lines, output):
    margin = 0.0001
    tg = TextGrid(maxTime=maxtime)
    linetier = IntervalTier(name="line", maxTime=maxtime)
    for l in lines:
        s, e, w = l.split()
        linetier.add(minTime=float(s) + margin, maxTime=float(e), mark=w)

    tg.append(linetier)
    print("successfully generator {}".format(output))
    tg.write(output)


def get_frames_timestamp(alignment):
    # convert alignment to a praat format, which is a doing phonetics
    # by computer and helps analyzing alignment
    timestamp = []
    # get frames level duration for each token
    start = 0
    end = 0
    while end < len(alignment):
        while end < len(alignment) and alignment[end] == 0:
            end += 1
        if end == len(alignment):
            timestamp[-1] += alignment[start:]
            break
        end += 1
        while end < len(alignment) and alignment[end - 1] == alignment[end]:
            end += 1
        timestamp.append(alignment[start:end])
        start = end
    return timestamp


def get_labformat(timestamp, frame_ms, token_dict):
    begin = 0
    labformat = []
    for idx, t in enumerate(timestamp):
        # time duration
        duration = len(t) * frame_ms
        if idx < len(timestamp) - 1:
            print("{:.2f} {:.2f} {}".format(begin, begin + duration,
                                            token_dict[t[-1]]))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, token_dict[t[-1]]))
        else:
            non_blank = 0
            for i in t:
                if i != 0:
                    token = i
                    break
            print("{:.2f} {:.2f} {}".format(begin, begin + duration, token_dict[token]))
            labformat.append("{:.2f} {:.2f} {}\n".format(
                begin, begin + duration, token_dict[token]))
        begin = begin + duration
    return labformat


def save_textgrid(alignment, token_dict, textgrid_path, frame_ms):
    timestamp = get_frames_timestamp(alignment)
    labformat = get_labformat(timestamp, frame_ms, token_dict)
    generator_textgrid(maxtime=(len(alignment) + 1) * frame_ms,
                       lines=labformat,
                       output=textgrid_path)
