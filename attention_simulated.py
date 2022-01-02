import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils

MAX_SEQ_LEN = 4096

__all__ = [
    'BigbirdBlockSpareAttention_sim',
]


def get_single_block_row_attention(block_id,
                                   to_start_block_id,
                                   to_end_block_id,
                                   num_rand_blocks,
                                   window_block_left=1,
                                   window_block_right=1,
                                   global_block_left=1,
                                   global_block_right=1):
    """
    For a single row block get random row attention.
    Args:
        block_id: int. block id of row.
        to_start_block_id: int. random attention coloum start id.
        to_end_block_id: int. random attention coloum end id.
        num_rand_blocks: int. number of random blocks to be selected.
        window_block_left: int. number of blocks of window to left of a block.
        window_block_right: int. number of blocks of window to right of a block.
        global_block_left: int. Number of blocks globally used to the left.
        global_block_right: int. Number of blocks globally used to the right.
    Returns:
        row containing the random attention vector of size num_rand_blocks.
    """

    # list of to_blocks from which to choose random attention
    to_block_list = np.arange(
        to_start_block_id, to_end_block_id, dtype=np.int32)
    # permute the blocks
    perm_block = np.random.permutation(to_block_list)

    # illegal blocks for the current block id, using window
    illegal_blocks = list(
        range(block_id - window_block_left, block_id + window_block_right + 1))

    # Add blocks at the start and at the end
    illegal_blocks.extend(list(range(global_block_left)))
    illegal_blocks.extend(
        list(range(to_end_block_id - global_block_right, to_end_block_id)))

    # The second from_block cannot choose random attention on second last to_block
    if block_id == 1:
        illegal_blocks.append(to_end_block_id - 2)

    # The second last from_block cannot choose random attention on second to_block
    if block_id == to_end_block_id - 2:
        illegal_blocks.append(1)

    selected_random_blokcs = []

    for i in range(to_end_block_id - to_start_block_id):
        if perm_block[i] not in illegal_blocks:
            selected_random_blokcs.append(perm_block[i])
        if len(selected_random_blokcs) == num_rand_blocks:
            break
    return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_rand_mask_with_head(from_seq_length,
                                      to_seq_length,
                                      from_block_size,
                                      to_block_size,
                                      num_heads,
                                      plan_from_length,
                                      plan_num_rand_blocks,
                                      window_block_left=1,
                                      window_block_right=1,
                                      global_block_top=1,
                                      global_block_bottom=1,
                                      global_block_left=1,
                                      global_block_right=1):
    """Create adjacency list of random attention.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_heads: int. total number of heads.
    plan_from_length: list. plan from lenght where num_rand are choosen from.
    plan_num_rand_blocks: list. number of rand blocks within the plan.
    window_block_left: int. number of blocks of window to left of a block.
    window_block_right: int. number of blocks of window to right of a block.
    global_block_top: int. number of blocks at the top.
    global_block_bottom: int. number of blocks at the bottom.
    global_block_left: int. Number of blocks globally used to the left.
    global_block_right: int. Number of blocks globally used to the right.
  Returns:
    adjacency list of size num_head where each element is of size
    from_seq_length//from_block_size-2 by num_rand_blocks
  """
    assert from_seq_length // from_block_size == to_seq_length // to_block_size, \
        "Error the number of blocks needs to be same!"

    assert from_seq_length in plan_from_length, \
        "Error from sequence length not in plan!"

    # Total number of blocks in the mmask
    num_blocks = from_seq_length // from_block_size
    # Number of blocks per plan
    plan_block_length = np.array(plan_from_length) // from_block_size
    # till when to follow plan
    max_plan_idx = plan_from_length.index(from_seq_length)
    # Random Attention adjajency list
    rand_attn = [
        np.zeros(
            (num_blocks, np.sum(plan_num_rand_blocks[:max_plan_idx + 1])),
            dtype=np.int32) for i in range(num_heads)
    ]

    # We will go iteratively over the plan blocks and pick random number of
    # Attention blocks from the legally allowed blocks
    for plan_idx in range(max_plan_idx + 1):
        rnd_r_cnt = 0
        if plan_idx > 0:
            # set the row for all from_blocks starting from 0 to
            # plan_block_length[plan_idx-1]
            # column indx start fromm plan_block_length[plan_idx-1] and ends at
            # plan_block_length[plan_idx]
            if plan_num_rand_blocks[plan_idx] > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
                for blk_rw_idx in range(global_block_top,
                                        plan_block_length[plan_idx - 1]):
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][
                            blk_rw_idx, rnd_r_cnt:
                            curr_r_cnt] = get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx -
                                                                    1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right)

            for pl_id in range(plan_idx):
                if plan_num_rand_blocks[pl_id] == 0:
                    continue
                for blk_rw_idx in range(plan_block_length[plan_idx - 1],
                                        plan_block_length[plan_idx]):
                    rnd_r_cnt = 0
                    to_start_block_id = 0
                    if pl_id > 0:
                        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                        to_start_block_id = plan_block_length[pl_id - 1]
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[
                            h][blk_rw_idx, rnd_r_cnt:
                               curr_r_cnt] = get_single_block_row_attention(
                                   block_id=blk_rw_idx,
                                   to_start_block_id=to_start_block_id,
                                   to_end_block_id=plan_block_length[pl_id],
                                   num_rand_blocks=plan_num_rand_blocks[pl_id],
                                   window_block_left=window_block_left,
                                   window_block_right=window_block_right,
                                   global_block_left=global_block_left,
                                   global_block_right=global_block_right)

        if plan_num_rand_blocks[plan_idx] == 0:
            continue
        # print("Start from here")
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
        from_start_block_id = global_block_top
        to_start_block_id = 0
        if plan_idx > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
            from_start_block_id = plan_block_length[plan_idx - 1]
            to_start_block_id = plan_block_length[plan_idx - 1]

        for blk_rw_idx in range(from_start_block_id,
                                plan_block_length[plan_idx]):
            for h in range(num_heads):
                # print("head", h, "blk_rw_idx", blk_rw_idx)
                rand_attn[h][blk_rw_idx, rnd_r_cnt:
                             curr_r_cnt] = get_single_block_row_attention(
                                 block_id=blk_rw_idx,
                                 to_start_block_id=to_start_block_id,
                                 to_end_block_id=plan_block_length[plan_idx],
                                 num_rand_blocks=plan_num_rand_blocks[
                                     plan_idx],
                                 window_block_left=window_block_left,
                                 window_block_right=window_block_right,
                                 global_block_left=global_block_left,
                                 global_block_right=global_block_right)

    for nh in range(num_heads):
        rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks -
                                      global_block_bottom, :]
    return rand_attn


def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
    """Gives the plan of where to put random attention.
    Args:
        from_seq_length: int. length of from sequence.
        from_block_size: int. size of block in from sequence.
        num_rand_blocks: int. Number of random chunks per row.
        Returns:
        plan_from_length: ending location of from block
        plan_num_rand_blocks: number of random ending location for each block
    """
    # general plan
    plan_from_length = []
    plan_num_rand_blocks = []
    if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(
            int((2 * num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(0)
    elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks // 2)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
    else:
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks)

    return plan_from_length, plan_num_rand_blocks


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
    """Create adjacency list of random attention.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
      if positive then num_rand_blocks blocks choosen only upto last_idx.
  Returns:
    adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
  """
    assert from_seq_length // from_block_size == to_seq_length // to_block_size, \
        "Error the number of blocks needs to be same!"

    rand_attn = np.zeros(
        (from_seq_length // from_block_size - 2, num_rand_blocks),
        dtype=np.int32)
    middle_seq = np.arange(
        1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(
                    middle_seq[:start])[:r]
            elif (end + 1) == last:
                rand_attn[i - 1, :] = np.random.permutation(
                    middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start],
                                    middle_seq[end + 1:last])))[:r]
    return rand_attn


def full_bigbird_mask(from_seq_length,
                      to_seq_length,
                      from_block_size,
                      to_block_size,
                      num_rand_blocks,
                      rand_attn=None,
                      focus=1024):
    """Calculate BigBird attention pattern as a full dense matrix.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    rand_attn: adjajency matrix for random attention.
    focus: pick random mask within focus
  Returns:
    attention mask matrix of shape [from_seq_length, to_seq_length]
  """
    if rand_attn is None:
        rand_attn = bigbird_block_rand_mask(MAX_SEQ_LEN, MAX_SEQ_LEN,
                                            from_block_size, to_block_size,
                                            num_rand_blocks, focus)

    attn_mask = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.int32)
    for i in range(1, (from_seq_length // from_block_size) - 1):
        attn_mask[(i) * from_block_size:(i + 1) * from_block_size, (i - 1) *
                  to_block_size:(i + 2) * to_block_size] = 1
        for j in rand_attn[i - 1, :]:
            attn_mask[i * from_block_size:(i + 1) * from_block_size, j *
                      to_block_size:(j + 1) * to_block_size] = 1

    attn_mask[:from_block_size, :] = 1
    attn_mask[:, :to_block_size] = 1
    attn_mask[:, -to_block_size:] = 1
    attn_mask[-from_block_size:, :] = 1
    clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
    return np.array(clipped_attn_mask, dtype=bool)


def create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn,
                                 num_attention_heads, num_rand_blocks,
                                 batch_size, from_seq_length, from_block_size):
    """Create 3D attention mask from a 2D tensor mask.
  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
    rand_attn: [batch_size, num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks]
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    batch_size: int. Batch size for computation.
    from_seq_length: int. length of from sequence.
    from_block_size: int. size of block in from sequence.
  Returns:
    float Tensor of shape [batch_size, num_attention_heads,
                           from_seq_length//from_block_size-2,
                           from_block_size, num_rand_blocks*to_block_size].
  """
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = utils.torch_gather4d(to_blocked_mask, rand_attn)
    rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows,
                               num_rand_blocks * from_block_size)
    rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1],
                             rand_mask)
    return rand_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
    mask = torch.einsum("bf,bt->bft", from_mask, to_mask)

    # expand to create a slot for heads.
    mask = torch.unsqueeze(mask, 1)

    return mask


class BigbirdBlockSpareAttention_sim(nn.Module):
    def __init__(self,
                 num_attention_heads,
                 size_per_head,
                 num_rand_blocks,
                 from_block_size,
                 to_block_size,
                 seed=None):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head

        self.num_rand_blocks = num_rand_blocks
        self.from_block_size = from_block_size
        self.to_block_size = to_block_size

        self.seed = seed

    def convert_attn_list_to_mask(self, rand_attn, from_seq_length, to_seq_length):
        temp_mask = [
            full_bigbird_mask(  # pylint: disable=g-complex-comprehension
                from_seq_length, to_seq_length,
                self.from_block_size, self.to_block_size,
                self.num_rand_blocks,
                rand_attn=rand_attn[i])
            for i in range(self.num_attention_heads)
        ]
        temp_mask = np.stack(temp_mask, axis=0)
        temp_mask = np.array(temp_mask, dtype=bool)
        rand_block_mask = torch.from_numpy(temp_mask).bool()  # [N, F, T]
        return rand_block_mask.float()
    def original_full_attention(self,
                              query_layer,
                              key_layer,
                              value_layer,
                              masks,
                              training=None):
     """Full quadratic attention calculation.
 
     Args:
       query_layer: float Tensor of shape [batch_size, num_attention_heads,
         from_seq_length, size_per_head]
       key_layer: float Tensor of shape [batch_size, num_attention_heads,
         to_seq_length, size_per_head]
       value_layer: float Tensor of shape [batch_size, num_attention_heads,
         to_seq_length, size_per_head]
       masks: a list containing float32 Tensor representing attention_mask
         of shape [batch_size, from_seq_length, to_seq_length].
         The values should be 1 or 0. The attention scores will effectively be
         set to -infinity for any positions in the mask that are 0, and
         will be unchanged for positions that are 1.
       training: Boolean indicating whether the call is training or inference.
 
     Returns:
       float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
         size_per_head].
     """
     attention_mask = masks[0]
 
     # Directly take n^2 dot product between "query" and "key".
     attention_scores = torch.einsum("bnfh,bnth->bnft", query_layer, key_layer)
     attention_scores = torch.multiply(attention_scores,
                                    1.0 / np.sqrt(float(self.size_per_head)))
 
     if attention_mask is not None:
       # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
       # masked positions, this operation will create a tensor which is 0.0 for
       # positions we want to attend and -10000.0 for masked positions.
       adder = (1.0 - attention_mask) * -10000.0
 
       # Since we are adding it to the raw scores before the softmax, this is
       # effectively the same as removing these entirely.
       attention_scores += adder
 
     # Normalize the attention scores to probabilities.
     # `attention_probs` = [B, N, F, T]
     attention_probs = F.softmax(attention_scores, -1)
 
     # This is actually dropping out entire tokens to attend to, which might
     # seem a bit unusual, but is taken from the original Transformer paper.
     #attention_probs = self.attention_dropout(attention_probs, training=training)
 
     # `context_layer` = [B, F, N, H]
     context_layer = torch.einsum("bnft,bnth->bfnh", attention_probs, value_layer)
     return context_layer

    def forward(self,
                query_layer,
                key_layer,
                value_layer,
                batch_size,
                from_seq_length,
                to_seq_length,
                plan_from_length=None,
                plan_num_rand_blocks=None):
        """BigBird attention sparse calculation using blocks in linear time.

        Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.
        A pure function with a long argument list to allow easy use outside our
        framework.

        Args:
          query_layer: float Tensor of shape [batch_size, num_attention_heads,
            from_seq_length, size_per_head]
          key_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
          value_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
          band_mask: float32 Tensor of shape [batch_size, 1,
            from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
            The values should be 1 or 0. The attention scores will effectively be
            set to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
          from_mask: float32 Tensor of shape [batch_size, 1, from_seq_length, 1].
            The values should be 1 or 0. The attention scores will effectively be set
            to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
          to_mask: float32 Tensor of shape [batch_size, 1, 1, to_seq_length].
            The values should be 1 or 0. The attention scores will effectively be set
            to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
          from_blocked_mask: float32 Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            Same as from_mask, just reshaped.
          to_blocked_mask: float32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
            Same as to_mask, just reshaped.
          rand_attn: int32 Tensor of shape [num_attention_heads,
            from_seq_length//from_block_size-2, num_rand_blocks] specifying which
            blocks to attend to for each from sequence block (except 2 global ones).
          num_attention_heads: int. Number of attention heads.
          size_per_head: int. Size of each attention head.
          num_rand_blocks: int. Number of random chunks per row.
          from_seq_length: int. length of from sequence.
          to_seq_length: int. length of to sequence.
          from_block_size: int. size of block in from sequence.
          to_block_size: int. size of block in to sequence.

        Returns:
          float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
            size_per_head].
        """
        assert from_seq_length // self.from_block_size == to_seq_length // self.to_block_size


        # generate random attention and corresponding masks
        np.random.seed(self.seed)
        if from_seq_length in [1024, 3072, 4096]:  # old plans used in paper
            rand_attn = [
                bigbird_block_rand_mask(  # pylint: disable=g-complex-comprehension
                    MAX_SEQ_LEN,
                    MAX_SEQ_LEN,
                    self.from_block_size,
                    self.to_block_size,
                    self.num_rand_blocks,
                    last_idx=1024)[:(
                        from_seq_length // self.from_block_size - 2)]
                for _ in range(self.num_attention_heads)
            ]
        else:
            raise NotImplementedError

        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = torch.from_numpy(rand_attn).long()

        rand_block_mask = self.convert_attn_list_to_mask(rand_attn, from_seq_length, to_seq_length)
        rand_block_mask = torch.unsqueeze(rand_block_mask, 0)  # [1, N, F, T]

        attention_mask = rand_block_mask
        return self.original_full_attention(
            query_layer, key_layer, value_layer, [attention_mask]
            )
