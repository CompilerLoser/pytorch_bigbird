import pdb

from attention import BigbirdBlockSpareAttention
from attention_simulated import BigbirdBlockSpareAttention_sim
import torch
import time
import os

batch_size = 16

num_attention_heads = 4
size_per_head = 512
num_rand_blocks = 3
from_seq_length = 4096  
to_seq_length = 4096
from_block_size = 32
to_block_size = 32

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


query_layer = torch.rand(batch_size, num_attention_heads, from_seq_length,
                         size_per_head).half().cuda()
key_layer = torch.rand(batch_size, num_attention_heads, to_seq_length,
                       size_per_head).half().cuda()
value_layer = torch.rand(batch_size, num_attention_heads, to_seq_length,
                         size_per_head).half().cuda()
# The values should be 1 or 0. The attention scores will effectively be
# set to -infinity for any positions in the mask that are 0, and will be
# unchanged for positions that are 1.

band_mask = torch.rand(batch_size, 1, from_seq_length // from_block_size - 4,
                       from_block_size, 3 * to_block_size).cuda()
from_mask = torch.rand(batch_size, 1, from_seq_length, 1).cuda()
to_mask = torch.rand(batch_size, 1, 1, to_seq_length).cuda()
from_blocked_mask = torch.rand(batch_size, from_seq_length // from_block_size,
                               from_block_size).cuda()
to_blocked_mask = torch.rand(batch_size, to_seq_length // to_block_size,
                             to_block_size).cuda()
rand_attn = torch.rand(num_attention_heads,
                       from_seq_length // from_block_size - 2, num_rand_blocks).cuda()

start = time.perf_counter()
attn = BigbirdBlockSpareAttention(
    num_attention_heads=num_attention_heads,
    num_rand_blocks=num_rand_blocks,
    size_per_head=size_per_head,
    from_block_size=from_block_size,
    to_block_size=to_block_size).cuda()

res = attn(query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, batch_size, from_seq_length, to_seq_length)
'''
attn_sim = BigbirdBlockSpareAttention_sim(
    num_attention_heads=num_attention_heads,
    num_rand_blocks=num_rand_blocks,
    size_per_head=size_per_head,
    from_block_size=from_block_size,
    to_block_size=to_block_size).cuda()
res = attn_sim(query_layer, key_layer, value_layer, batch_size, from_seq_length, to_seq_length)
'''
end = time.perf_counter()

print(batch_size*num_attention_heads*from_seq_length/(end - start)/1000)
