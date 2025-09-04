import torch
import torch.nn as nn
from models.layers.normalization import LayerNorm, ConditionalLayerNorm
from models.layers.attention import MultiHeadAttention
from models.layers.mlp import PositionwiseFeedForward
from modules.utils import clones


#############################
# Normal Transformer Blocks #
#############################
class SublayerConnection(nn.Module):
    
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, left=None, right=None):
        if left is None or right is None:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            left_modal, right_modal = x.split([left, right], dim=1)
            left_modal = left_modal + self.dropout(sublayer(self.norm(left_modal)))
            right_modal = right_modal + self.dropout(sublayer(self.norm(right_modal)))
            return torch.cat((left_modal, right_modal), dim=1)


class TransBlock(nn.Module):
	
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(TransBlock, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask, left=None, right=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class TransLayer(nn.Module):
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super(TransLayer, self).__init__()
        layer = TransBlock(
            d_model, 
            MultiHeadAttention(num_heads, d_model),
            PositionwiseFeedForward(d_model, d_ff, dropout),
            dropout
            )
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask, left=None, right=None):
        for layer in self.layers:
            x = layer(x, mask, left, right)
        return self.norm(x)


########################################
# Relational Memory Transformer Blocks #
########################################
class ConditionalSublayerConnection(nn.Module):
    
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class TransRMBlock(nn.Module):

    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(TransRMBlock, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.GELU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.GELU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)
        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)
        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class TransRMLayer(nn.Module):

    def __init__(self, num_layers, num_heads, d_model, d_ff, dropout, rm_num_slots, rm_d_model, rm_num_heads):
        super(TransRMLayer, self).__init__()
        layer = TransRMBlock(
            d_model,
            MultiHeadAttention(num_heads, d_model),
            MultiHeadAttention(num_heads, d_model),
            PositionwiseFeedForward(d_model, d_ff, dropout),
            dropout,
            rm_num_slots,
            rm_d_model
            )
        self.layers = clones(layer, num_layers)
        self.rm = RelationalMemory(rm_num_slots, rm_d_model, rm_num_heads)
        self.norm = LayerNorm(d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(x, memory)
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)













