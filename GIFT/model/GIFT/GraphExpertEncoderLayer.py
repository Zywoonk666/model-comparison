import torch
import torch.nn as nn
import math
from transformers.models.t5.modeling_t5 import T5LayerNorm


class GraphExpertAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.inner_dim = num_heads * self.head_dim

        # === 文本专属 QKV ===
        self.q_text = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_text = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_text = nn.Linear(d_model, self.inner_dim, bias=False)

        # === 图分子专属 QKV ===
        self.q_graph = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_graph = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_graph = nn.Linear(d_model, self.inner_dim, bias=False)

        # 共享的输出投影层
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, hidden_states, position_bias=None, graph_length=0):
        batch_size, seq_length = hidden_states.shape[:2]

        # 1. 动态路由：根据 graph_length 使用不同的权重矩阵
        if graph_length > 0:
            graph_states = hidden_states[:, :graph_length, :]
            text_states = hidden_states[:, graph_length:, :]

            q_g = self.q_graph(graph_states)
            k_g = self.k_graph(graph_states)
            v_g = self.v_graph(graph_states)

            q_t = self.q_text(text_states)
            k_t = self.k_text(text_states)
            v_t = self.v_text(text_states)

            # 在序列维度重新拼接
            q = torch.cat([q_g, q_t], dim=1)
            k = torch.cat([k_g, k_t], dim=1)
            v = torch.cat([v_g, v_t], dim=1)
        else:
            q = self.q_text(hidden_states)
            k = self.k_text(hidden_states)
            v = self.v_text(hidden_states)

        # 2. 形状转换以适应多头注意力计算
        def shape(tensor):
            return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape(q)
        k = shape(k)
        v = shape(v)

        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(2, 3))

        if position_bias is not None:
            scores += position_bias

        # 归一化并执行 Softmax
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, v)

        # 4. 形状复原与输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class GraphExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        # 文本专属 FFN
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model, bias=False)
        )
        # 图分子专属 FFN
        self.graph_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, hidden_states, graph_length=0):
        if graph_length > 0:
            graph_states = hidden_states[:, :graph_length, :]
            text_states = hidden_states[:, graph_length:, :]

            graph_out = self.graph_ffn(graph_states)
            text_out = self.text_ffn(text_states)

            return torch.cat([graph_out, text_out], dim=1)
        else:
            return self.text_ffn(hidden_states)


class GraphExpertEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.attention = GraphExpertAttention(d_model, num_heads)
        self.layer_norm_1 = T5LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.ffn = GraphExpertFFN(d_model, d_ff, dropout_rate)
        self.layer_norm_2 = T5LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states,
        position_bias=None,
        graph_length=0,
        use_cache=False,
        **kwargs
    ):
        # 1. Self-Attention Block (Pre-LayerNorm 机制)
        normed_hidden_states = self.layer_norm_1(hidden_states)
        attn_output = self.attention(
            normed_hidden_states,
            position_bias=position_bias,
            graph_length=graph_length
        )
        hidden_states = hidden_states + self.dropout_1(attn_output)

        # 2. Feed-Forward Block (Pre-LayerNorm 机制)
        normed_hidden_states = self.layer_norm_2(hidden_states)
        ffn_output = self.ffn(normed_hidden_states, graph_length=graph_length)
        hidden_states = hidden_states + self.dropout_2(ffn_output)

        # 3. 构造返回值元组，对齐 GIMLETEncoderStack.py 中的 layer_outputs 解包期望
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (None,) # 模拟 past_key_value
        
        # 返回结构: (hidden_states, past_key_value, position_bias, attentions, cross_attentions, cross_attention_weights)
        return outputs + (position_bias, None, None, None)