import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.t5.modeling_t5 import T5Stack
from model.GIFT.gnn_model import GNN
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from .GraphExpertEncoderLayer import GraphExpertEncoderLayer
from model.graphormer.modules.graphormer_graph_encoder import GraphEncoder

logger = logging.get_logger(__name__)


def compute_optimal_transport_plan(hidden_states, hidden_states_graph_mapped, epsilon=0.1, n_iters=100):
    """
    Compute optimal transport plan T.
    
    Args:
        hidden_states: Text features H_t, shape [Batch, Text_Len, Dim]
        hidden_states_graph_mapped: Graph features H_m, shape [Batch, Graph_Len, Dim]
        epsilon: Entropy regularization coefficient (controls smoothness, smaller is harder to converge)
        n_iters: Sinkhorn iteration times
        
    Returns:
        T: Optimal transport assignment plan matrix, shape [Batch, Text_Len, Graph_Len]
    """
    B, N_t, Dim = hidden_states.shape
    _, N_m, _ = hidden_states_graph_mapped.shape

    # ==========================================
    # Step 1: Compute local transport cost matrix C
    # ==========================================
    # L2 normalize features on the last dimension
    Ht_norm = F.normalize(hidden_states, p=2, dim=-1)           # [B, N_t, Dim]
    Hm_norm = F.normalize(hidden_states_graph_mapped, p=2, dim=-1) # [B, N_m, Dim]

    # Compute cosine similarity via matrix multiplication: cos(h^t_i, h^m_j)
    # Result shape: [B, N_t, N_m]
    cos_sim = torch.bmm(Ht_norm, Hm_norm.transpose(1, 2))

    # Compute cost matrix: C_{ij} = 1 - cos(h^t_i, h^m_j)
    C = 1.0 - cos_sim

    mu = torch.empty(B, N_t, dtype=C.dtype, device=C.device).fill_(1.0 / N_t) # Text side distribution
    nu = torch.empty(B, N_m, dtype=C.dtype, device=C.device).fill_(1.0 / N_m) # Graph side distribution

    # ==========================================
    # Step 2: Use Sinkhorn algorithm to solve for T
    # ==========================================
    # Compute Kernel matrix: exp(-C / epsilon)
    K = torch.exp(-C / epsilon)

    # Initialize dual variables u and v
    u = torch.empty(B, N_t, dtype=C.dtype, device=C.device).fill_(1.0 / N_t)
    v = torch.empty(B, N_m, dtype=C.dtype, device=C.device).fill_(1.0 / N_m)

    # Sinkhorn iterations to find optimal solution
    with torch.no_grad():
        for _ in range(n_iters):
            # Update u
            u = mu / torch.bmm(K, v.unsqueeze(2)).squeeze(2)
            # Update v
            v = nu / torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)

    # Compute final optimal transport assignment matrix T
    # T_{ij} = u_i * K_{ij} * v_j
    T = u.unsqueeze(2) * K * v.unsqueeze(1)

    return T, C


class ResMLPBlock(nn.Module):
    """
    MLP module with residual connection, adding Dropout to prevent overfitting.
    """
    def __init__(self, d_model, activation='relu', dropout=0.4): # Add dropout parameter
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        
        if activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Unsupported activation. Please choose 'gelu' or 'relu'.")
            
        self.dropout = nn.Dropout(dropout) # Initialize Dropout
        self.fc2 = nn.Linear(d_model, d_model)
        
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.eye_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out) # Add Dropout after activation function
        out = self.fc2(out)
        return out + residual


class DeepGraphProjector(nn.Module):
    """
    Deep Projector stacked by multiple residual modules.
    """
    def __init__(self, d_model, num_blocks=2, activation='relu', dropout=0.4):
        """
        :param d_model: Feature dimension of input and output
        :param num_blocks: Number of residual blocks (2 blocks equal 4 linear layers)
        :param activation: Type of activation function ('gelu' 或 'relu')
        """
        super().__init__()
        # Stack multiple modules
        self.layers = nn.Sequential(
            *[ResMLPBlock(d_model, activation=activation, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.layers(x)


class GIMLETEncoderStack(T5Stack):
    def __init__(self, config, graph_args,embed_tokens=None):
        super().__init__(config,embed_tokens=embed_tokens)
        #print("GIMLETEncoderStack_init's graph_args:", graph_args)
        #print("GIMLETEncoderStack_init's projector_dropout:", getattr(graph_args, 'projector_dropout', 0.1))
        #print("GIMLETEncoderStack_init's use_ot_alignment:", getattr(graph_args, 'use_ot_alignment', False))
        if graph_args.encoder_embed_dim>0 and graph_args.encoder_embed_dim!=self.config.d_model:
            raise ValueError('Warning! Inconsistent graph encoder_embed_dim and transformer d_model for Unimodel')
        graph_args.encoder_embed_dim=self.config.d_model
        if graph_args.encoder_attention_heads>0 and graph_args.encoder_attention_heads!=self.config.num_heads:
            raise ValueError('Warning! Inconsistent graph encoder_attention_heads and transformer num_heads for Unimodel')
        graph_args.encoder_attention_heads=self.config.num_heads
        if graph_args.encoder_layers>0 and graph_args.encoder_layers!=self.config.num_layers:
            raise ValueError('Warning! Inconsistent graph encoder_layers and transformer num_layers for Unimodel')
        graph_args.encoder_layers = self.config.num_layers

        self.graph_encoder = GraphEncoder(
        num_atoms=graph_args.num_atoms,
        num_in_degree=graph_args.num_in_degree,
        num_out_degree=graph_args.num_out_degree,
        num_edges=graph_args.num_edges,
        num_spatial=graph_args.num_spatial,
        num_edge_dis=graph_args.num_edge_dis,
        edge_type=graph_args.edge_type,
        multi_hop_max_dist=graph_args.multi_hop_max_dist,
        # >
        num_encoder_layers=graph_args.encoder_layers,
        embedding_dim=graph_args.encoder_embed_dim,
        num_attention_heads=graph_args.encoder_attention_heads,
        dropout=graph_args.dropout,
        encoder_normalize_before=graph_args.encoder_normalize_before,
        apply_graphormer_init=graph_args.apply_graphormer_init,)

        self.graph_encoder.args=graph_args
        self.position_embedding_graph=nn.Embedding(1, self.block[1].layer[0].SelfAttention.n_heads)
        self.t5_relative_attention_bias = self.block[0].layer[0].SelfAttention.relative_attention_bias
        self.t5_compute_bias = self.block[0].layer[0].SelfAttention.compute_bias

        self.use_ot_alignment = getattr(graph_args, 'use_ot_alignment', False)

        # Disable progressive decoupling, reduce mask impact
        self.use_progressive_mask = getattr(graph_args, "use_progressive_mask", False)

        # Keep the original T5 blocks during from_pretrained() so official
        # weights can load first. get_model() converts them after loading.
        self.requested_use_graph_expert = getattr(self.graph_encoder.args, "use_graph_expert", True)
        self.use_graph_expert = False
    
        self.use_projector = getattr(graph_args, 'use_projector', True)
        self.projector_dropout = getattr(graph_args, 'projector_dropout', 0.1)
        num_res_blocks = getattr(graph_args, 'projector_blocks', 2)
        act_fn = getattr(graph_args, 'projector_act', 'relu')
        self.graph_projector = DeepGraphProjector(
            d_model=self.config.d_model, 
            num_blocks=num_res_blocks, 
            activation=act_fn,
            dropout=self.projector_dropout
        )

    def convert_to_graph_experts(self):
        """Replace T5 encoder blocks with graph experts after copying weights."""
        if self.use_graph_expert:
            return

        for i, source_block in enumerate(self.block):
            source_attention = source_block.layer[0].SelfAttention
            source_ffn = source_block.layer[1].DenseReluDense
            expert_block = GraphExpertEncoderLayer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                d_ff=self.config.d_ff,
                dropout_rate=self.config.dropout_rate,
            ).to(
                device=source_attention.q.weight.device,
                dtype=source_attention.q.weight.dtype,
            )

            with torch.no_grad():
                for target in (expert_block.attention.q_text, expert_block.attention.q_graph):
                    target.weight.copy_(source_attention.q.weight)
                for target in (expert_block.attention.k_text, expert_block.attention.k_graph):
                    target.weight.copy_(source_attention.k.weight)
                for target in (expert_block.attention.v_text, expert_block.attention.v_graph):
                    target.weight.copy_(source_attention.v.weight)
                expert_block.attention.o_proj.weight.copy_(source_attention.o.weight)
                expert_block.layer_norm_1.weight.copy_(source_block.layer[0].layer_norm.weight)

                if hasattr(source_ffn, "wi"):
                    source_wi = source_ffn.wi
                elif hasattr(source_ffn, "wi_0"):
                    source_wi = source_ffn.wi_0
                else:
                    raise AttributeError("Unsupported T5 FFN: missing wi/wi_0 weight")
                for target in (expert_block.ffn.text_ffn[0], expert_block.ffn.graph_ffn[0]):
                    target.weight.copy_(source_wi.weight)
                for target in (expert_block.ffn.text_ffn[3], expert_block.ffn.graph_ffn[3]):
                    target.weight.copy_(source_ffn.wo.weight)
                expert_block.layer_norm_2.weight.copy_(source_block.layer[1].layer_norm.weight)

            self.block[i] = expert_block

        self.use_graph_expert = True
        self.graph_encoder.args.use_graph_expert = True
    
    
    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.mask_ratio = getattr(self.graph_encoder.args, "dynamic_mask_ratio", 1.0)
        graph=graph.to(input_ids.device)
        attention_mask_graph=graph['attn_bias'][:,0,:]

        hidden_state_graph, attn_bias_graph, padding_mask_graph = self.graph_encoder(graph)
        graph_encoder_raw = hidden_state_graph.clone()
        if self.use_projector:
            g_b_feats = hidden_state_graph.detach().cpu().reshape(-1, hidden_state_graph.shape[-1]).numpy()
            hidden_state_graph = self.graph_projector(hidden_state_graph)

        attention_bias_graph_unimodel = attn_bias_graph.masked_fill(
            padding_mask_graph.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"))


        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        # construct position_bias for T5, where graph tokens are all treated as a special distance to each text token
        #position_bias=self.block[0].layer[0].SelfAttention.compute_bias(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device)
        position_bias=self.t5_compute_bias(hidden_states.shape[1], hidden_states.shape[1], device=hidden_states.device)
        position_embedding_graph_spaned=self.position_embedding_graph(
            torch.zeros(1,hidden_states.shape[1],hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
            .transpose(3,1)
        position_embedding_graph_spaned_2=self.position_embedding_graph(
            torch.zeros(1,hidden_state_graph.shape[0],hidden_states.shape[1]+hidden_state_graph.shape[0]).long().to(self.position_embedding_graph.weight.device))\
            .transpose(3,1)
        position_bias_merged=torch.cat([position_embedding_graph_spaned,position_bias],2)
        position_bias_merged = torch.cat([position_embedding_graph_spaned_2, position_bias_merged], 3)

        
        position_bias_merged = position_bias_merged.repeat(attention_bias_graph_unimodel.shape[0], 1, 1, 1) #attention_bias_graph_unimodel.shape[0] is batch size
        position_bias_merged[:, :, 0:attention_bias_graph_unimodel.shape[2],
                            0:attention_bias_graph_unimodel.shape[3]] = attention_bias_graph_unimodel
        
        # Unmasked
        position_bias_unmasked = position_bias_merged.clone()
        # Strictly masked
        position_bias_masked = position_bias_merged.clone()

        if self.graph_encoder.args.maskt2g:
            mask_value = float("-inf")
            target_region = position_bias_masked[:, :, 0:attention_bias_graph_unimodel.shape[2], attention_bias_graph_unimodel.shape[3]:]
            
            if self.training and 0.0 < self.mask_ratio < 1.0:
                # Training phase: Generate random mask matrix according to probability mask_ratio.
                rand_mask = torch.rand(target_region.shape, device=target_region.device) < self.mask_ratio
                position_bias_masked[:, :, 0:attention_bias_graph_unimodel.shape[2], attention_bias_graph_unimodel.shape[3]:] = \
                    torch.where(rand_mask, torch.tensor(mask_value, device=target_region.device, dtype=target_region.dtype), target_region)
            elif self.mask_ratio >= 1.0 or not self.training:
                # If ratio>=1.0 or in test mode, 100% execute strict cross-modal mask
                position_bias_masked[:, :, 0:attention_bias_graph_unimodel.shape[2], attention_bias_graph_unimodel.shape[3]:] = mask_value

        # Assuming 12 layers in total, e.g. keep first 8 layers masked, and unmask last 4 layers
        total_layers = len(self.block)
        relax_layer_idx = int(total_layers * 0.8) if self.use_progressive_mask else int(total_layers)


        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            current_position_bias = position_bias_masked if i < relax_layer_idx else position_bias_unmasked
            #Because hidden_state_graph size is length,batch,embedding_dim, need transpose before input to T5
            # transpose to be consistant with the non-unimodel
            # hidden_state_graph=hidden_state_graph.transpose(0,1)
            
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states_graph_mapped = hidden_state_graph.transpose(0, 1)

            if self.use_ot_alignment:
                #text_weights = self.text_weight_proj(hidden_states).squeeze(-1)
                #mu = torch.softmax(text_weights, dim=-1)

                #graph_weights = self.graph_weight_proj(hidden_states_graph_mapped).squeeze(-1)
                #nu = torch.softmax(graph_weights, dim=-1)

                T, C = compute_optimal_transport_plan(hidden_states, hidden_states_graph_mapped)
                multi_token_reps = torch.bmm(T.transpose(1, 2), hidden_states)
                enhanced_graph_states = hidden_states_graph_mapped + multi_token_reps

                

                hidden_states_merged = torch.cat([enhanced_graph_states, hidden_states], dim=1)

                #T2, C2 = compute_optimal_transport_plan(hidden_states_graph_mapped, hidden_states)
                #multi_token_reps2 = torch.bmm(T2.transpose(1, 2), hidden_states_graph_mapped)
                #enhanced_text_states = hidden_states + multi_token_reps2
                #hidden_states_merged = torch.cat([enhanced_graph_states, enhanced_text_states], dim=1)                
            else:
                hidden_states_merged = torch.cat([hidden_states_graph_mapped, hidden_states], 1)
                
            attention_mask_merged = torch.cat([attention_mask_graph.unsqueeze(1).unsqueeze(1), extended_attention_mask], 3)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module, g_len):
                    def custom_forward(*inputs):
                        if getattr(self, "use_graph_expert", False):
                            return tuple(module(*inputs, use_cache=use_cache, output_attentions=output_attentions, graph_length=g_len))
                        else:
                            return tuple(module(*inputs, use_cache=use_cache, output_attentions=output_attentions))
                    return custom_forward


                hidden_states_graph_mapped = hidden_state_graph.transpose(0, 1)
                attention_mask_merged = torch.cat([attention_mask_graph.unsqueeze(1).unsqueeze(1), extended_attention_mask], 3)
                
                layer_outputs = checkpoint(
                    create_custom_forward(layer_module, hidden_state_graph.shape[0]), # Pass graph_length
                    hidden_states_merged,
                    current_position_bias + attention_mask_merged,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value
                )
            else:
                if getattr(self, "use_graph_expert", False):
                    layer_outputs = layer_module(
                        hidden_states_merged,
                        position_bias=current_position_bias + attention_mask_merged,
                        graph_length=hidden_state_graph.shape[0], # Graph expert layer specific parameters
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states_merged,
                        position_bias=current_position_bias + attention_mask_merged,
                        
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

            # layer_outputs is a tuple with:

            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states_merged, present_key_value_state = layer_outputs[:2]

            hidden_states = hidden_states_merged[:, hidden_states_graph_mapped.shape[1]:, :]
            # transpose again for consistance
            hidden_state_graph=(hidden_states_merged[:, 0:hidden_states_graph_mapped.shape[1], :]).transpose(0, 1)

            assert hidden_states.shape[1] == extended_attention_mask.shape[3]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))


        hidden_states = self.final_layer_norm(hidden_states)

        #for debug
        # hidden_states = self.final_layer_norm(hidden_state_graph.transpose(0,1))

        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def forward_graph_only(
        self,
        graph=None,
        # input_ids=None,
        # attention_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        graph=graph.to(self.device)
        attention_mask_graph=graph['attn_bias'][:,0,:]

        hidden_state_graph, attn_bias_graph, padding_mask_graph = self.graph_encoder(graph)
        if self.use_projector:
            hidden_state_graph = self.graph_projector(hidden_state_graph)

        attention_bias_graph_unimodel = attn_bias_graph.masked_fill(
            padding_mask_graph.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"))


        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"


        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)


        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None



        position_bias_merged=attention_bias_graph_unimodel

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)

                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states_graph_mapped = hidden_state_graph.transpose(0, 1)
            hidden_states_merged = hidden_states_graph_mapped
            attention_mask_merged = attention_mask_graph.unsqueeze(1).unsqueeze(1)
            layer_position_bias = position_bias_merged + attention_mask_merged

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module, g_len):
                    def custom_forward(*inputs):
                        if getattr(self, "use_graph_expert", False):
                            return tuple(module(*inputs, use_cache=use_cache, output_attentions=output_attentions, graph_length=g_len)) 
                        else:
                            return tuple(module(*inputs, use_cache=use_cache, output_attentions=output_attentions))
                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module, hidden_state_graph.shape[0]),
                    hidden_states_merged,
                    layer_position_bias, 
                    encoder_decoder_position_bias,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:

                hidden_states_graph_mapped = hidden_state_graph.transpose(0, 1)
                hidden_states_merged = hidden_states_graph_mapped
                attention_mask_merged = attention_mask_graph.unsqueeze(1).unsqueeze(1)

                layer_outputs = layer_module(
                    hidden_states_merged,
                    # attention_mask=attention_mask_merged,
                    position_bias=position_bias_merged + attention_mask_merged,
                    graph_length=hidden_states_merged.shape[1],
                    # encoder_hidden_states=encoder_hidden_states,
                    # encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    # layer_head_mask=layer_head_mask,
                    # cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states_merged, present_key_value_state = layer_outputs[:2]

            hidden_states = hidden_states_merged
            # transpose again for consistance
            hidden_state_graph=hidden_states_merged.transpose(0, 1)

            # assert hidden_states.shape[1] == extended_attention_mask.shape[3]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            # if self.is_decoder and encoder_hidden_states is not None:
            #     encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))



        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    attention_mask_merged
            )
        return dict(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask_merged=attention_mask_merged
        )


class GinT5EncoderStack(T5Stack):
    def __init__(self, config,graph_args, embed_tokens=None):
        super().__init__(config)

        # The Gin in this model only use base feature, i.e. the first two rows of x, and the first two rows of edge_feat, even if rich feature is given

        self.graph_encoder = GNN(**vars(graph_args))
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_args.emb_dim, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_size, config.hidden_size)
        )


    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        if inputs_embeds is None:
            input_embeds_new = self.embed_tokens(input_ids)
        else:
            input_embeds_new = inputs_embeds
        device = input_ids.device
        B, _ = input_ids.shape

        # if self.has_graph:

        graph_rep = self.graph_encoder(graph)
        graph_rep = self.graph_encoder.pool(graph_rep, graph['batch'])

        graph_rep = self.graph_projector(graph_rep)
        # graph_rep = torch.nn.functional.normalize(graph_rep, dim=1)
        input_embeds_new = torch.cat([graph_rep.unsqueeze(1), input_embeds_new[:, :-1, :]], dim=1)
        attention_mask = torch.cat([torch.ones(B, 1).to(device), attention_mask[:, :-1]], dim=1)

        input_ids=None
        inputs_embeds=input_embeds_new


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )



GraphT5EncoderStack_dict={'gint5':GinT5EncoderStack,
                       'gimlet':GIMLETEncoderStack}

