from __future__ import annotations

"""
Support attention backend for flashMLA.

Current initial integration of FlashMLA shows normal accuracy, but performance is slightly lacking.
#TODO
Support FlashMLA decode with cudagraph
Enable speculative sampling in FlashMLA
Integrate FA3 prefill
"""


from typing import TYPE_CHECKING, Optional, Union

import torch
import triton
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput


# FlashMLA only supports pagesize=64
PAGE_SIZE = 64


class FlashMLABackend(FlashInferMLAAttnBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        
        self.forward_metadata = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            bs = forward_batch.batch_size
            max_seqlen_pad = triton.cdiv(forward_batch.seq_lens.max().item(), PAGE_SIZE)
            flashmla_index = torch.full(
                (bs, max_seqlen_pad), -1, dtype=torch.int32, device=forward_batch.req_pool_indices.device
            )
            create_flashmla_kv_indices_triton[(bs,)](
                self.indices_updater_decode.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                None,
                flashmla_index,
                self.indices_updater_decode.req_to_token.size(1),
                flashmla_index.size(1),
                max_seqlen_pad,
            )

            mla_metadata, mla_splits = get_mla_metadata(
                forward_batch.seq_lens.to(torch.int32),
                1 * self.num_q_heads // self.num_kv_heads,
                self.num_kv_heads,
            )
        else:
            flashmla_index = None
            mla_metadata = None
            mla_splits = None

        self.forward_metadata = (
            flashmla_index,
            mla_metadata,
            mla_splits,
        )


    def init_cuda_graph_state(
        self, max_bs: int, kv_indices_buf: Optional[torch.Tensor] = None
    ):
        # kv_indices equal to flashmla_index
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.full(
                (max_bs, self.max_context_len,),
                -1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        # mla_metadata, mla_splits = get_mla_metadata(
        #     seq_lens.to(torch.int32),
        #     1 * self.num_q_heads // self.num_kv_heads,
        #     self.num_kv_heads,
        # )
        mla_splits = None
        mla_metadata = None
        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        self.cuda_graph_mla_metadata = mla_splits
        self.cuda_graph_mla_splits = mla_metadata

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                print("capture cuda graph-------------------")
                flashmla_index = self.cuda_graph_kv_indices
                # max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)
                # flashmla_index = torch.full(
                #     (bs, max_seqlen_pad), -1, dtype=torch.int32, device=req_pool_indices.device
                # )
                # self.cuda_graph_kv_indices = flashmla_index
                max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)
                create_flashmla_kv_indices_triton[(bs,)](
                    self.indices_updater_decode.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    None,
                    flashmla_index,
                    self.indices_updater_decode.req_to_token.size(1),
                    flashmla_index.size(1),
                    max_seqlen_pad,
                )
            else:
                flashmla_index = spec_info.kv_indices

            # mla_metadata = self.cuda_graph_mla_metadata
            # mla_splits = self.cuda_graph_mla_splits
            self.cuda_graph_mla_metadata, self.cuda_graph_mla_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                1 * self.num_q_heads // self.num_kv_heads,
                self.num_kv_heads,
            )
            mla_metadata = self.cuda_graph_mla_metadata
            mla_splits = self.cuda_graph_mla_splits

        elif forward_mode.is_target_verify():
            # TODO: Add support for nextn
            pass
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

        self.forward_metadata = (
            flashmla_index,
            mla_metadata,
            mla_splits,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                print("replay cuda graph-------------------")
                # flashmla_index = self.cuda_graph_kv_indices
                max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)
                flashmla_index = torch.full(
                    (bs, max_seqlen_pad), -1, dtype=torch.int32, device=req_pool_indices.device
                )
                create_flashmla_kv_indices_triton[(bs,)](
                    self.indices_updater_decode.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    None,
                    flashmla_index,
                    self.indices_updater_decode.req_to_token.size(1),
                    flashmla_index.size(1),
                    max_seqlen_pad,
                )
            else:
                flashmla_index[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
        elif forward_mode.is_target_verify():
            # TODO: Add support for nextn
            pass
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")       
    
    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        bs = forward_batch.batch_size

        # max_seqlen_pad = triton.cdiv(forward_batch.seq_lens.max().item(), PAGE_SIZE)
        # flashmla_index = torch.full(
        #     (bs, max_seqlen_pad), -1, dtype=torch.int32, device=q.device
        # )
        # create_flashmla_kv_indices_triton[(bs,)](
        #     self.indices_updater_decode.req_to_token,
        #     forward_batch.req_pool_indices,
        #     forward_batch.seq_lens,
        #     None,
        #     flashmla_index,
        #     self.indices_updater_decode.req_to_token.size(1),
        #     flashmla_index.size(1),
        #     max_seqlen_pad,
        # )

        # mla_metadata, mla_splits = get_mla_metadata(
        #     forward_batch.seq_lens.to(torch.int32),
        #     1 * self.num_q_heads // self.num_kv_heads,
        #     self.num_kv_heads,
        # )

        flashmla_index, mla_metadata, mla_splits = self.forward_metadata

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)

        o, _ = flash_mla_with_kvcache(
            q=reshape_q,
            k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
            block_table=flashmla_index,
            cache_seqlens=forward_batch.seq_lens.to(torch.int32),
            head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
            tile_scheduler_metadata=mla_metadata,
            num_splits=mla_splits,
            softmax_scale=layer.scaling,
            causal=False,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
