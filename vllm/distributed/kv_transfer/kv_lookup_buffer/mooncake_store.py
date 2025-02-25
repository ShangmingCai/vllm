# SPDX-License-Identifier: Apache-2.0
"""
This file contains a new class `MooncakeStore` that allows developers to 
think of KV cache transfer operations as put new KV cache entries (`insert`) 
into a remote KVStore-based lookup buffer and querying existing KV caches
from this remote lookup buffer.
"""
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeStoreConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", 3355443200),
            local_buffer_size=config.get("local_buffer_size", 1073741824),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)


class MooncakeStore(KVLookupBufferBase):

    def __init__(
        self,
        config: VllmConfig,
    ):
        try:
            from mooncake_vllm_adaptor import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/mooncake-store-preview.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")

            self.store.setup(self.config.local_hostname,
                             self.config.metadata_server,
                             self.config.global_segment_size,
                             self.config.local_buffer_size,
                             self.config.protocol, self.config.device_name,
                             self.config.master_server_address)

        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        self.put_submit_thread = ThreadPoolExecutor(max_workers=1)
        self.get_submit_thread = ThreadPoolExecutor(max_workers=1)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor, hidden: torch.Tensor,
               store_keys_prefix: str, tp_rank: int) -> None:

        kvcache_to_sent = torch.stack((key, value), dim=0)
        # send kvcache with store keys from model_input
        store_keys = store_keys_prefix + "_" + str(tp_rank)
        self.put(store_keys, kvcache_to_sent)

        # call self.kv_store to put hidden_or_intermediate_states
        tmp_keys = store_keys_prefix + "_" + "hidden" + "_" + str(tp_rank)
        self.put(tmp_keys, hidden)

        # call self.kv_store to put hidden_or_intermediate_states
        tmp_keys = store_keys_prefix + "_" + "roi" + "_" + str(tp_rank)
        self.put(tmp_keys, roi)
        return

    def drop_select(self, input_tokens: Optional[torch.Tensor],
                    roi: Optional[torch.Tensor], load_keys_prefix: str,
                    tp_rank: int) -> List[Optional[torch.Tensor]]:
        load_keys = load_keys_prefix + "_" + str(tp_rank)
        remote_kv = self.get(load_keys)
        # call self.kv_store to put hidden_or_intermediate_states
        tmp_keys = load_keys_prefix + "_" + "hidden" + "_" + str(tp_rank)
        hidden = self.get(tmp_keys)

        tmp_keys = load_keys_prefix + "_" + "roi" + "_" + str(tp_rank)
        roi = self.get(tmp_keys)
        return [remote_kv, hidden, roi]

    def close(self):
        pass

    def put(
        self,
        key: str,
        value: Optional[torch.Tensor],
    ) -> None:
        # submit asynchronous put thread
        if value is not None:
            self.put_submit_thread.submit(self._put_impl, key, value)

    def get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        # submit asynchronous get thread
        value = self.get_submit_thread.submit(self._get_impl, key).result()
        return value

    def _put_impl(
        self,
        key: str,
        value: torch.Tensor,
    ) -> None:
        """Put KVCache to Mooncake Store"""
        value_bytes = pickle.dumps(value)
        try:
            self.store.put(key, value_bytes)
        except TypeError as err:
            raise TypeError("Mooncake Store Put Type Error.") from err

    def _get_impl(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """Get KVCache from Mooncake Store"""
        try:
            data = self.store.get(key)
            if data:
                return pickle.loads(data)
        except TypeError as err:
            raise TypeError("Mooncake Store Get Type Error.") from err

        return None
