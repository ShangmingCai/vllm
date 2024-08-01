from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase

logger = init_logger(__name__)


class MultiProposersWorker(ProposerWorkerBase, LoraNotSupportedWorkerBase):
    def __init__(self, *args, **kwargs):
        self.vocab_size = kwargs["model_config"].get_vocab_size()
        self._workers = kwargs.pop('worker_list', {})

        draft_parallel_config: ParallelConfig = kwargs['parallel_config']
        draft_tp = draft_parallel_config.tensor_parallel_size
        
        # TP>1 is not supported currently because DraftModelRunner does
        # not support TP>1.
        # TODO: Remove this when TP>1 is supported and #5814 is fixed.
        if draft_tp != 1:
            raise ValueError(
                f"speculative_draft_tensor_parallel_size cannot be "
                f"other value than 1 when using MultiProposersWorker. "
                f"Got {draft_tp} instead.")


    def init_device(self) -> None:
        for worker in self._workers.values():
            worker.init_device()

    def load_model(self) -> None:
        for worker in self._workers.values():
            worker.load_model()

    def set_include_gpu_probs_tensor(self) -> None:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                worker.set_include_gpu_probs_tensor()

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[Optional[List[Optional[SamplerOutput]]], bool]:
        """No need to implement sampler_output for MultiProposersWorker,
        as the optional proposers of MultiProposersWorker will use their
        own Top1Proposers to call their sampler_output functions.
        """
        raise NotImplementedError
    
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. Ideally, we
        recieve and process all sequences in the same batch with one specified
        proposer. However, if multiple proposers are specified, we currently
        use the proposer with the lowest proposal latency for the whole batch.

        If we use different proposers for different sequences in the same
        batch, all proposers will need to wait for the slowest proposer to
        finish on each batch for further scoring. It means those proposers
        with lower acceptance rates but faster speed, like Ngram, will be
        dragged down by the slowest proposer for each step when there remain
        more steps for them to complete. Therefore, a better strategy is to
        use the fastest proposer adaptively among all specified proposers for
        the current batch.
        """
        chosen_proposer = self._get_proposer_for_this_step(
            execute_model_req,
            schedule_policy="proposal_latency")
        
        # print("chosen_proposer:", chosen_proposer)

        return self._workers[chosen_proposer].get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform speculative decoding on the input batch. This procedure
        will respect the SpecDecodeParams of each sequence.
        """

        # To perform KV operations, the 'non_driver_ranks' of SpecDecodeWorker
        # might call this function with execute_model_req set to None for
        # many times.
        if execute_model_req is None:
            return []
        
        # Curently, if one seq_group require to perform execute_model through
        # MultiStepWorker, all seq_groups in the same batch need to perform
        # execute_model together. We have not found a good way to avoid this.
        proposer: str = '[ngram]'
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        valid_proposers = list(self._workers.keys())
        for _, seq in enumerate(seq_group_metadata_list):
            sd_params = seq.spec_decode_params
            if sd_params is not None:
                proposer = sd_params.get_proposer()
                if proposer not in valid_proposers:
                    logger.info(
                        "proposer_name must be in %s, or set to None. "
                        "Got '%s' instead. Use '[ngram]' as replacement.",
                        valid_proposers, proposer)
                    proposer = '[ngram]'
                    sd_params.set_proposer(proposer)
                if self.is_multi_step_worker_instance(self._workers[proposer]):
                    break
        else:
            return []
        
        return self._workers[proposer].execute_model(execute_model_req)

    def get_cache_block_size_bytes(self) -> int:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                return worker.get_cache_block_size_bytes()
            
        return 0
    
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                return worker.determine_num_available_blocks()
            
        return -1, -1

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        for worker in self._workers.values():
            if self.is_multi_step_worker_instance(worker):
                worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
            
        return
    
    def _get_proposer_for_this_step(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
        schedule_policy: Optional[str] = "proposal_latency",
    ) -> str:
        """Get the current proposer for the given sequence batch according to
        required schedule_policy.
        """
        chosen_proposer = '[ngram]'
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        valid_proposers = list(self._workers.keys())

        if schedule_policy == "popularity":
            proposer_count: Dict[str, int] = {}
            # Count the proposer choices
            for seq in seq_group_metadata_list:
                sd_params = seq.spec_decode_params
                if sd_params is not None:
                    proposer = sd_params.get_proposer()
                    if proposer not in valid_proposers:
                        continue
                    if proposer not in proposer_count:
                        proposer_count[proposer] = 0
                    proposer_count[proposer] += 1
            # Choose the proposer with the highest count
            if len(proposer_count.keys()) != 0:
                chosen_proposer = max(proposer_count, key=proposer_count.get)

        elif schedule_policy == "proposal_latency":
            for _, seq in enumerate(seq_group_metadata_list):
                sd_params = seq.spec_decode_params
                if sd_params:
                    # Since MultiProposersWorker only supports Ngram as the
                    # backup proposer currently, we should use Ngram for the
                    # whole batch if any seq_group specifies it.
                    # TODO: Refactor this when flexible backup speculative
                    # model choices and latency metrics are supported.
                    proposer = sd_params.get_proposer()
                    if proposer == '[ngram]':
                        chosen_proposer = proposer
                        break

                    if proposer not in valid_proposers:
                        continue
                    else:
                        chosen_proposer = proposer

        elif schedule_policy == "proposal_quality":
            # TODO: Use SpecDecodeWorkerMetrics to select the best proposer
            raise NotImplementedError(
                f"schedule_policy: '{schedule_policy}' has not been "
                f"implemented yet.")
        else:
            raise ValueError(f"Invalid schedule_policy: '{schedule_policy}'.")

        return chosen_proposer
    
    def _get_combined_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. This method
        use multiple speculative proposers to generate speculations and return
        the combined results.
        """
        
        # Get speculative proposals from each proposer.
        proposer_requests: Dict[str, List[SequenceGroupMetadata]] = {}
        original_indices: Dict[str, List[int]] = {}
        valid_proposers = list(self._workers.keys())

        for idx, seq in enumerate(execute_model_req.seq_group_metadata_list):
            sd_params = seq.spec_decode_params
            if sd_params:
                proposer = sd_params.get_proposer()
                if proposer not in valid_proposers:
                    # Got unknown proposer. Use '[ngram]' as default instead.
                    proposer = '[ngram]'
                if proposer not in proposer_requests:
                    proposer_requests[proposer] = []
                    original_indices[proposer] = []
                proposer_requests[proposer].append(seq)
                original_indices[proposer].append(idx)
        
        all_proposals: Dict[str, SpeculativeProposals] = {}

        # Although we use ThreadPoolExecutor to get_spec_proposals concurently,
        # we still need to wait for the slowest proposer to finish on each
        # batch for further scoring.
        # TODO: Fix this when there are multiple scorer instances available for
        # scoring.
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._workers[proposer].get_spec_proposals,
                                execute_model_req.clone(sq_list),
                                seq_ids_with_bonus_token_in_last_step):
                                proposer
                for proposer, sq_list in proposer_requests.items()
                if len(sq_list) != 0}
            
            for future in futures:
                proposer = futures[future]
                all_proposals[proposer] = future.result()

        seq_group_metadata_length = len(
            execute_model_req.seq_group_metadata_list)
        merged_token_ids = [None] * seq_group_metadata_length
        merged_probs = [None] * seq_group_metadata_length
        merged_lens = [None] * seq_group_metadata_length
        
        # Combine and restore the original order of the proposals
        for proposer, indices in original_indices.items():
            proposals = all_proposals[proposer]
            if len(indices) != 0:
                for i, idx in enumerate(indices):
                    merged_token_ids[idx] = proposals.proposal_token_ids[i]
                    merged_probs[idx] = proposals.proposal_probs[i]
                    merged_lens[idx] = proposals.proposal_lens[i]
                    
        combined_proposals = SpeculativeProposals(
            proposal_token_ids=torch.stack(merged_token_ids),
            proposal_probs=torch.stack(merged_probs),
            proposal_lens=torch.stack(merged_lens)
        )
        return combined_proposals

    def is_multi_step_worker_instance(self, obj: ProposerWorkerBase) -> bool:
        if isinstance(obj, MultiStepWorker):
            return True
        elif isinstance(obj, SmallerTpProposerWorker):
            if hasattr(obj, '_worker'):
                return self.is_multi_step_worker_instance(obj._worker)
        else:
            return False