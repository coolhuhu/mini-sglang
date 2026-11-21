import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        #
        # token_pool 管理的是请求的 input_ids，它和 page_table 的关系是，
        # token_pool[i] 和 page_table[i] 表示的是同一个 Req[i]，
        # token_pool[i][j] 表示的是 Req[i] 的第 j 个 token 的 token id；
        # page_table[i][j] 表示的是 Req[i] 的第 j 个 token 的 KVCache 存放在 kvcache_pool 的那个位置，
        # 即通过 page_table[i][j] 在 kvcache_pool 中查找 kvcache
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
