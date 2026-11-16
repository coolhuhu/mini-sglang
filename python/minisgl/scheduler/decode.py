from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        # 当前处于 decode 阶段的请求中（running_reqs），未来将生成 token 数量的估计值，
        # 包含的是每个请求未来生成token的数量以及 page 对齐所浪费的token数量的总和
        #
        # 这样计算的目的是为了保证，已开始生成的请求不会因新 prefill 抢占 cache 而中途 OOM
        # page_size - 1 是一个上限值
        # 假设一个  req 的 remain_len 为 17，page_size 为 16，
        # 最不佳的情况下，当前 req 申请的 page 中没有额外的空间了，那么还需要预留 2 个 page，
        # 才能保证当前的 req 有足够的空间完成 decode。page_size - 1 保证了decode预留空间的上限值
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
