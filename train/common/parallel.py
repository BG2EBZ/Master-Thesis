from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


POLICY_SEARCH_PROCESS_START_METHOD = "spawn"


def build_process_pool(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=int(max_workers),
        mp_context=mp.get_context(POLICY_SEARCH_PROCESS_START_METHOD),
    )
