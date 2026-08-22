import unittest

from train.common.parallel import (
    POLICY_SEARCH_PROCESS_START_METHOD,
    build_process_pool,
)


class PolicySearchParallelTest(unittest.TestCase):
    def test_process_pool_uses_spawn_context(self):
        with build_process_pool(max_workers=1) as executor:
            self.assertEqual(
                executor._mp_context.get_start_method(),
                POLICY_SEARCH_PROCESS_START_METHOD,
            )


if __name__ == "__main__":
    unittest.main()
