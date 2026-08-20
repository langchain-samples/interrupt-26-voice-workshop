from __future__ import annotations

import asyncio
import unittest

from voice_workshop.gemini_agent import _wait_for_workers


class WorkerCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_cancellation_is_clean_shutdown(self) -> None:
        worker = asyncio.create_task(asyncio.Event().wait())
        current = asyncio.current_task()
        self.assertIsNotNone(current)
        asyncio.get_running_loop().call_soon(current.cancel)  # type: ignore[union-attr]

        await _wait_for_workers(worker)

        self.assertTrue(worker.cancelled())
        self.assertEqual(current.cancelling(), 0)  # type: ignore[union-attr]

    async def test_unexpected_child_cancellation_is_not_swallowed(self) -> None:
        worker = asyncio.create_task(asyncio.sleep(60))
        worker.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await _wait_for_workers(worker)


if __name__ == "__main__":
    unittest.main()
