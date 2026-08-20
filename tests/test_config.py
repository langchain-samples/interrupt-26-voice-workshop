from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from voice_workshop.tracing import configure_langsmith, require_environment


class EnvironmentTests(unittest.TestCase):
    def test_missing_environment_names_keys_without_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                require_environment("OPENAI_API_KEY")

    def test_configure_langsmith_uses_backend_default(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "placeholder"}, clear=True):
            project = configure_langsmith("pipecat")
            self.assertEqual(project, "interrupt-voice-workshop-pipecat")

if __name__ == "__main__":
    unittest.main()
