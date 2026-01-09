"""Test that fitsbolt imports within acceptable time."""

import subprocess
import sys


def test_import_time():
    """Test that importing fitsbolt takes less than 200ms.

    This guards against accidentally adding heavy imports at module level.
    Baseline before optimization was ~1200ms, optimized to ~150ms.
    """
    max_import_time = 0.2  # 200ms threshold

    # Run import in a fresh Python process to get accurate timing
    code = """
import time
start = time.perf_counter()
import fitsbolt
end = time.perf_counter()
print(f"{end - start:.6f}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Import failed: {result.stderr}"

    import_time = float(result.stdout.strip())
    assert (
        import_time < max_import_time
    ), f"Import took {import_time:.3f}s, expected < {max_import_time}s"
