"""Non-interactive pipeline entrypoint. Equivalent to `python workflows/pipeline.py`."""

import runpy

if __name__ == "__main__":
    runpy.run_path("workflows/pipeline.py", run_name="__main__")
