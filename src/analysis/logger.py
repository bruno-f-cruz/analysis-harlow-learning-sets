"""Package-wide logger.

Every pipeline cell and library module should import from here::

    from analysis.logger import log

The logger is named ``"pipeline"`` and starts with no handlers (NullHandler
by default). ``run_setup`` in ``workflows/pipeline.py`` installs a stdout
handler with a timestamp format so output appears both in marimo's cell-output
area and in the ``out.log`` captured by ``scripts/start_prod.sh``.
"""

import logging

log = logging.getLogger("pipeline")
log.addHandler(logging.NullHandler())
