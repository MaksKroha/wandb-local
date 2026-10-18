"""``RunDisabled`` is what the real client returns for ``mode="disabled"``.

wandb-local returns a regular :class:`wandb.Run` with ``run.disabled == True``;
these aliases keep ``isinstance`` checks in third-party code working.
"""

from ..._run import Run as RunDisabled
from ..._summary import Summary as SummaryDisabled

__all__ = ["RunDisabled", "SummaryDisabled"]
