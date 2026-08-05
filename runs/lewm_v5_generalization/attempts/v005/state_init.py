#!/usr/bin/env python3
"""Initialize or verify the durable generalization recovery state."""

from __future__ import annotations

import json

from study_common import initialize_state


if __name__ == "__main__":
    print(json.dumps(initialize_state(), sort_keys=True))
