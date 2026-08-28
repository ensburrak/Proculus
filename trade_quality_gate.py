# -*- coding: utf-8 -*-
"""Compatibility entrypoint for the quality-gate package."""
from __future__ import annotations

import _trade_quality_gate_static as _static
from decision._static_facade import expose_static_module

expose_static_module(_static, globals())
