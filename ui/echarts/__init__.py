"""Dash wrapper for Apache ECharts indicator panels."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from dash import html


class EChartsPanel(html.Div):
    """Lightweight Dash component that delegates rendering to ECharts assets."""

    def __init__(
        self,
        id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        style: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        attrs = {
            'data-config': json.dumps(config or {}),
            'data-last-render': '',
            'data-component': 'echarts-panel',
            'style': {
                'width': '100%',
                'minHeight': '200px',
                'position': 'relative',
                **(style or {}),
            }
        }
        attrs.update(kwargs)
        super().__init__(id=id, **attrs)

    @staticmethod
    def script_tag() -> html.Script:
        return html.Script(src='/assets/echarts/indicator_panels.js')
