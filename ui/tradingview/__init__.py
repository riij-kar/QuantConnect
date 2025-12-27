"""TradingView chart component scaffolding.

This module exposes a thin Dash wrapper that pairs with the JavaScript helper
in ``price_volume.js``. The wrapper does not yet bind to Dash callbacks; it
simply reserves a container ``div`` and defers to the JavaScript module for
chart lifecycle management. Future phases will add the necessary clientside
callbacks that bridge Dash data payloads to the TradingView renderer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from dash import html


class PriceVolumeChart(html.Div):
    """Placeholder Dash component for the TradingView price/volume chart.

    Parameters
    ----------
    id : str
        Dash component identifier used by callbacks.
    candles : Iterable[Dict[str, Any]], optional
        Initial candle data to expose via ``data-candles`` attribute.
    volume : Iterable[Dict[str, Any]], optional
        Initial volume histogram entries to expose via ``data-volume`` attribute.
    markers : Iterable[Dict[str, Any]], optional
        Optional marker definitions for entry/exit signals.
    chart_options : Dict[str, Any], optional
        Lightweight-charts configuration dictionary serialized into
        ``data-chart-options`` for the JavaScript helper.
    style : Dict[str, Any], optional
        Additional CSS applied to the underlying ``div``.
    """

    def __init__(
        self,
        id: str,
        *,
        candles: Optional[Iterable[Dict[str, Any]]] = None,
        volume: Optional[Iterable[Dict[str, Any]]] = None,
        overlays: Optional[Dict[str, Any]] = None,
        markers: Optional[Iterable[Dict[str, Any]]] = None,
        chart_options: Optional[Dict[str, Any]] = None,
        style: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        attrs = {
            'data-candles': json.dumps(list(candles) if candles else []),
            'data-volume': json.dumps(list(volume) if volume else []),
            'data-overlays': json.dumps(overlays or {'lines': {}, 'supertrend': {}, 'legend': []}),
            'data-markers': json.dumps(list(markers) if markers else []),
            'data-chart-options': json.dumps(chart_options or {}),
            'data-meta': json.dumps({}),
            'data-last-render': '',
            'data-component': 'price-volume-chart',
            'style': {
                'width': '100%',
                'height': '500px',
                'position': 'relative',
                **(style or {}),
            }
        }
        attrs.update(kwargs)
        super().__init__(id=id, **attrs)

    @staticmethod
    def script_tag() -> html.Script:
        """Return a ``<script>`` tag that loads the helper from Dash assets."""

        return html.Script(src='/assets/tradingview/price_volume.js')
