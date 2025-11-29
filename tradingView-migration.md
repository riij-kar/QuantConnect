## Objective
Convert `ui/app_new.py` into a TradingView lightweight-charts dashboard while keeping `ui/app.py` on Plotly as an untouched fallback.

## Phase 1 – Component Scaffolding
- Add `ui/tradingview/price_volume.js` that loads the UNPKG lightweight-charts bundle(https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js) and mounts synchronized candlestick plus histogram charts.
- Expose props for `candles`, `volume`, `markers`, and `chartOptions`; ensure resize handling and cleanup in the component lifecycle.
- Provide a Python wrapper (e.g., `ui/tradingview/__init__.py`) exporting the Dash component class for server-side usage.

## Phase 2 – Data Serialization
- Extend `ui/utils/chart_utils.py` with helpers that convert `price_df` into lightweight-charts payloads `{ time, open, high, low, close, volume }` using ISO timestamps.
- Include logic for up/down volume coloring consistent with current Plotly behavior.
- Emit entry/exit markers and indicator overlays as neutral arrays ready for lightweight-charts series consumption.

## Phase 3 – Layout Integration
- Replace the Plotly `dcc.Graph` in `get_main_layout` of `ui/app_new.py` with the TradingView component.
- Ensure layout styles accommodate the new chart and provide placeholder data to confirm rendering.

## Phase 4 – Callback Updates
- Modify the main visualization callback to return serialized TradingView payloads instead of a Plotly figure.
- Drop unused Plotly imports and outputs in `ui/app_new.py` once the new component is wired.

## Phase 5 – Feature Parity
- Reintroduce overlays (SMA/EMA, Bollinger Bands, etc.), volume tooltips, and entry/exit toggles by feeding corresponding series through the new serializer.
- Validate resampling and timezone conversions still behave correctly with the new chart data.

## Phase 6 – Validation & Documentation
- Test across browsers for responsiveness and accuracy.
- Update `README_APPPY.md` to document the TradingView-powered flow and how it differs from the legacy Plotly app.
- When parity is achieved, consider removing Plotly-specific code paths from `ui/app_new.py`.
