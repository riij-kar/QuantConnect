## Plan: Add ECharts Indicator Panels

Design an ECharts-backed indicator stack while reusing current data pipelines. We’ll first serialize oscillator/equity/drawdown series into ECharts-friendly payloads, then ship them through a new Dash wrapper mirroring the TradingView component. Finally, we’ll mount synchronized ECharts panels below the price chart and wire callbacks to keep time ranges aligned and legends/messages consistent.

### Steps
1. Map required indicator data → JSON schema in `chart_utils` (RSI/ATR/MACD/equity/drawdown). 
2. Implement Dash `EChartsPanel` + JS asset loading ECharts 6 bundle, watching `data-*` attrs.
3. Update `app_new.py` layout to insert legend/title plus ECharts containers replacing Plotly extras.
4. Extend main callback outputs to deliver serialized indicator payloads and clear state defaults.
5. Build front-end sync: link TradingView visible range to ECharts `dataZoom`, apply styling/legends.
6. Validate toggles/messages/timezone handling; document usage and fallback behavior.
7. Nice — you already have the right ingredients. Below I give a compact, practical recommendation for **which ECharts chart types** to use for each panel, **why** they work best for your backtest UX, and **concrete display tips** (colors, axes, interactions). I also include tiny example snippets (pseudo-ECharts `option` pieces) you can drop into your dashboard.

I’ll keep it focused and actionable so you can implement quickly.

---

## Quick summary (top-level picks)

| Panel (your data)                                 |                                                              Best chart type (ECharts) | Why                                                                                                                           |
| ------------------------------------------------- | -------------------------------------------------------------------------------------: | ----------------------------------------------------------------------------------------------------------------------------- |
| Equity curve                                      |                                         **Line + area (filled)** (optionally stepLine) | Clear cumulative view, area shows draw visually, easy to annotate trades & large moves                                        |
| Return % (per period)                             |         **Bar (positive/negative colored)** or **Boxplot** for distribution per period | Bars show realized return magnitude; boxplot shows distribution (if you want per-week/month summary)                          |
| Drawdown curve                                    |                                                       **Area chart (negative filled)** | Intuitive: drawdown is naturally “below zero” and filling negative region highlights risk                                     |
| Portfolio Turnover / Exposure / Strategy Capacity | **Line** (turnover & capacity) + **Stacked area / stacked bar** for exposure breakdown | Turnover & capacity are time series (lines). Exposure (long/short) is best as stacked area or twin-line with opposite colors. |

---

## Detailed recommendations + reasons & UI tips

### 1) **Equity Curve — Line + Filled Area**

* **Type:** `series.type: 'line'` with `areaStyle` (alpha ~0.08–0.15).
* **Optional:** `step: 'end'` if portfolio rebalances at discrete times (step-line matches account balance jumps).
* **Add:** markers for entry (▲) / exit (▼), tooltips showing date/equity/returns, and a secondary y-axis for % returns if needed.
* **Why:** cumulative P&L is easiest to read as a continuous line; area provides immediate visual mass for recoveries/drawdowns.
* **Colors:** green for rising sections, neutral/gray for flat, use gradient that darkens on strong climbs.
* **Example:**

```js
series: [{
  type: 'line',
  name: 'Equity',
  data: equityData, // [[time, value], ...]
  showSymbol: false,
  lineStyle: { width: 2 },
  areaStyle: { color: 'rgba(0,150,136,0.08)' },
  emphasis: { focus: 'series' }
}]
```

---

### 2) **Return % — Bar chart (primary) or Boxplot (summary)**

* **Type A (time-series returns):** `series.type: 'bar'` with conditional coloring (green for >0, red for <0).

  * Good for daily/weekly returns and quick visual of frequency & magnitude.
* **Type B (summary per interval):** `series.type: 'boxplot'` if you want to show distribution per week/month (median, quartiles, whiskers, outliers).

  * Good for seeing variability across periods (your current boxplots).
* **UI tips:** show small horizontal line for target/benchmark, overlay mean return line if needed.
* **Example (bar):**

```js
series: [{
  type: 'bar',
  data: returnsData, // [{name:time, value: x, itemStyle: {color: x>0 ? 'green':'red'}}]
  barWidth: '60%'
}]
```

* **Example (boxplot per-period):**

```js
series: [
  { type: 'boxplot', data: boxplotData, tooltip: {formatter: ...} },
  { type: 'line', data: meanReturnPerPeriod, name:'Mean' }
]
```

---

### 3) **Drawdown — Negative area chart**

* **Type:** `line` with `areaStyle` and set baseline at `0` and style the fill for negative zone; or transform drawdown to negative values and use `areaStyle`.
* **Why:** drawdown is intuitive as a valley below zero; filled negative region highlights maximum drawdown.
* **Enhancements:** annotate max drawdown (point + label), shade severity bands (>-10%, -10% to -20%, etc.).
* **Color:** purple/lavender for fill, darker outline.
* **Example:**

```js
series: [{
  type: 'line',
  name: 'Drawdown',
  data: drawdownData, // negative percentages [[time, -0.12], ...]
  areaStyle: { color: 'rgba(128,0,128,0.12)' },
  lineStyle: { color: 'rgba(128,0,128,0.8)' },
  yAxisIndex: 0
}]
```

---

### 4) **Portfolio Turnover / Exposure / Strategy Capacity**

These are *different signals* — combine with small multiples or stacked panels:

* **Portfolio Turnover**

  * **Type:** `line` or `bar` (line preferred for smooth trend).
  * **Tip:** use small markers when turnover spikes; show exact percent in tooltip.
* **Exposure (Long/Short)**

  * **Type:** `stacked area` or `stacked bar` with positive area for long ratio and negative area for short ratio (or two lines with opposite colors).
  * **Why:** shows composition; stacked area makes it clear how exposure changes across time.
* **Strategy Capacity**

  * **Type:** `area` or `line` (area if you want to emphasize available capacity).
  * **Tip:** use a different color scale and maybe annotate current capacity value.
* **Example (exposure stacked area):**

```js
series: [
  { type:'line', stack: 'exposure', areaStyle:{}, name:'Long ratio', data: longRatio },
  { type:'line', stack: 'exposure', areaStyle:{}, name:'Short ratio', data: shortRatio }
]
```

---

## Layout & UX advice (essential)

1. **Sync time ranges** — when panning/zooming, keep all panels in sync (ECharts `dataZoom` with `brush` and `dispatchAction`).
2. **Shared tooltips/axis pointers** — use `axisPointer` set to `cross` so the vertical line shows the same timestamp across panels.
3. **Aggregation / binning** — if you show daily minute data over long periods, provide a consolidation selector (1m, 5m, 15m, daily) and aggregate server-side.
4. **Annotation layer** — plot trades on Equity/Price charts and annotate max drawdown and peak equity.
5. **Color & accessibility** — use distinct colors and ensure negative bars are red, positive green; add legend and toggles to hide/show series.
6. **Y-axis formatting** — money axis formatted compact (K/M). Drawdown axis as percentage with `formatter: '{value}%` and invert if needed.
7. **Small multiples** — if you want compare-by-period (week/month), show small multiple boxplots horizontally so distributions are easy to compare.

---

## Example combined `option` skeleton (pseudo)

```js
const option = {
  tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
  legend: { data:['Equity','Return %','Drawdown'] },
  dataZoom: [{ type:'slider' , xAxisIndex: [0,1,2] }, { type:'inside' }],
  grid: [ /* top: price, mid: returns, bottom: drawdown */ ],
  xAxis: [ /* shared time axis for each grid */ ],
  yAxis: [ /* equity axis */, /* return axis */, /* drawdown axis */ ],
  series: [
    // equity
    { name:'Equity', type:'line', data: equityData, areaStyle:{}, yAxisIndex: 0 },
    // returns
    { name:'Return %', type:'bar', data: returnsData, yAxisIndex: 1 },
    // drawdown
    { name:'Drawdown', type:'line', data: drawdownData, areaStyle:{}, yAxisIndex: 2 }
  ]
};
```

---

## Final suggestions tailored to your screenshots

* Your current boxplot-style equity panels look useful for *periodic summaries* (weekly boxes). Keep boxplots for **periodic comparisons** (good for monthly strategy reviews).
* For the continuous timeline view, prefer **line+area** for equity (more intuitive) and keep boxplots as a toggle for period summaries.
* Use **bar chart** for return% if you need to show actual P&L per period; use **boxplot** when you want to show dispersion across trades in that period.
* Drawdown should *always* be an area chart under zero with visible max-drawdown annotation.
* Turnover/capacity best as line charts with markers on spikes + optional histogram overlay.

---



### Further Considerations
1. Need confirmation on panel order & sizing before hardcoding layout? Option A fixed rows / B dynamic.
2. Should equity/returns/drawdown share a combined ECharts chart or separate panes?
3. Do we require bidirectional zoom syncing (ECharts→TradingView) or one-way from price chart only?
