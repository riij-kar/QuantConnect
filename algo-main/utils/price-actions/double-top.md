The detailed implementation plan for the `calculate_double_tops` function, structured to align with your segmented approach requirements.

### Implementation of the Double Top Pattern

<hr>

### 1. The Visual Roadmap

This roadmap illustrates the psychological phases of the pattern, from the initial uptrend to the final breakout.

```text
       Phase 2      Phase 3
      (The Dip)    (The Retry)
          ↓            ↑
      Peak 1        Peak 2
       ____          ____
      /    \        /    \
     /      \      /      \      Phase 5
    /        \    /        \    (The Crash)
   /          \__/          \        ↓
  /          (Neckline)      \     _ _ _ _  <-- BREAKOUT!
 /              ↑             \   /
Phase 1      Support           \ /
(The Rise)   Level              ↓
```

<hr>

Understood.
Below is your **exact final document in pure MARKDOWN format**, with all formulas rendered using LaTeX syntax (`$...$` or `$$...$$`) — ready for GitHub, Notion, Obsidian, or PDF conversion.

You can copy-paste this directly as a `.md` file.

---

# 📘 *A Fully Statistical, Volatility-Adaptive, Causal Double-Top Detection Framework for Retail + Institutional Trading*

---

## **Overview**

This document defines a complete, production-grade probabilistic and volatility-normalized **Double Top** pattern detection system optimized for:

* **Intraday trading** (1m/5m/15m bars)
* **Swing & positional trading** (daily/weekly)
* **QC/Lean backtesting + live trading**

The framework is:

* **Mathematically grounded** (probability, volatility-scaling, significance testing)
* **Dynamically adaptive** (percentile-based thresholds)
* **Causally safe** (no future leakage → live-capable)
* **High-precision with low false positives**

---

# **1. Inputs & Warm-Up**

### **Required Inputs**

* OHLCV bar stream
* Stable price series:
  $$P_t = \frac{H_t + L_t + C_t}{3}$$
* ATR (default 14)

### **Warm-Up**

Before detection:

* ATR must be **IsReady == True**
* Minimum history:

  * Intraday: $$H_{\text{calibration}} \ge 500$$
  * Daily: $$H_{\text{calibration}} \ge 200$$

Used for percentile-based calibration.

---

# **2. Causal Smoothing (Preserve Geometry)**

Standard MAs (SMA/EMA) distort peaks.
Instead, use **one-sided local polynomial regression** (causal Savitzky–Golay).

### **Method**

Given past window:

$$[y_{t-W+1}, \ldots, y_t]$$

Fit polynomial degree (p):

$$\min_{\beta} \sum_{i=0}^{W-1} \left( y_{t-i} - \sum_{k=0}^{p} \beta_k i^k \right)^2$$

Smoothed value:

$$S_t = \beta_0$$

### **Adaptive Window**

$$W = \text{odd}\left(\mathrm{round}\left( \frac{\mathrm{median}(\Delta t_{\text{peaks}})}{3} \right)\right)$$

Benefits:

* Causal (no lookahead)
* Preserves curvature
* Removes high-frequency noise

---

# **3. Volatility Normalization (ATR Scaling)**

Normalize geometric differences by ATR:

$$X_{\text{atr}} = \frac{X}{ATR_t}$$

Applied to:

* Peak equality
* Trough depth
* Prominence
* DTW distances

Eliminates false positives in high-volatility regimes.

---

# **4. Automatic Time-Scale & Threshold Calibration**

Uses rolling window (H).
No fixed parameters → fully data-driven.

---

## **Step 1 — Loose Peak Scan**

Detect peaks in smoothed series using low prominence → enables statistical sampling.

---

## **Step 2 — Compute Distributions**

For each adjacent peak pair ((p_1, p_2)):

### **Temporal gap**

$$\Delta t = p_2 - p_1$$

### **Height difference**

$$hdiff_{\text{atr}} = \frac{\lvert S_{p1} - S_{p2} \rvert}{ATR_{p2}}$$

### **Trough depth**

$$depth_{\text{atr}} = \frac{(S_{p1} + S_{p2})/2 - T}{ATR_T}$$

### **Prominence**

$$prom_{\text{atr}} = \frac{\text{prominence}}{ATR_p}$$

### **Symmetry**

$$sym = 1 - \frac{\lvert dt_1 - dt_2 \rvert}{dt_1 + dt_2}$$

---

## **Step 3 — Percentile-Based Thresholds**

| Parameter      | Formula                                           | Interpretation             |
| -------------- | ------------------------------------------------- | -------------------------- |
| Min spacing    | $$\text{min_dt} = p_5(\Delta t)$$                 | Avoid noise peaks          |
| Max spacing    | $$\text{max_dt} = p_{95}(\Delta t)$$              | Avoid overly wide patterns |
| Peak equality  | $$eq_peak_k = p_{33}(hdiff_{\text{atr}})$$        | Accept equal tops          |
| Min depth      | $$\text{min_depth} = p_{40}(depth_{\text{atr}})$$ | Require meaningful dip     |
| Min prominence | $$\text{min_prom} = p_{50}(prom_{\text{atr}})$$   | Avoid flat noise           |
| Min symmetry   | $$\text{min_sym} = p_{30}(sym)$$                  | Balanced structure         |

Fully automatic across instruments/timeframes.

---

# **5. Candidate Detection (Geometry + Statistics)**

## **Step 1 — Peak Detection**

Peaks must satisfy:

* spacing ≥ min_dt
* prominence ≥ min_prom × ATR

## **Step 2 — Validate Peak Pair**

### **Condition A — Temporal spacing**

$$\text{min_dt} \le p_2 - p_1 \le \text{max_dt}$$

### **Condition B — Height equality**

$$hdiff_{\text{atr}} \le eq_peak_k$$

### **Condition C — Trough depth**

$$depth_{\text{atr}} \ge \text{min_depth}$$

### **Condition D — Prominence**

$$prom_{\text{atr}} \ge \text{min_prom}$$

### **Condition E — Symmetry**

$$sym \ge \text{min_sym}$$

✔ If all satisfied → **Candidate Double Top**

---

# **6. Neckline Break Confirmation (Causal)**

A candidate pattern becomes **Confirmed** when:

$$\exists t' \in (p_2,\ p_2 + N] : Close_{t'} < T$$

Where:

$$N = \mathrm{round}\left(\frac{\mathrm{median}(\Delta t)}{2}\right)$$

This is the **most reliable** entry confirmation.

---

# **7. DTW Shape Validation**

DTW compares pattern to canonical M-shaped template.

### Steps:

1. Build template (M) of length (L)
2. Extract candidate window → resample to (L)
3. Normalize (z-score)
4. Compute
   $$d = DTW(W, M)$$
5. Convert to quantile:
   $$q = P(DTW_{\text{hist}} \le d)$$

Accept if:

$$q \le 0.05$$

---

# **8. Composite Quality Score (Probabilistic Decision Rule)**

Let each metric produce score (s_i \in [0,1]).

Final decision score:

$$Q = \sum_{i=1}^{5} w_i s_i , \quad \sum w_i = 1$$

Thresholds:

* **Candidate:** (Q \ge Q_{\text{cand}})
* **Confirmed:** (Q \ge Q_{\text{conf}}) *or* neckline broken

---

# **9. Evaluation & Walk-Forward Validation**

## **A. Classification Metrics**

### Precision

$$Precision = \frac{TP}{TP + FP}$$

### Recall

$$Recall = \frac{TP}{TP + FN}$$

### F1 Score

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

Goal for retail:

* High Precision
* Acceptable Recall

---

## **B. P&L Metrics**

### Expectancy

$$E = P(win)\cdot AvgWin - P(loss)\cdot AvgLoss$$

Other metrics:

* Sharpe
* Drawdown
* Win-rate

---

## **C. Walk-Forward Calibration Workflow**

1. Split dataset into rolling Train/Test
2. Train window:

   * Recompute percentiles
   * Optimize score weights
3. Test window:

   * Validate precision, recall, P&L
4. Aggregate stable parameters

---

## **D. Statistical Significance (Bootstrap Tests)**

Depth significance:

$$p = P(depth_{\text{null}} \ge depth_{\text{obs}})$$

Require:
$$p \le 0.05$$

Same test applied to DTW distances.

---

# **10. QC/Lean Implementation Notes**

### **Causality**

* Use **one-sided** smoothing (no SciPy SavGol)

### **ATR**

```python
self.atr = self.AverageTrueRange(symbol, 14, MovingAverageType.Wilders)
```

### **Peak Detection (No SciPy)**

Compute prominence manually:

* left minimum
* right minimum
* prominence = peak - max(left, right)

### **Streaming Percentiles**

Use:

* P² quantile estimator
* or rolling window

### **No Lookahead**

All logic must use data available at bar (t).

---

# **11. FP vs FN Optimization Summary**

### **Reduce False Positives**

* ATR normalization
* Percentile thresholds
* Neckline confirmation
* DTW validation
* Bootstrap significance filtering

### **Reduce False Negatives**

* Loose early peak scan
* Adaptive thresholds
* DTW rescue
* Geometry-preserving smoothing

---

# 🎉 **The Markdown Document Is Now Fully Generated and Ready**

If you'd like, I can also generate:

📄 **PDF**
📄 **Word (.docx)**
🧠 **Research poster**
📊 **Flowcharts / diagrams**
💻 **QC/Lean-ready code template**

Just tell me!


1.  **Initial Checks**: The function immediately checks `self.is_ready()` to ensure data integrity before proceeding.
2.  **Peak and Trough Detection**:
    *   The loop iterates through historical bars comparing `current_bar` and `next_bar`.
    *   It identifies a **First Peak** when the price transitions from rising to falling.
    *   It then continuously updates the **Trough** (support level) as the lowest point reached after the first peak.
3.  **Finding the Second Peak**:
    *   The logic listens for a rally after the trough. If the price rises but fails to exceed the `first_peak` and then turns down again, it identifies a **Second Peak**.
    *   This adheres to the psychology that the "second rally fell short."
4.  **Confirmation Logic**:
    *   The critical validation occurs when `next_bar.Close < trough`. This signifies the **Break of Neckline**.
    *   Once confirmed, the pattern is recorded, and variables are reset to detect subsequent formations without overlap errors.
5.  **Invalidation**:
    *   If the price during the second rally exceeds the `first_peak`, the pattern is invalidated, and the logic treats this new high as a potential new `first_peak`.






```python
def calculate_double_tops(self) -> Dict[str, Any]:
    if not self.is_ready():
        return {}

    history = np.array([bar.Close for bar in self.pattern_history])
    peaks_indices = np.where(
        (history[1:-1] > history[:-2]) & (history[1:-1] > history[2:])
    )[0] + 1
    troughs_indices = np.where(
        (history[1:-1] < history[:-2]) & (history[1:-1] < history[2:])
    )[0] + 1

    if len(peaks_indices) < 2 or len(troughs_indices) < 1:
        return {}

    # default: no pattern
    latest_state: Dict[str, Any] = {}

    for i in range(len(peaks_indices) - 1):
        p1_idx = peaks_indices[i]
        p2_idx = peaks_indices[i + 1]

        interim_troughs = troughs_indices[
            (troughs_indices > p1_idx) & (troughs_indices < p2_idx)
        ]
        if len(interim_troughs) == 0:
            continue

        trough_idx = interim_troughs[np.argmin(history[interim_troughs])]

        p1_price = float(history[p1_idx])
        p2_price = float(history[p2_idx])
        trough_price = float(history[trough_idx])

        potential = (p2_price < p1_price) and (p2_price > trough_price)

        post_p2 = history[p2_idx + 1:]
        breakdown_occurred = bool(post_p2.size and np.any(post_p2 < trough_price))
        retest_detected = bool(
            post_p2.size
            and np.any(post_p2 >= trough_price)
            and np.any(post_p2 < p1_price)
        )

        latest_state = {
            "pattern_name": "Double_Top",
            "potential": potential,
            "breakdown": breakdown_occurred,
            "retest": retest_detected,
            "pattern_formed": potential and breakdown_occurred,
            "neckline_price": trough_price,
            "peak_1_price": p1_price,
            "peak_2_price": p2_price,
            "time": str(self.pattern_history[p2_idx].EndTime),
        }

        # If you only care about the most recent pair, break here
        break

    return latest_state
```

Key points:

- Return type is a single dict, so change annotation to `Dict[str, Any]` and update docstring.
- Always return the same fields; caller can check `potential`, `breakdown`, `retest`, `pattern_formed`.
- No shared mutable `pattern_info` outside the loop; no stale state.
- If you prefer a list for future multiple patterns, then return `results` (a list of such dicts) and update the type accordingly.

If you want, I can apply this exact refactor into `price_action_patterns.py` for you and adjust the type hints to match.

riij-kar: explain me blow:
Denosies the data (Savitzky-Golay).
Detects Peaks (Local Maxima).
Filters based on ATR (adapting to market noise).
Visualizes the result (crucial for retail verification).

GitHub Copilot: Here’s each line in plain language, with how it fits trading.

**1. Denoises the data (Savitzky–Golay)**  
- Candle closes (or prices) are noisy: lots of tiny wiggles that aren’t real “moves.”  
- A Savitzky–Golay filter is a smoothing technique that:
  - Fits a small polynomial over a moving window.
  - Keeps the general shape and turning points better than a simple moving average.  
- “Denoise” here means: apply this filter so prices become smoother, making real swings easier to see and tiny jitters less important.

**2. Detects Peaks (Local Maxima)**  
- On the (optionally smoothed) price series, a “peak” (local maximum) is a bar higher than its neighbors.  
- The algorithm looks for points where:
  - $P[i] > P[i-1]$ and $P[i] > P[i+1]$ (basic version), or
  - Satisfies a more advanced peak-detection rule.  
- This gives you candidate swing highs you can use for double tops, resistance, etc.

**3. Filters based on ATR (adapting to market noise)**  
- ATR (Average True Range) measures typical bar-to-bar volatility.  
- Using ATR as a filter means:
  - Ignore peaks where the move is smaller than some multiple of ATR (e.g. 0.5× ATR, 1× ATR).  
  - This adjusts automatically to market conditions: in quiet markets, even small moves can matter; in volatile markets, you require larger swings.  
- Result: only “significant” peaks survive, not tiny wiggles.

**4. Visualizes the result (crucial for retail verification)**  
- After detection, you plot:
  - Original (or smoothed) price series.
  - Markers on detected peaks (and troughs).  
- For a retail trader, this is essential:
  - They can visually verify that the algorithm is marking the same swing points they would see by eye.
  - It builds trust that the pattern logic (e.g., double tops) matches reality, not just math on paper.

If you want, next I can sketch how you’d integrate Savitzky–Golay + ATR filtering into your `PriceActionPatterns` peak detection.

riij-kar: explain below
Prominence + symmetry
DTW template matching

GitHub Copilot: Here’s what those mean in the context of pattern detection.

**Prominence + symmetry**

- **Prominence**  
  - A peak’s “prominence” is how much it stands out from surrounding price.  
  - Think: height of the peak relative to nearby troughs, not just absolute price.  
  - In practice you:
    - Take a local maximum.
    - Look down to the closest valleys (troughs) on left and right.
    - The prominence is the vertical distance from the peak to the higher of those two valleys.  
  - Using prominence as a filter means: only keep peaks that stick out enough compared to local noise (e.g., > 0.5× ATR).

- **Symmetry**  
  - For patterns like double tops/bottoms, symmetry measures how “balanced” the pattern is:
    - **Price symmetry:** peaks at similar prices (within some tolerance).  
    - **Time symmetry:** similar distance in bars from the trough to each peak, or roughly equal width of the two “legs.”  
  - A more symmetric pattern is often considered “cleaner” and higher quality.
  - You can score symmetry and require it above a threshold:
    - e.g., `abs(peak1_price - peak2_price) / peak1_price < 0.005`
    - and `abs(left_leg_bars - right_leg_bars) <= k`.

Together: you first find peaks, then use **prominence** (is this swing significant?) and **symmetry** (does it look like a textbook pattern?) to filter out weak/ugly shapes.

**DTW template matching**

- **DTW (Dynamic Time Warping)**  
  - A distance metric between two time series that allows stretching/compressing in time.  
  - Unlike Euclidean distance (which compares point-by-point), DTW can say:
    - “These two shapes are the same pattern even if one is slower / faster or slightly shifted.”

- **Template matching with DTW**  
  - You define a **template** price pattern, e.g., a canonical double-top shape normalized to [0, 1] in time and price.
  - For each window of your real price series:
    - Normalize that window (scale time/price to same range).
    - Compute DTW distance between window and the template.
    - If the distance < threshold, label it as a match.  
  - This is powerful when:
    - Patterns are similar in shape but vary in duration and slight timing.
    - You want a quantitative “how close is this to my ideal pattern?” score.

In short:

- **Prominence + symmetry** = geometric filters on peaks and structure, still working in native price/time coordinates.
- **DTW template matching** = comparing the *whole shape* of a price segment to an ideal pattern, tolerant to stretching/compression in time.

riij-kar: All the implementation/logic has been mention in #file:double-top.md  Denosies the data (Savitzky-Golay).
Detects Peaks (Local Maxima).
Filters based on ATR (adapting to market noise).
Prominence + symmetry
DTW template matching
looking at the #file:price_action_patterns.py implement those, Do you think limitation in current implementation? if any limitation What changes do you need? What details we need in #file:double-top.md  to perfectly work which mentioned in #file:double-top.md ?



GitHub Copilot: You’re right: the current `calculate_double_tops` is much simpler than what `double-top.md` describes. It has several limitations, and we’d need more concrete specs in the doc to implement the full framework.

I’ll break it into three parts:

**1. Limitations of the current `calculate_double_tops`**

Right now it:

- **Uses raw closes, no denoising**  
  - No Savitzky–Golay or any smoothing; peaks are detected directly on noisy `Close`.
  - This makes it very sensitive to micro wiggles and bar noise.

- **Pure geometric peak/trough, no ATR normalization**  
  - Peaks/troughs are detected via simple 3-point comparisons:
    - Local max: `P[i] > P[i-1]` and `P[i] > P[i+1]`.
    - Local min: symmetric.
  - No ATR-based scaling:
    - No requirement like “difference between peaks/trough > k × ATR”.
    - No volatility adjustment as described with $X_\text{atr} = X / ATR_t$.

- **No prominence, no symmetry scoring**  
  - It only checks:
    - Second peak lower than first.
    - Second peak above neckline.
  - Missing:
    - Prominence: how much the peak stands out vs surrounding valleys, possibly scaled by ATR.
    - Symmetry: balance of left vs right legs (in time and size).

- **No percentile-based auto thresholds**  
  - The doc describes:
    - Collecting distributions (`Δt`, `hdiff_atr`, `depth_atr`, `prom_atr`, `sym`).
    - Computing percentiles (`p_5, p_33, p_40, …`) to set min/max thresholds per instrument/timeframe.
  - Current code has **no statistical calibration**: just fixed inequalities.

- **No DTW / template matching**  
  - All decisions are local geometric checks; no shape comparison vs canonical M-pattern.
  - No DTW, no distance score, no “pattern quality” metric.

- **No multi-step state / quality score**  
  - It returns a pattern only when:
    - `potential` and `breakdown_occurred` are both True.
  - It does not:
    - Return intermediate states (potential only, or after DTW filter).
    - Compute a composite score $Q$.
    - Incorporate significance testing / bootstrap.

- **Causality is only partially considered**  
  - It’s using a rolling window of past bars and post-peak bars, which is causal.
  - But:
    - Smoothing is not one-sided Savitzky–Golay.
    - There’s no explicit “only use data up to t” design, warm-up, or calibration windows.

So: current implementation is fine for a *simple* geometric double-top, but it’s not yet the full statistical, volatility-adaptive, DTW-based framework from `double-top.md`.

---

**2. What changes are needed in `price_action_patterns.py`**

At a high level we’d need to expand this class a lot. Concretely:

- **A. Add ATR and possibly other inputs**  
  - Either:
    - Pass an `atr_series` into `PriceActionPatterns`, or
    - Give it a way to read `bar`’s ATR from outside.
  - We need ATR at least at peaks and troughs to compute:
    - $hdiff_\text{atr}$, $depth_\text{atr}$, $prom_\text{atr}$.

- **B. Add causal smoothing infrastructure**  
  - Implement (or wrap) a **causal** Savitzky–Golay-like smoother:
    - Given `self.pattern_history` (deque of TradeBars), compute smoothed price `S_t` using only past W bars.
  - Store:
    - Raw closes `C_t`.
    - Smoothed series `S_t` for peak detection.

- **C. Richer feature extraction for each candidate pair**  
  - For each (p1, p2, trough) triple we must compute:
    - Δt (bar distance between peaks).
    - $hdiff_\text{atr}$ = |S[p1] – S[p2]| / ATR[p2].
    - $depth_\text{atr}$ = ((S[p1] + S[p2])/2 – S[trough]) / ATR[trough].
    - $prom_\text{atr}$ using prominence / ATR.
    - Symmetry score `sym` using left vs right leg durations, and possibly price symmetry.
  - This implies:
    - We need access to ATR at multiple indices.
    - We need utility methods to compute prominence and symmetry robustly.

- **D. Percentile-based calibration and thresholds**  
  - We need code paths for “calibration mode”:
    - Over a long warm-up window (`H_calibration`).
    - Collect all candidate pairs.
    - Record distributions of Δt, $hdiff_\text{atr}$, $depth_\text{atr}$, $prom_\text{atr}$, `sym`.
    - Compute percentiles and store thresholds inside the class:
      - `self.min_dt`, `self.max_dt`, `self.eq_peak_k`, `self.min_depth`, etc.
  - That probably means:
    - Either a separate “trainer” object.
    - Or methods like `update_calibration_metrics(...)` + `finalize_thresholds()`.

- **E. DTW template infrastructure**  
  - We’d need:
    - A canonical template M-shape (as a normalized array).
    - A way to extract candidate price windows around (p1, p2, trough), resample them to length L.
    - A lightweight DTW implementation (no SciPy in QC) to compute distance.
  - Then:
    - Convert DTW distances to quantiles (again requiring a history of distances).
    - Store a threshold (e.g., q ≤ 0.05) inside the class.

- **F. Output structure / scoring**  
  - Current `calculate_double_tops` returns a list of dicts with booleans.  
  - We’d want:
    - A richer `pattern_info` structure with:
      - All raw and ATR-normalized metrics.
      - DTW distance and quantile.
      - A composite quality score Q.
      - Explicit flags: `candidate`, `confirmed`, `retest`, `valid_after_DTW`, etc.

- **G. Separation of “scan/calibrate” vs “live detect” modes**  
  - To avoid running heavy calibration every bar in live:
    - `PriceActionPatterns` should have:
      - A calibration phase (offline, backtest, or long warm-up).
      - A runtime detection phase using frozen thresholds.
  - We’ll need some method or config flag to switch between those.

---

**3. What extra details we need in `double-top.md` to implement it precisely**

The markdown is excellent conceptually, but for coding we need to turn many “ideas” into exact numeric rules. Specific clarifications would help:

- **Causal Savitzky–Golay details**  
  - Window size:
    - You suggest adaptive $W$ based on median Δt, but we still need:
      - Minimum and maximum W.
      - Fallback if median Δt isn’t yet available.
  - Polynomial degree:
    - Fixed (e.g., p = 2) or configurable?
  - Exact formula/weights or acceptance of a simpler approximation?

- **ATR integration**  
  - ATR period is 14 by default—confirm we should hardcode 14 or read from config.
  - Should ATR be the same one already used in your strategy (LEAN indicator), or a separate UI-only ATR?
  - How do we handle missing ATR values at the very beginning (warm-up bars)?

- **Prominence definition**  
  - Exact algorithm in your context:
    - How far do we look left and right to find the “valleys” for prominence?
    - Do we use smoothed S or raw close?
    - Use the higher of left/right valleys, or something else?
  - Do we require a minimum horizontal separation for defining valleys?

- **Symmetry calculation**  
  - Which symmetry metrics are mandatory:
    - Time: bars from p1 to trough vs trough to p2.
    - Price: |S[p1] – S[p2]| or some ATR-normalized variant.
  - Exact formula and thresholds:
    - E.g., `sym = 1 - |dt1 - dt2| / (dt1 + dt2)` is listed; confirm we should use that exact formula and which percentile is our min.

- **Threshold percentiles & calibration window**  
  - You list example percentiles (5, 33, 40, 50, 30).  
  - For real implementation, we need:
    - Confirmed percentiles per metric (they can be config-driven).
    - Exact calibration window length H for intraday vs daily (or formula based on period).
  - Should thresholds be:
    - Fixed once at strategy start.
    - Or re-estimated periodically (walk-forward)?

- **DTW implementation parameters**  
  - Template:
    - How many points (L)?
    - Exact numeric shape of the template M (we can define a normalized vector).
  - Distance:
    - Vanilla DTW or constrained window (Sakoe-Chiba band)?
  - Normalization:
    - Do we z-score each window (zero mean, unit variance) or scale to [0, 1] in price/time?
  - Threshold:
    - Confirmation that q ≤ 0.05 from historical DTW_dist is the rule.
    - Size of historical DTW_dist pool for quantile estimation.

- **Quality score weights**  
  - You specify $Q = \sum w_i s_i$, but:
    - How many scores s_i (which metrics exactly)?
    - Initial default weights w_i (before optimization).
    - Default thresholds Q_cand, Q_conf if user doesn’t optimize.

- **Runtime contract for the alpha / strategy**  
  - What do you want `calculate_double_tops` to return **per bar**?
    - Single best pattern state (dict)?
    - List of patterns inside the window with full metrics?
  - What flags should the alpha look at to:
    - Start “monitoring” a forming double top.
    - Fire entry on confirmation.
    - Filter or rank patterns.

---

**Summary**

- Yes, there are significant limitations in the current `calculate_double_tops` compared to `double-top.md`: it’s purely geometric, no denoising, no ATR normalization, no prominence/symmetry, no DTW, no statistical thresholds.
- To fully implement the framework in `price_action_patterns.py`, we need:
  - ATR access, smoothing, metric extraction, calibration, DTW, and richer outputs.
- To do this “perfectly,” `double-top.md` should add:
  - Exact numeric rules for:
    - Causal smoothing (W, p).
    - Prominence and symmetry definitions.
    - Calibration window sizes and percentile thresholds.
    - DTW template and thresholds.
    - Default quality score weights and decision rules.
    - Return format the alpha expects per bar.

If you like, the next concrete step I can take is: design a first “practical” version of this framework inside `PriceActionPatterns` (using reasonable defaults for gaps in the doc), so you can start testing and then refine the markdown with more precise numbers based on what you see.


**Actual Algorithm used:**
# 
````
FUNCTION DetectDoubleTop(PriceData, ATR, RSI, Volume):
    CONSTANTS:
        K_REVERSAL = 2.0
        K_MATCH = 0.5
        K_DEPTH = 1.0

    STATIC VARIABLES (Persist between calls):
        Trend = UP
        Last_High = 0
        Last_Low = Infinity
        Pivots = List of Objects {Price, Type, Time, RSI, Volume}
        Potential_Pattern = None

    CALCULATE Reversal_Zone = ATR * K_REVERSAL

    # 1. Pivot Detection (ZigZag)
    IF Trend IS UP:
        IF Current_High > Last_High:
            Last_High = Current_High
            Last_High_RSI = Current_RSI
            Last_High_Volume = Current_Volume
        
        ELSE IF Current_Close < (Last_High - Reversal_Zone):
            # Peak Confirmed
            APPEND {Price: Last_High, Type: 'PEAK', RSI: Last_High_RSI, Volume: Last_High_Volume} TO Pivots
            
            # Check for Potential Pattern Formation
            IF Pivots.Count >= 3:
                Peak_2 = Pivots[Last]
                Trough = Pivots[Last-1]
                Peak_1 = Pivots[Last-2]
                
                Diff = ABS(Peak_1.Price - Peak_2.Price)
                Drop = Peak_1.Price - Trough.Price
                
                IF (Peak_1.Type == 'PEAK') AND 
                   (Diff <= ATR * K_MATCH) AND 
                   (Drop >= ATR * K_DEPTH):
                       # Store as Potential Pattern waiting for confirmation
                       Potential_Pattern = {Peak_1, Trough, Peak_2}
            
            # Switch Trend
            Trend = DOWN
            Last_Low = Current_Low

    ELSE IF Trend IS DOWN:
        IF Current_Low < Last_Low:
            Last_Low = Current_Low
            
        ELSE IF Current_Close > (Last_Low + Reversal_Zone):
            # Trough Confirmed
            APPEND {Price: Last_Low, Type: 'TROUGH'} TO Pivots
            
            # Switch Trend
            Trend = UP
            Last_High = Current_High

    # 2. Neckline Break Confirmation
    IF Potential_Pattern IS NOT None:
        Neckline_Price = Potential_Pattern.Trough.Price
        
        # Check if Price Breaks Below Neckline
        IF Current_Close < Neckline_Price:
            
            # 3. Optional Confirmations
            Is_Volume_Divergence = (Potential_Pattern.Peak_2.Volume < Potential_Pattern.Peak_1.Volume) AND (Current_Volume > Average_Volume)
            Is_RSI_Divergence = (Potential_Pattern.Peak_2.RSI < Potential_Pattern.Peak_1.RSI)
            
            RETURN SIGNAL "DOUBLE TOP CONFIRMED" WITH {Volume_Div: Is_Volume_Divergence, RSI_Div: Is_RSI_Divergence}
            
            # Reset Potential Pattern to avoid duplicate signals
            Potential_Pattern = None

    RETURN None
```
