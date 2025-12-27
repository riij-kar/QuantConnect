# ui TODO List

## 🎯 CRITICAL PRIORITY (Blockers)

### 1. Implement Pattern Overlay Rendering
**Status:** 🔴 Not Started  
**Complexity:** High  
**Estimated Effort:** 4-5 days

#### Current State
- ✅ TradingView chart component (`PriceVolumeChart`) exists
- ✅ `price_volume.js` has overlay infrastructure for lines and markers
- ✅ `data-overlays` attribute wired up
- ⚠️ Missing: Pattern-specific overlay rendering
- ⚠️ Missing: Pattern metadata display
- ⚠️ Missing: Loading patterns from backtest logs

#### What Needs to be Done

1. **Define Pattern Overlay Format**
   - Design JSON schema for pattern overlays:
     ```json
     {
       "patterns": {
         "double_top_1": {
           "type": "double_top",
           "state": "confirmed",
           "peak_1": {"time": 1234567890, "price": 100.5},
           "peak_2": {"time": 1234567900, "price": 100.3},
           "trough": {"time": 1234567895, "price": 98.0},
           "neckline": {"price": 98.0},
           "breakdown": {"time": 1234567905, "price": 97.5},
           "retest": {"time": 1234567910, "price": 98.2},
           "metadata": {
             "volume_divergence": true,
             "rsi_divergence": false,
             "quality_score": 0.85
           }
         }
       }
     }
     ```
   - Files: Document in `ui/docs/overlay_schema.md`

2. **Load Pattern Data in Dashboard**
   - Check for `price_actions.json` in backtest folder
   - Parse pattern log file
   - Convert to overlay format
   - Add to `tv_payload['overlays']`
   - Files: `ui/app_new.py` in `update_visual()`

3. **Extend JavaScript Renderer**
   - Add pattern overlay rendering to `price_volume.js`:
     - Draw neckline as horizontal line (dashed, gray)
     - Add markers for Peak 1, Peak 2, Trough
     - Add marker for breakdown point (red arrow down)
     - Add marker for retest (yellow triangle)
   - Color code by pattern state:
     - Potential: yellow/warning
     - Confirmed: red/danger
     - Retested: orange/caution
   - Files: `ui/assets/tradingview/price_volume.js`

4. **Add Pattern Tooltips**
   - Show rich tooltip on marker hover:
     - Pattern type and state
     - Peak prices and times
     - Neckline price
     - Breakdown/retest info
     - Quality score and divergences
   - Files: `ui/assets/tradingview/price_volume.js`

5. **UI Controls for Patterns**
   - Add checkbox to enable/disable pattern overlays
   - Add filter by pattern type (double top, double bottom, etc.)
   - Add filter by pattern state (potential, confirmed, retested)
   - Add quality score slider filter
   - Files: `ui/app_new.py` layout section

**Acceptance Criteria:**
- [ ] Patterns from backtests appear as overlays on chart
- [ ] Markers are clearly visible and labeled
- [ ] Tooltips show complete pattern information
- [ ] Overlays can be toggled on/off
- [ ] Performance: <100ms render time for 50 patterns
- [ ] Handles missing pattern files gracefully

**Dependencies:** Pattern detection must work in algo-main

---

### 2. Fix and Validate Timezone Handling
**Status:** 🟡 Partially Implemented  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Current State
- ✅ `build_lightweight_price_payload` returns metadata with timezone info
- ✅ `price_volume.js` has timezone formatter functions
- ✅ Wall-time preservation logic added
- ⚠️ Not fully tested across timezones
- ⚠️ Edge cases may exist

#### What Needs to be Done

1. **Add Comprehensive Timezone Tests**
   - Test UTC data → UTC display
   - Test naive data with timezone assumption
   - Test wall-time preservation mode
   - Test DST transitions
   - Test invalid timezone names
   - Files: New `ui/tests/test_timezone_handling.py`

2. **Validate JavaScript Formatter**
   - Test `buildCrosshairFormatter` with various timezone labels
   - Test `buildTickFormatter` with trading days set
   - Test missing `data-meta` graceful fallback
   - Files: New `ui/tests/test_price_volume_js.js` (if setting up JS tests)

3. **Document Timezone Behavior**
   - Explain UTC vs wall-time mode
   - Document configuration options
   - Show examples for different markets
   - Files: New `ui/docs/timezone_guide.md`

4. **Fix Any Discovered Issues**
   - Handle edge cases
   - Improve error messages
   - Add validation for timezone names
   - Files: `ui/utils/chart_utils.py`, `ui/assets/tradingview/price_volume.js`

**Acceptance Criteria:**
- [ ] All timezone tests pass
- [ ] Crosshair shows correct timezone label
- [ ] Tick marks use trading days correctly
- [ ] No errors with missing meta
- [ ] Documentation clear and complete

**Dependencies:** None

---

### 3. Improve Backtest Selection UX
**Status:** 🟡 Partially Done  
**Complexity:** Low  
**Estimated Effort:** 1-2 days

#### Current State
- ✅ Symbol extraction from order events attempted
- ⚠️ Error handling needs improvement
- ⚠️ Missing other useful metadata

#### What Needs to be Done

1. **Robust Symbol Extraction**
   - Handle missing order events gracefully
   - Try multiple sources (summary, config, order events)
   - Show clear indicator when symbol unavailable
   - Files: `ui/app_new.py` in `list_backtests()`

2. **Add More Backtest Metadata**
   - Show date range (start - end)
   - Show timeframe/resolution
   - Show total bars processed
   - Show pattern count (if patterns exist)
   - Files: `ui/app_new.py` in `list_backtests()`

3. **Better Backtest Sorting**
   - Sort by date (newest first) by default
   - Add option to sort by P&L
   - Add option to filter by symbol
   - Files: `ui/app_new.py`

4. **Backtest Status Indicators**
   - Color code by profitability (green/red)
   - Show warning icon for errors
   - Show pattern icon if patterns detected
   - Files: `ui/app_new.py` layout

**Acceptance Criteria:**
- [ ] Symbol shown for all backtests (or "Unknown")
- [ ] Additional metadata enriches dropdown
- [ ] Sorting and filtering work correctly
- [ ] Visual indicators helpful

**Dependencies:** None

---

## 🔥 HIGH PRIORITY

### 4. Add Pattern Analysis Panel
**Status:** 🔴 Not Started  
**Complexity:** Medium  
**Estimated Effort:** 3-4 days

#### What Needs to be Done

1. **Create Pattern Summary Section**
   - Count patterns by type
   - Count by state (potential/confirmed/retested)
   - Show average quality score
   - Files: `ui/app_new.py` in `update_visual()`

2. **Pattern Table**
   - List all detected patterns in a table
   - Columns: Type, Time, State, Peaks, Neckline, Quality
   - Sortable by any column
   - Clickable row to zoom chart to pattern
   - Files: `ui/app_new.py` layout + callback

3. **Pattern Statistics**
   - Success rate (patterns followed by profitable trades)
   - Average time to breakdown
   - Average price movement after confirmation
   - Files: `ui/utils/pattern_analyzer.py` (new)

4. **Export Patterns**
   - Export patterns to CSV
   - Include all metadata
   - Files: `ui/app_new.py` callback

**Acceptance Criteria:**
- [ ] Summary shows pattern counts
- [ ] Table displays all patterns correctly
- [ ] Clicking row zooms to pattern
- [ ] Statistics are accurate
- [ ] CSV export works

**Dependencies:** Item #1 (patterns must render first)

---

### 5. Add Pattern Performance Tracking
**Status:** 🔴 Not Started  
**Complexity:** High  
**Estimated Effort:** 4-5 days

#### What Needs to be Done

1. **Link Patterns to Trades**
   - Match pattern confirmation times with trade entries
   - Determine which trades were pattern-based
   - Track P&L per pattern type
   - Files: New `ui/utils/pattern_trade_analyzer.py`

2. **Pattern Metrics**
   - Win rate per pattern type
   - Average profit/loss per pattern
   - Best/worst patterns
   - Pattern reliability score
   - Files: `ui/utils/pattern_trade_analyzer.py`

3. **Visualization**
   - Chart: Win rate by pattern type (bar chart)
   - Chart: P&L distribution per pattern (box plot)
   - Chart: Pattern count over time (line chart)
   - Files: Add to `ui/app_new.py` using Plotly

4. **Integration**
   - Load patterns and trades together
   - Compute metrics in callback
   - Display in new dashboard tab
   - Files: `ui/app_new.py`

**Acceptance Criteria:**
- [ ] Patterns correctly linked to trades
- [ ] Metrics calculated accurately
- [ ] Charts render correctly
- [ ] Insights are actionable

**Dependencies:** Item #1, algo-main item #3 (trade integration)

---

### 6. Implement Real-time Pattern Monitoring (Future Live Trading)
**Status:** 🔴 Not Started  
**Complexity:** Very High  
**Estimated Effort:** 7-10 days

#### What Needs to be Done

1. **WebSocket/Polling Infrastructure**
   - Set up real-time data feed from algorithm
   - Stream pattern detections as they occur
   - Update UI without full refresh
   - Files: New `ui/realtime/` module

2. **Live Pattern Display**
   - Show forming patterns in real-time
   - Update state as pattern progresses
   - Alert on confirmation/breakdown
   - Files: `ui/app_new.py` + WebSocket integration

3. **Alerting System**
   - Email/SMS on pattern confirmation
   - Browser notifications
   - Configurable alert rules
   - Files: New `ui/alerts/` module

**Acceptance Criteria:**
- [ ] Patterns update in real-time (<1s latency)
- [ ] Alerts fire reliably
- [ ] UI remains responsive

**Dependencies:** Live trading infrastructure (future work)

---

## 📊 MEDIUM PRIORITY

### 7. Improve Chart Performance
**Status:** 🟡 Adequate for Current Use  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### What Needs to be Done

1. **Optimize Data Transfer**
   - Compress candle data (binary format)
   - Lazy load indicators
   - Paginate historical data
   - Files: `ui/app_new.py`, `ui/utils/chart_utils.py`

2. **Optimize Rendering**
   - Implement marker clustering for dense patterns
   - Throttle updates on zoom/pan
   - Use canvas instead of SVG for heavy overlays
   - Files: `ui/assets/tradingview/price_volume.js`

3. **Add Loading States**
   - Show spinner while loading
   - Progressive rendering
   - Cancel pending requests on navigation
   - Files: `ui/app_new.py`

**Acceptance Criteria:**
- [ ] Chart loads in <2s for 10K candles
- [ ] Smooth zoom/pan with overlays
- [ ] No UI freezing

**Dependencies:** None

---

### 8. Enhance Indicator Configuration
**Status:** 🟡 Basic Configuration Exists  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### What Needs to be Done

1. **Add Pattern-Specific Indicators**
   - Show ATR value on chart
   - Show smoothed price series
   - Show ZigZag pivots
   - Files: `ui/app_new.py`, `ui/utils/chart_utils.py`

2. **Indicator Presets**
   - Create preset configs for pattern trading
   - One-click apply presets
   - Save/load custom presets
   - Files: `ui/app_new.py`

3. **Dynamic Indicator Updates**
   - Change indicator params without reload
   - Recalculate on-the-fly (for UI-only indicators)
   - Files: `ui/app_new.py` callbacks

**Acceptance Criteria:**
- [ ] ATR and pivots visible on chart
- [ ] Presets work correctly
- [ ] Dynamic updates responsive

**Dependencies:** Item #1 (pattern overlays)

---

### 9. Add Comparison Mode
**Status:** 🔴 Not Started  
**Complexity:** Medium  
**Estimated Effort:** 3-4 days

#### What Needs to be Done

1. **Multi-Backtest Comparison**
   - Select multiple backtests
   - Show equity curves side-by-side
   - Compare pattern statistics
   - Files: `ui/app_new.py` + new layout tab

2. **A/B Testing Support**
   - Compare same strategy with/without patterns
   - Statistical significance tests
   - Visualize differences
   - Files: New `ui/utils/comparison.py`

**Acceptance Criteria:**
- [ ] Can select and compare 2+ backtests
- [ ] Charts clearly show differences
- [ ] Stats help decision making

**Dependencies:** None

---

## 🛠️ LOW PRIORITY / ENHANCEMENTS

### 10. Add Dark/Light Theme Toggle
**Status:** 🟡 Currently Dark Theme Only  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### What Needs to be Done
- Define light theme colors
- Add theme toggle button
- Store preference in browser
- Apply theme to all components

**Acceptance Criteria:**
- [ ] Theme toggle works
- [ ] All components respect theme

**Dependencies:** None

---

### 11. Mobile Responsiveness
**Status:** 🔴 Desktop Only  
**Complexity:** Medium  
**Estimated Effort:** 3-4 days

#### What Needs to be Done
- Add responsive breakpoints
- Optimize charts for mobile
- Touch-friendly controls
- Test on various devices

**Acceptance Criteria:**
- [ ] Usable on tablets
- [ ] Readable on phones

**Dependencies:** None

---

### 12. Add Data Export Features
**Status:** 🔴 Limited Export  
**Complexity:** Low  
**Estimated Effort:** 1-2 days

#### What Needs to be Done
- Export chart as PNG/SVG
- Export data as CSV/JSON
- Export patterns separately
- Batch export multiple backtests

**Acceptance Criteria:**
- [ ] All export formats work
- [ ] File names are meaningful

**Dependencies:** None

---

### 13. Add User Documentation
**Status:** 🟡 Partial README  
**Complexity:** Low  
**Estimated Effort:** 2-3 days

#### What Needs to be Done
1. **User Guide**
   - How to run the dashboard
   - How to interpret charts
   - Pattern overlay explanation
   - Files: `ui/docs/user_guide.md`

2. **Developer Guide**
   - Architecture overview
   - How to add new overlays
   - How to add new charts
   - Files: `ui/docs/developer_guide.md`

3. **API Documentation**
   - Document all utility functions
   - Document data formats
   - Files: Update docstrings, generate Sphinx docs

**Acceptance Criteria:**
- [ ] New users can run dashboard
- [ ] Developers can extend UI
- [ ] All formats documented

**Dependencies:** None

---

### 14. Automated UI Testing
**Status:** 🔴 No Tests  
**Complexity:** Medium  
**Estimated Effort:** 3-4 days

#### What Needs to be Done

1. **Integration Tests**
   - Test backtest loading
   - Test chart rendering
   - Test pattern overlay loading
   - Files: New `ui/tests/test_integration.py`

2. **Smoke Tests**
   - Test dashboard startup
   - Test navigation
   - Test error handling
   - Files: New `ui/tests/test_smoke.py`

3. **Visual Regression Tests**
   - Capture screenshots
   - Compare to baseline
   - Detect unintended changes
   - Files: New `ui/tests/test_visual.py`

**Acceptance Criteria:**
- [ ] Tests run in CI
- [ ] Coverage >70%
- [ ] Visual regressions caught

**Dependencies:** CI setup

---

### 15. Cleanup and Refactoring
**Status:** 🟡 Ongoing  
**Complexity:** Low  
**Estimated Effort:** 1-2 days

#### What Needs to be Done
- Remove commented code in `app_new.py`
- Split large callback into smaller functions
- Add type hints throughout
- Improve error messages
- Standardize naming conventions

**Acceptance Criteria:**
- [ ] Code passes linting
- [ ] Type checking enabled
- [ ] No commented code

**Dependencies:** None

---

## 📋 Implementation Order Recommendation

### Phase 1: Pattern Visualization (Weeks 1-2)
1. Implement pattern overlay rendering (item #1)
2. Fix timezone handling (item #2)
3. Improve backtest selection (item #3)

### Phase 2: Analysis Tools (Week 3)
4. Add pattern analysis panel (item #4)
5. Add performance tracking (item #5)

### Phase 3: Enhancements (Week 4)
6. Improve chart performance (item #7)
7. Enhance indicator config (item #8)

### Phase 4: Polish (Week 5+)
8. Add comparison mode (item #9)
9. Add export features (item #12)
10. Add documentation (item #13)
11. Add automated testing (item #14)
12. Cleanup and refactoring (item #15)

### Future: Live Trading Support
- Implement real-time monitoring (item #6)
- Add alerting system
- Mobile app

---

## 🔗 External Dependencies
- Pattern detection must be working in algo-main
- Backtest folders must contain `price_actions.json`
- JavaScript lightweight-charts library (already integrated)
- Dash/Plotly framework (already integrated)

## 📝 Notes
- UI should remain responsive even with 100+ patterns
- Prioritize clarity over complexity in visualizations
- All times should respect timezone settings
- Gracefully handle missing data/files
- Test with real backtest data early and often
- Consider accessibility (WCAG standards)
