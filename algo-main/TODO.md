# algo-main TODO List

## 🎯 CRITICAL PRIORITY (Blockers)

### 1. Complete Double Top Pattern Implementation
**Status:** 🔴 In Progress  
**Complexity:** High  
**Estimated Effort:** 5-7 days

#### Current State
- ✅ `DoubleTopDetector` class created with ZigZag/ATR-based pivot detection
- ✅ Basic pattern recognition (Peak1 → Trough → Peak2)
- ✅ Neckline break confirmation logic
- ✅ Retest detection capability
- ✅ Volume and RSI divergence tracking
- ⚠️ Missing: Causal smoothing (Savitzky-Golay)
- ⚠️ Missing: Prominence calculation
- ⚠️ Missing: Symmetry scoring
- ⚠️ Missing: DTW template matching
- ⚠️ Missing: Percentile-based threshold calibration

#### What Needs to be Done
1. **Implement Causal Smoothing**
   - Add one-sided Savitzky-Golay filter (no scipy dependency)
   - Implement adaptive window sizing based on peak spacing
   - Preserve geometry while removing noise
   - Files: `utils/price_action_patterns.py`

2. **Add Prominence Calculation**
   - Implement prominence metric: peak height relative to surrounding valleys
   - Normalize by ATR: `prom_atr = prominence / ATR`
   - Add minimum prominence threshold filter
   - Files: `utils/pattern_detectors/double_top_detector.py`

3. **Implement Symmetry Scoring**
   - Time symmetry: `1 - |dt1 - dt2| / (dt1 + dt2)`
   - Price symmetry: balance of peak heights relative to neckline
   - Add configurable minimum symmetry threshold
   - Files: `utils/pattern_detectors/double_top_detector.py`

4. **Add DTW Shape Validation** (Optional but Recommended)
   - Create canonical M-shape template
   - Implement lightweight DTW distance calculation (no scipy)
   - Compute distance quantiles from historical patterns
   - Accept patterns with q ≤ 0.05
   - Files: New `utils/dtw_matcher.py` + integration in detector

5. **Percentile-Based Calibration System**
   - Create calibration mode that collects distributions:
     - Peak spacing (Δt)
     - Height differences (hdiff_atr)
     - Trough depth (depth_atr)
     - Prominence (prom_atr)
     - Symmetry scores
   - Compute percentile thresholds:
     - min_dt = p5(Δt)
     - max_dt = p95(Δt)
     - eq_peak_k = p33(hdiff_atr)
     - min_depth = p40(depth_atr)
     - min_prom = p50(prom_atr)
     - min_sym = p30(sym)
   - Save thresholds to `config.json` under `pattern_calibration`
   - Files: `cli/calibrate_patterns.py` + `utils/pattern_detectors/double_top_detector.py`

**Acceptance Criteria:**
- [ ] Smoothing preserves peaks and reduces noise
- [ ] Prominence filtering removes insignificant peaks
- [ ] Symmetry scoring identifies clean vs messy patterns
- [ ] DTW rejects shapes that don't match M-pattern
- [ ] Calibration produces instrument-adaptive thresholds
- [ ] Unit tests pass for all features

**Dependencies:** None

---

### 2. Add Comprehensive Unit Tests
**Status:** 🔴 Not Started  
**Complexity:** Medium  
**Estimated Effort:** 3-4 days

#### What Needs to be Done
1. **Create Test Infrastructure**
   - Set up pytest structure in `tests/` directory
   - Create fixtures for synthetic TradeBar generation
   - Add helper functions for pattern validation
   - Files: `tests/conftest.py`, `tests/fixtures.py`

2. **Test DoubleTopDetector**
   - Test pivot detection (peaks and troughs)
   - Test potential pattern formation
   - Test neckline break confirmation
   - Test retest detection
   - Test invalidation scenarios (new high above peak1)
   - Test edge cases (insufficient data, flat price, extreme volatility)
   - Files: `tests/test_double_top_detector.py`

3. **Test PriceActionPatterns**
   - Test window management (deque vs RollingWindow)
   - Test `is_ready()` logic
   - Test `calculate_double_tops()` integration
   - Test logging and flush mechanism
   - Files: `tests/test_price_action_patterns.py`

4. **Test Integration**
   - Mock algorithm environment
   - Test pattern detection in full backtest flow
   - Verify log file creation and JSON structure
   - Files: `tests/test_integration.py`

**Acceptance Criteria:**
- [ ] All tests pass with >90% code coverage
- [ ] Tests are deterministic and reproducible
- [ ] Edge cases documented and handled
- [ ] Tests run in <30 seconds

**Dependencies:** Complete item #1 (at least core detection)

---

### 3. Wire Patterns to Strategy and Alpha Models
**Status:** 🟡 Partially Done  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Current State
- ✅ `PriceActionPatterns` instantiated in `main.py`
- ✅ Basic pattern check in `mean_reversion.py`
- ⚠️ Missing: Proper signal extraction and filtering
- ⚠️ Missing: Pattern quality score integration
- ⚠️ Missing: Multiple pattern handling

#### What Needs to be Done
1. **Enhance Strategy Integration**
   - Extract pattern signals with quality scores
   - Filter patterns by minimum quality threshold
   - Handle multiple concurrent patterns (ranking)
   - Add pattern age/staleness filtering
   - Files: `strategies/mean_reversion.py`

2. **Add Pattern-Based Entry Logic**
   - Entry on neckline break with confirmation
   - Optional: Wait for retest before entry
   - Stop-loss at pattern height + buffer
   - Take-profit at projected target (height of pattern)
   - Files: `strategies/mean_reversion.py`

3. **Logging and Debugging**
   - Log pattern detections with full metadata
   - Track pattern success rate (entry → profit/loss)
   - Add debug mode for pattern visualization
   - Files: `strategies/mean_reversion.py`

**Acceptance Criteria:**
- [ ] Strategy responds to confirmed patterns
- [ ] Entry/exit logic uses pattern geometry
- [ ] Pattern logs include timestamps, prices, scores
- [ ] Backtest shows pattern-based trades

**Dependencies:** Item #1 (core detection), Item #2 (tests for validation)

---

## 🔥 HIGH PRIORITY

### 4. Complete CandlestickPatternManager Implementation
**Status:** 🟡 Partially Implemented  
**Complexity:** High  
**Estimated Effort:** 4-5 days

#### Current State
- ✅ Basic structure and initialization
- ✅ Log directory resolution via `resolve_log_dir`
- ⚠️ Missing: `_build_detectors()` implementation
- ⚠️ Missing: `ensure_timeframe()` consolidator setup
- ⚠️ Missing: `_on_consolidated()` handler
- ⚠️ Missing: `get_recent_signals()` and filtering
- ⚠️ Missing: `apply_filter_config()` and `filter_passes()`

#### What Needs to be Done
1. **Implement Detector Building**
   - Parse detector configuration from config
   - Instantiate detector classes dynamically
   - Map detector names to pattern types
   - Files: `utils/candlestick_pattern.py`

2. **Implement Consolidation Logic**
   - Create consolidators for different timeframes
   - Subscribe to consolidated bars
   - Route bars to appropriate detectors
   - Handle multiple timeframes simultaneously
   - Files: `utils/candlestick_pattern.py`

3. **Implement Signal Management**
   - Store recent signals with metadata
   - Filter by time window
   - Filter by direction (bullish/bearish)
   - Filter by pattern type
   - Implement `has_pattern()` query method
   - Files: `utils/candlestick_pattern.py`

4. **Implement Configuration System**
   - Parse filter configuration from JSON
   - Apply filters to pattern detections
   - Support dynamic filter updates
   - Files: `utils/candlestick_pattern.py`

**Acceptance Criteria:**
- [ ] Manager detects candlestick patterns on consolidated bars
- [ ] Signals are queryable and filterable
- [ ] Configuration changes update filters dynamically
- [ ] Unit tests verify all methods

**Dependencies:** None (parallel with pattern work)

---

### 5. Create Calibration Utility
**Status:** 🔴 Not Started  
**Complexity:** Medium  
**Estimated Effort:** 3-4 days

#### What Needs to be Done
1. **Build CLI Tool**
   - Create `cli/calibrate_patterns.py`
   - Accept backtest folder or CSV data as input
   - Scan historical data for candidate patterns
   - Collect distribution statistics
   - Files: `cli/calibrate_patterns.py`

2. **Statistical Analysis**
   - Compute distributions for all metrics
   - Calculate percentile thresholds
   - Validate threshold stability across windows
   - Generate calibration report
   - Files: `cli/calibrate_patterns.py`

3. **Configuration Output**
   - Save thresholds to JSON format
   - Support multiple instruments/timeframes
   - Include calibration metadata (date, sample size)
   - Merge with existing config
   - Files: Output to `config.json` or `pattern_thresholds.json`

4. **Integration with Detector**
   - Load thresholds on startup
   - Override default values
   - Validate threshold sanity
   - Files: `utils/pattern_detectors/double_top_detector.py`

**Acceptance Criteria:**
- [ ] CLI runs successfully on sample data
- [ ] Thresholds are instrument-specific
- [ ] Detector loads and applies thresholds
- [ ] Documentation explains calibration process

**Dependencies:** Item #1 (pattern detection must work first)

---

## 📊 MEDIUM PRIORITY

### 6. Implement Double Bottom Detector
**Status:** 🔴 Scaffold Only  
**Complexity:** Medium  
**Estimated Effort:** 3-4 days

#### Current State
- ✅ Scaffold file created: `utils/pattern_detectors/double_bottom_detector.py`
- ⚠️ No logic implemented

#### What Needs to be Done
1. **Mirror Double Top Logic**
   - Invert ZigZag trend direction (start DOWN)
   - Detect TROUGH → PEAK → TROUGH sequences
   - Apply same ATR normalization
   - Add neckline break confirmation (upward)
   - Add retest detection (downward)
   - Files: `utils/pattern_detectors/double_bottom_detector.py`

2. **Integrate with PriceActionPatterns**
   - Add `calculate_double_bottoms()` method
   - Wire detector updates
   - Log pattern detections
   - Files: `utils/price_action_patterns.py`

3. **Add Tests**
   - Copy and adapt double top tests
   - Test inverted pattern logic
   - Files: `tests/test_double_bottom_detector.py`

**Acceptance Criteria:**
- [ ] Detector identifies W-shaped patterns
- [ ] Confirmation and retest logic works
- [ ] Tests pass with good coverage
- [ ] Logs include double bottom patterns

**Dependencies:** Item #1 (learn from double top implementation)

---

### 7. Implement Head and Shoulders Detector
**Status:** 🔴 Scaffold Only  
**Complexity:** High  
**Estimated Effort:** 5-6 days

#### Current State
- ✅ Scaffold file created: `utils/pattern_detectors/head_and_shoulders_detector.py`
- ⚠️ No logic implemented

#### What Needs to be Done
1. **Design Pattern Logic**
   - Detect 5-pivot sequence: LEFT_SHOULDER → HEAD → RIGHT_SHOULDER
   - Validate head is higher than both shoulders
   - Validate shoulders at similar heights
   - Compute neckline (line through troughs)
   - Files: `utils/pattern_detectors/head_and_shoulders_detector.py`

2. **Implement Detection**
   - Use same ZigZag pivot framework
   - Add shoulder symmetry validation
   - Add head prominence requirement
   - Add neckline slope tolerance
   - Files: `utils/pattern_detectors/head_and_shoulders_detector.py`

3. **Integration and Testing**
   - Add `calculate_head_and_shoulders()` method
   - Create comprehensive tests
   - Files: `utils/price_action_patterns.py`, `tests/test_head_and_shoulders_detector.py`

**Acceptance Criteria:**
- [ ] Detector identifies 5-pivot H&S patterns
- [ ] Shoulder symmetry validated
- [ ] Neckline calculation correct
- [ ] Tests include complex scenarios

**Dependencies:** Item #1 and #6 (leverage existing patterns)

---

### 8. Add Visualization/Debug Utilities
**Status:** 🔴 Not Started  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### What Needs to be Done
1. **Create Plot Utilities**
   - Function to plot price with detected pivots
   - Overlay pattern annotations (peaks, troughs, neckline)
   - Show ATR bands
   - Color-code pattern states
   - Files: New `utils/pattern_visualizer.py`

2. **Jupyter Notebook Examples**
   - Load backtest data
   - Run pattern detection
   - Visualize results
   - Compare patterns to actual trades
   - Files: New `research/pattern_visualization.ipynb`

3. **CLI Visualization Tool**
   - Accept backtest folder as input
   - Generate PNG/HTML charts
   - Save to backtest folder
   - Files: `cli/visualize_patterns.py`

**Acceptance Criteria:**
- [ ] Charts clearly show pattern detection
- [ ] Notebook runs end-to-end
- [ ] CLI generates useful outputs
- [ ] Documented with examples

**Dependencies:** Item #1, #2, #3 (patterns must work first)

---

## 🛠️ LOW PRIORITY / ENHANCEMENTS

### 9. Optimize Performance
**Status:** 🔴 Not Started  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### What Needs to be Done
- Profile pattern detection performance
- Optimize rolling window operations
- Cache ATR and smoothed values
- Minimize allocations in hot path
- Add performance benchmarks

**Acceptance Criteria:**
- [ ] Pattern detection adds <1ms per bar overhead
- [ ] Memory usage stable over long backtests

**Dependencies:** All core features complete

---

### 10. Documentation and Examples
**Status:** 🟡 Partial (double-top.md exists)  
**Complexity:** Low  
**Estimated Effort:** 2-3 days

#### What Needs to be Done
1. **Complete API Documentation**
   - Docstrings for all public methods
   - Type hints everywhere
   - Usage examples in docstrings
   - Files: All detector and pattern files

2. **Create User Guide**
   - How to enable pattern detection
   - Configuration options explained
   - Interpretation of pattern signals
   - Files: New `docs/pattern_detection_guide.md`

3. **Add Example Strategies**
   - Simple pattern-based strategy
   - Pattern filtering examples
   - Multi-pattern combination
   - Files: New `strategies/pattern_example.py`

**Acceptance Criteria:**
- [ ] All code has proper docstrings
- [ ] Guide covers common use cases
- [ ] Examples run successfully

**Dependencies:** Core features complete

---

### 11. Add Continuous Integration
**Status:** 🔴 Not Started  
**Complexity:** Low  
**Estimated Effort:** 1-2 days

#### What Needs to be Done
- Create GitHub Actions workflow
- Run pytest on every commit
- Add linting (flake8, black)
- Add type checking (mypy)
- Generate coverage reports

**Acceptance Criteria:**
- [ ] CI runs on push and PR
- [ ] Tests must pass to merge
- [ ] Coverage tracked over time

**Dependencies:** Item #2 (tests must exist)

---

### 12. Cleanup and Refactoring
**Status:** 🔴 Not Started  
**Complexity:** Low  
**Estimated Effort:** 1-2 days

#### What Needs to be Done
- Remove `pass` placeholders in `main.py`
- Add type hints where missing
- Fix any pylint warnings
- Remove commented-out code
- Standardize naming conventions
- Add `__all__` exports where needed

**Acceptance Criteria:**
- [ ] No `pass` statements in production code
- [ ] Linter shows no warnings
- [ ] Type hints complete

**Dependencies:** None (ongoing maintenance)

---

## 📋 Implementation Order Recommendation

### Phase 1: Core Detection (Weeks 1-2)
1. Complete double top implementation (item #1)
2. Add comprehensive tests (item #2)
3. Wire to strategy (item #3)

### Phase 2: Supporting Infrastructure (Week 3)
4. Finish candlestick manager (item #4)
5. Create calibration utility (item #5)

### Phase 3: Additional Patterns (Week 4)
6. Implement double bottom (item #6)
7. Begin head and shoulders (item #7)

### Phase 4: Polish (Week 5+)
8. Add visualization (item #8)
9. Optimize performance (item #9)
10. Complete documentation (item #10)
11. Setup CI (item #11)
12. Cleanup refactoring (item #12)

---

## 🔗 External Dependencies
- Ensure ATR indicator is properly configured and warm
- Verify `RollingWindow` behavior matches expectations
- Confirm log directory creation permissions

## 📝 Notes
- Prioritize quality over speed for pattern detection
- False positives are worse than false negatives for retail trading
- Keep all logic causal (no lookahead bias)
- Document all threshold choices and their rationale
- Test with multiple instruments and timeframes
