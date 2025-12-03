(function (global) {
    'use strict';

    var LIB_URL = 'https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js';
    var COLOR_PALETTE = ['#089981', '#f97316', '#1d4ed8', '#10b981', '#d946ef', '#ef4444', '#3b82f6', '#6366f1', '#22c55e', '#fb7185'];
    var SUPER_TREND_COLORS = {
        upper: '#ef4444',
        lower: '#22c55e'
    };
    var VWAP_COLOR = '#000000';

    var CONTROL_STYLE_ID = 'tv-lite-controls-styles';

    function hashString(str) {
        var hash = 0;
        if (!str) {
            return hash;
        }
        for (var i = 0; i < str.length; i += 1) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash |= 0; // Convert to 32bit integer
        }
        return hash;
    }

    function injectControlStyles() {
        if (document.getElementById(CONTROL_STYLE_ID)) {
            return;
        }
        var style = document.createElement('style');
        style.id = CONTROL_STYLE_ID;
        style.textContent = '' +
            '.tv-lite-controls{position:absolute;left:50%;bottom:18px;transform:translateX(-50%);' +
            'display:flex;gap:6px;padding:6px 8px;background:rgba(17,24,39,0.75);' +
            'border-radius:8px;box-shadow:0 12px 24px -16px rgba(15,23,42,0.55);z-index:40;}' +
            '.tv-lite-controls button{border:0;border-radius:6px;background:#1f2937;color:#f8fafc;' +
            'padding:6px 10px;font-size:12px;font-weight:600;cursor:pointer;line-height:1;' +
            'transition:background 0.2s ease, transform 0.2s ease;}' +
            '.tv-lite-controls button:hover{background:#334155;transform:translateY(-1px);}' +
            '.tv-lite-controls button:active{transform:translateY(0);background:#0f172a;}' +
            '.tv-lite-controls button:disabled{opacity:0.45;cursor:default;transform:none;background:#1f2937;}';
        document.head.appendChild(style);
    }

    function createControlButton(label, title, handler) {
        var button = document.createElement('button');
        button.type = 'button';
        button.textContent = label;
        button.title = title;
        button.addEventListener('click', function (event) {
            event.preventDefault();
            if (typeof handler === 'function') {
                handler();
            }
        });
        return button;
    }

    function attachControls(container, instance) {
        if (!container || !instance) {
            return null;
        }
        injectControlStyles();
        var toolbar = document.createElement('div');
        toolbar.className = 'tv-lite-controls';
        container.appendChild(toolbar);

        function adjustZoom(factorMultiplier) {
            if (!instance.chart || !instance.chart.timeScale) {
                return;
            }
            var scale = instance.chart.timeScale();
            if (!scale || typeof scale.getVisibleLogicalRange !== 'function') {
                return;
            }
            var range = scale.getVisibleLogicalRange();
            if (!range || typeof range.from !== 'number' || typeof range.to !== 'number') {
                return;
            }
            var length = range.to - range.from;
            if (!isFinite(length) || length <= 0) {
                return;
            }
            var midpoint = range.from + length / 2;
            var nextLength = length * factorMultiplier;
            if (!isFinite(nextLength) || nextLength <= 0) {
                return;
            }
            var nextRange = { from: midpoint - nextLength / 2, to: midpoint + nextLength / 2 };
            if (typeof scale.setVisibleLogicalRange === 'function') {
                scale.setVisibleLogicalRange(nextRange);
            }
        }

        function scrollBy(percent) {
            if (!instance.chart || !instance.chart.timeScale) {
                return;
            }
            var scale = instance.chart.timeScale();
            if (!scale || typeof scale.getVisibleLogicalRange !== 'function') {
                return;
            }
            var range = scale.getVisibleLogicalRange();
            if (!range || typeof range.from !== 'number' || typeof range.to !== 'number') {
                return;
            }
            var length = range.to - range.from;
            if (!isFinite(length) || length <= 0) {
                return;
            }
            var shift = length * percent;
            var nextRange = { from: range.from + shift, to: range.to + shift };
            if (typeof scale.setVisibleLogicalRange === 'function') {
                scale.setVisibleLogicalRange(nextRange);
            }
        }

        var buttons = [
            {
                label: '-',
                title: 'Zoom Out',
                handler: function () { adjustZoom(1.35); }
            },
            {
                label: '+',
                title: 'Zoom In',
                handler: function () { adjustZoom(0.7); }
            },
            {
                label: '<',
                title: 'Scroll Left',
                handler: function () { scrollBy(-0.35); }
            },
            {
                label: '>',
                title: 'Scroll Right',
                handler: function () { scrollBy(0.35); }
            },
            {
                label: 'Reset',
                title: 'Fit Visible Range',
                handler: function () {
                    if (instance && instance.chart && typeof instance.chart.timeScale === 'function') {
                        var scale = instance.chart.timeScale();
                        if (scale && typeof scale.fitContent === 'function') {
                            scale.fitContent();
                        }
                    }
                }
            }
        ];

        buttons.forEach(function (spec) {
            toolbar.appendChild(createControlButton(spec.label, spec.title, spec.handler));
        });

        instance.__controlsToolbar = toolbar;
        return toolbar;
    }

    function colorFor(name) {
        if (!COLOR_PALETTE.length) {
            return '#1d4ed8';
        }
        var idx = Math.abs(hashString(String(name || ''))) % COLOR_PALETTE.length;
        return COLOR_PALETTE[idx];
    }

    function defaultLineOptions(name) {
        var upperName = String(name || '');
        var upperNameUpper = upperName.toUpperCase();
        var fallbackColor = upperNameUpper.indexOf('VWAP') === 0 ? VWAP_COLOR : colorFor(upperName);
        return {
            color: fallbackColor,
            lineWidth: 2,
            lineStyle: undefined,
            priceLineVisible: false,
            lastValueVisible: false
        };
    }

    function mergeLineOptions(raw, fallback) {
        var config = {
            data: [],
            color: fallback && typeof fallback.color === 'string' ? fallback.color : colorFor(''),
            lineWidth: fallback && typeof fallback.lineWidth === 'number' ? fallback.lineWidth : 2,
            lineStyle: fallback ? fallback.lineStyle : undefined,
            priceLineVisible: typeof (fallback && fallback.priceLineVisible) === 'boolean' ? fallback.priceLineVisible : false,
            lastValueVisible: typeof (fallback && fallback.lastValueVisible) === 'boolean' ? fallback.lastValueVisible : false
        };
        if (Array.isArray(raw)) {
            config.data = raw;
            return config;
        }
        if (!raw || typeof raw !== 'object') {
            return config;
        }
        if (Array.isArray(raw.data)) {
            config.data = raw.data;
        }
        if (typeof raw.color === 'string') {
            config.color = raw.color;
        }
        if (typeof raw.lineWidth === 'number') {
            config.lineWidth = raw.lineWidth;
        }
        if (typeof raw.lineStyle === 'number') {
            config.lineStyle = raw.lineStyle;
        }
        if (typeof raw.priceLineVisible === 'boolean') {
            config.priceLineVisible = raw.priceLineVisible;
        }
        if (typeof raw.lastValueVisible === 'boolean') {
            config.lastValueVisible = raw.lastValueVisible;
        }
        return config;
    }

    function extractSeriesOptions(config) {
        var options = {
            color: config.color,
            lineWidth: config.lineWidth,
            priceLineVisible: typeof config.priceLineVisible === 'boolean' ? config.priceLineVisible : false,
            lastValueVisible: typeof config.lastValueVisible === 'boolean' ? config.lastValueVisible : false
        };
        if (typeof config.lineStyle === 'number') {
            options.lineStyle = config.lineStyle;
        }
        return options;
    }

    function safeRemoveSeries(chart, series) {
        if (!series) {
            return;
        }
        try {
            if (chart && typeof chart.removeSeries === 'function') {
                chart.removeSeries(series);
            } else if (typeof series.remove === 'function') {
                series.remove();
            } else if (typeof series.destroy === 'function') {
                series.destroy();
            }
        } catch (err) {
            console.warn('[TradingViewPriceVolume] Failed to remove series', err);
        }
    }

    function ensureLibrary() {
        if (global.LightweightCharts) {
            return Promise.resolve(global.LightweightCharts);
        }
        if (global.__tradingViewLibraryPromise) {
            return global.__tradingViewLibraryPromise;
        }
        global.__tradingViewLibraryPromise = new Promise(function (resolve, reject) {
            var script = document.createElement('script');
            script.src = LIB_URL;
            script.async = true;
            script.onload = function () {
                if (global.LightweightCharts) {
                    resolve(global.LightweightCharts);
                } else {
                    reject(new Error('LightweightCharts did not initialize.'));
                }
            };
            script.onerror = function () {
                reject(new Error('Failed to load LightweightCharts from ' + LIB_URL));
            };
            document.head.appendChild(script);
        });
        return global.__tradingViewLibraryPromise;
    }

    function mountChart(container, props) {
        var instance = {
            chart: null,
            candleSeries: null,
            volumeSeries: null,
            overlayLines: {},
            supertrendSeries: {},
            markersAttached: false
        };
        return ensureLibrary().then(function (lightweight) {
            if (!lightweight || typeof lightweight.createChart !== 'function') {
                throw new Error('LightweightCharts library missing createChart');
            }
            var options = props && props.chartOptions ? props.chartOptions : {};
            instance.chart = lightweight.createChart(container, options);
            instance.library = lightweight;
            attachControls(container, instance);

            var candlesOptions = props && props.candlesOptions ? props.candlesOptions : {};
            if (typeof instance.chart.addCandlestickSeries === 'function') {
                instance.candleSeries = instance.chart.addCandlestickSeries(candlesOptions);
            } else if (typeof instance.chart.addSeries === 'function') {
                var candleType = (lightweight.SeriesType && lightweight.SeriesType.Candlestick) ? lightweight.SeriesType.Candlestick : 'Candlestick';
                try {
                    instance.candleSeries = instance.chart.addSeries(candleType, candlesOptions);
                } catch (err) {
                    var candleConfig = Object.assign({ type: candleType }, candlesOptions);
                    try {
                        instance.candleSeries = instance.chart.addSeries(candleConfig);
                    } catch (err2) {
                        throw err2 || err;
                    }
                }
            } else {
                throw new Error('LightweightCharts version does not support candlestick series');
            }

            var baseVolumeOptions = Object.assign({
                priceFormat: { type: 'volume' }
            }, props && props.volumeOptions ? props.volumeOptions : {});
            if (typeof instance.chart.addHistogramSeries === 'function') {
                var legacyOptions = Object.assign({ priceScaleId: 'volume' }, baseVolumeOptions);
                instance.volumeSeries = instance.chart.addHistogramSeries(legacyOptions);
                instance.chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
            } else if (typeof instance.chart.addSeries === 'function') {
                var histType = (lightweight.SeriesType && lightweight.SeriesType.Histogram) ? lightweight.SeriesType.Histogram : 'Histogram';
                try {
                    instance.volumeSeries = instance.chart.addSeries(histType, baseVolumeOptions);
                } catch (err3) {
                    var volumeConfig = Object.assign({ type: histType }, baseVolumeOptions);
                    try {
                        instance.volumeSeries = instance.chart.addSeries(volumeConfig);
                    } catch (err4) {
                        throw err4 || err3;
                    }
                }
                if (instance.volumeSeries && typeof instance.volumeSeries.priceScale === 'function') {
                    instance.volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
                }
            } else {
                throw new Error('LightweightCharts version does not support histogram series');
            }
            updateSeries(instance, props || {});
            handleResize(container, instance);
            return instance;
        });
    }

    function updateSeries(instance, props) {
        if (!instance || !instance.candleSeries || !instance.volumeSeries) {
            return;
        }
        var candleData = Array.isArray(props.candles) ? props.candles : [];
        var volumeData = Array.isArray(props.volume) ? props.volume : [];
        instance.candleSeries.setData(candleData);
        instance.volumeSeries.setData(volumeData);
        if (Array.isArray(props.markers) && props.markers.length) {
            instance.candleSeries.setMarkers(props.markers);
            instance.markersAttached = true;
        } else if (instance.markersAttached) {
            instance.candleSeries.setMarkers([]);
            instance.markersAttached = false;
        }
        var overlays = normalizeOverlayPayload(props && props.overlays);
        syncOverlays(instance, overlays);
    }

    function syncOverlays(instance, overlays) {
        if (!instance || !instance.chart) {
            return;
        }
        var overlayPayload = (overlays && typeof overlays === 'object') ? overlays : {};
        syncOverlayLines(instance, overlayPayload.lines);
        syncSupertrend(instance, overlayPayload.supertrend);
    }

    function ensureLineSeries(instance, options) {
        if (!instance || !instance.chart) {
            return null;
        }
        var chart = instance.chart;
        var opts = Object.assign({
            priceLineVisible: false,
            lastValueVisible: false
        }, options || {});
        if (typeof chart.addLineSeries === 'function') {
            return chart.addLineSeries(opts);
        }
        if (typeof chart.addSeries === 'function') {
            var library = instance.library || global.LightweightCharts;
            var lineType = (library && library.SeriesType && library.SeriesType.Line) ? library.SeriesType.Line : 'Line';
            try {
                return chart.addSeries(lineType, opts);
            } catch (err) {
                var config = Object.assign({ type: lineType }, opts);
                return chart.addSeries(config);
            }
        }
        return null;
    }

    function syncOverlayLines(instance, linesPayload) {
        var chart = instance && instance.chart;
        if (!chart) {
            return;
        }
        var payload = (linesPayload && typeof linesPayload === 'object') ? linesPayload : {};
        if (!instance.overlayLines) {
            instance.overlayLines = {};
        }
        var existing = instance.overlayLines;
        var incomingKeys = Object.keys(payload);

        Object.keys(existing).forEach(function (name) {
            if (incomingKeys.indexOf(name) === -1) {
                safeRemoveSeries(chart, existing[name] && existing[name].series);
                delete existing[name];
                return;
            }
            var fallback = defaultLineOptions(name);
            var normalized = mergeLineOptions(payload[name], fallback);
            if (!normalized.data.length) {
                safeRemoveSeries(chart, existing[name] && existing[name].series);
                delete existing[name];
            }
        });

        incomingKeys.forEach(function (name) {
            var fallback = defaultLineOptions(name);
            var normalized = mergeLineOptions(payload[name], fallback);
            if (!normalized.data.length) {
                return;
            }
            var entry = existing[name];
            var options = extractSeriesOptions(normalized);
            if (!entry || !entry.series) {
                var series = ensureLineSeries(instance, options);
                if (!series) {
                    return;
                }
                entry = { series: series };
                existing[name] = entry;
            }
            if (entry.series && typeof entry.series.setData === 'function') {
                entry.series.setData(normalized.data);
            }
            if (entry.series && typeof entry.series.applyOptions === 'function') {
                entry.series.applyOptions(options);
            }
        });
    }

    function syncSupertrend(instance, supertrendPayload) {
        var chart = instance && instance.chart;
        if (!chart) {
            return;
        }
        var payload = (supertrendPayload && typeof supertrendPayload === 'object') ? supertrendPayload : {};
        if (!instance.supertrendSeries) {
            instance.supertrendSeries = {};
        }
        var existing = instance.supertrendSeries;
        var incomingKeys = Object.keys(payload);

        Object.keys(existing).forEach(function (name) {
            if (incomingKeys.indexOf(name) === -1) {
                removeSupertrendEntry(chart, existing[name]);
                delete existing[name];
            }
        });

        incomingKeys.forEach(function (name) {
            var data = payload[name] || {};
            var upperConfig = mergeLineOptions(data.upper, {
                color: SUPER_TREND_COLORS.upper,
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
                lastValueVisible: false
            });
            var lowerConfig = mergeLineOptions(data.lower, {
                color: SUPER_TREND_COLORS.lower,
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
                lastValueVisible: false
            });
            var hasAny = upperConfig.data.length || lowerConfig.data.length;
            var entry = existing[name];

            if (!hasAny) {
                if (entry) {
                    removeSupertrendEntry(chart, entry);
                    delete existing[name];
                }
                return;
            }

            if (!entry) {
                entry = {};
                existing[name] = entry;
            }

            if (upperConfig.data.length) {
                if (!entry.upper) {
                    entry.upper = ensureLineSeries(instance, extractSeriesOptions(upperConfig));
                }
                if (entry.upper && typeof entry.upper.setData === 'function') {
                    entry.upper.setData(upperConfig.data);
                }
                if (entry.upper && typeof entry.upper.applyOptions === 'function') {
                    entry.upper.applyOptions(extractSeriesOptions(upperConfig));
                }
            } else if (entry.upper) {
                safeRemoveSeries(chart, entry.upper);
                delete entry.upper;
            }

            if (lowerConfig.data.length) {
                if (!entry.lower) {
                    entry.lower = ensureLineSeries(instance, extractSeriesOptions(lowerConfig));
                }
                if (entry.lower && typeof entry.lower.setData === 'function') {
                    entry.lower.setData(lowerConfig.data);
                }
                if (entry.lower && typeof entry.lower.applyOptions === 'function') {
                    entry.lower.applyOptions(extractSeriesOptions(lowerConfig));
                }
            } else if (entry.lower) {
                safeRemoveSeries(chart, entry.lower);
                delete entry.lower;
            }

            if (!entry.upper && !entry.lower) {
                delete existing[name];
            }
        });
    }

    function removeSupertrendEntry(chart, entry) {
        if (!entry) {
            return;
        }
        safeRemoveSeries(chart, entry.upper);
        safeRemoveSeries(chart, entry.lower);
    }

    function handleResize(container, instance) {
        if (!container || !instance || !instance.chart) {
            return;
        }
        function resizeChart() {
            var width = container.clientWidth;
            var height = container.clientHeight;
            if (width && height) {
                instance.chart.resize(width, height);
            }
        }
        instance.__resizeObserver = new ResizeObserver(function () {
            resizeChart();
        });
        instance.__resizeObserver.observe(container);
        resizeChart();
    }

    function destroyChart(instance) {
        if (!instance) {
            return;
        }
        if (instance.__resizeObserver && instance.__resizeObserver.disconnect) {
            instance.__resizeObserver.disconnect();
        }
        var chart = instance.chart;
        if (instance.overlayLines) {
            Object.keys(instance.overlayLines).forEach(function (name) {
                var entry = instance.overlayLines[name];
                if (entry && entry.series) {
                    safeRemoveSeries(chart, entry.series);
                }
            });
            instance.overlayLines = {};
        }
        if (instance.supertrendSeries) {
            Object.keys(instance.supertrendSeries).forEach(function (name) {
                removeSupertrendEntry(chart, instance.supertrendSeries[name]);
            });
            instance.supertrendSeries = {};
        }
        if (instance.chart) {
            instance.chart.remove();
        }
        if (instance.__controlsToolbar && instance.__controlsToolbar.parentNode) {
            instance.__controlsToolbar.parentNode.removeChild(instance.__controlsToolbar);
        }
        instance.__controlsToolbar = null;
    }

    global.TradingViewPriceVolume = {
        mount: mountChart,
        update: updateSeries,
        destroy: destroyChart
    };

    var WATCHED_ATTRS = ['data-last-render', 'data-candles', 'data-volume', 'data-overlays', 'data-markers'];
    var registry = new Map();

    function parseAttribute(element, attr, fallback) {
        var raw = element.getAttribute(attr);
        if (!raw || !raw.length) {
            return fallback;
        }
        try {
            return JSON.parse(raw);
        } catch (err) {
            console.warn('[TradingViewPriceVolume] Failed to parse', attr, err);
            return fallback;
        }
    }

    function normalizeOverlayPayload(raw) {
        if (!raw || typeof raw !== 'object') {
            return { lines: {}, supertrend: {} };
        }
        var lines = (raw.lines && typeof raw.lines === 'object') ? raw.lines : {};
        var supertrend = (raw.supertrend && typeof raw.supertrend === 'object') ? raw.supertrend : {};
        var legend = Array.isArray(raw.legend) ? raw.legend : [];
        return { lines: lines, supertrend: supertrend, legend: legend };
    }

    function readProps(element) {
        return {
            candles: parseAttribute(element, 'data-candles', []),
            volume: parseAttribute(element, 'data-volume', []),
            overlays: normalizeOverlayPayload(parseAttribute(element, 'data-overlays', { lines: {}, supertrend: {} })),
            markers: parseAttribute(element, 'data-markers', []),
            chartOptions: parseAttribute(element, 'data-chart-options', {})
        };
    }

    function createAttributeObserver(element, entry) {
        var observer = new MutationObserver(function (mutations) {
            var shouldUpdate = false;
            for (var i = 0; i < mutations.length; i += 1) {
                var mutation = mutations[i];
                if (mutation.type === 'attributes' && WATCHED_ATTRS.indexOf(mutation.attributeName) !== -1) {
                    shouldUpdate = true;
                    break;
                }
            }
            if (shouldUpdate && entry.instance) {
                try {
                    var props = readProps(element);
                    updateSeries(entry.instance, props);
                } catch (err) {
                    console.error('[TradingViewPriceVolume] Failed to update series', err);
                }
            }
        });
        observer.observe(element, { attributes: true, attributeFilter: WATCHED_ATTRS });
        return observer;
    }

    function connectElement(element) {
        if (!element || registry.has(element)) {
            var existing = registry.get(element);
            if (existing && existing.instance) {
                var props = readProps(element);
                updateSeries(existing.instance, props);
            }
            return;
        }

        var entry = { instance: null, observer: null };
        registry.set(element, entry);

        mountChart(element, readProps(element))
            .then(function (instance) {
                entry.instance = instance;
                entry.observer = createAttributeObserver(element, entry);
                try {
                    updateSeries(instance, readProps(element));
                } catch (err) {
                    console.error('[TradingViewPriceVolume] Failed to apply initial data', err);
                }
            })
            .catch(function (error) {
                console.error('[TradingViewPriceVolume] Mount error:', error);
                registry.delete(element);
            });
    }

    function disconnectElement(element) {
        if (!element || !registry.has(element)) {
            return;
        }
        var entry = registry.get(element);
        if (entry.observer && entry.observer.disconnect) {
            entry.observer.disconnect();
        }
        if (entry.instance) {
            destroyChart(entry.instance);
        }
        registry.delete(element);
    }

    function scanNode(node, callback) {
        if (!(node instanceof HTMLElement)) {
            return;
        }
        if (node.dataset && node.dataset.component === 'price-volume-chart') {
            callback(node);
        }
        var descendants = node.querySelectorAll ? node.querySelectorAll('[data-component="price-volume-chart"]') : [];
        for (var i = 0; i < descendants.length; i += 1) {
            callback(descendants[i]);
        }
    }

    function boot() {
        var elements = document.querySelectorAll('[data-component="price-volume-chart"]');
        for (var i = 0; i < elements.length; i += 1) {
            connectElement(elements[i]);
        }

        var bodyObserver = new MutationObserver(function (mutations) {
            for (var m = 0; m < mutations.length; m += 1) {
                var mutation = mutations[m];
                for (var a = 0; a < mutation.addedNodes.length; a += 1) {
                    scanNode(mutation.addedNodes[a], connectElement);
                }
                for (var r = 0; r < mutation.removedNodes.length; r += 1) {
                    scanNode(mutation.removedNodes[r], disconnectElement);
                }
            }
        });

        bodyObserver.observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})(window);
