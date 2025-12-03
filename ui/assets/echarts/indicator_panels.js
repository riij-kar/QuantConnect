(function (global) {
    'use strict';

    var LIB_URL = 'https://cdn.jsdelivr.net/npm/echarts@6.0.0/dist/echarts.min.js';
    var PANEL_DEFAULT_HEIGHT = 240;

    function ensureLibrary() {
        if (global.echarts) {
            return Promise.resolve(global.echarts);
        }
        if (global.__echartsLibraryPromise) {
            return global.__echartsLibraryPromise;
        }
        global.__echartsLibraryPromise = new Promise(function (resolve, reject) {
            var script = document.createElement('script');
            script.src = LIB_URL;
            script.async = true;
            script.onload = function () {
                if (global.echarts) {
                    resolve(global.echarts);
                } else {
                    reject(new Error('ECharts did not load.'));
                }
            };
            script.onerror = function () {
                reject(new Error('Failed to load ECharts from ' + LIB_URL));
            };
            document.head.appendChild(script);
        });
        return global.__echartsLibraryPromise;
    }

    function parseAttribute(element, attr, fallback) {
        var raw = element.getAttribute(attr);
        if (!raw) {
            return fallback;
        }
        try {
            return JSON.parse(raw);
        } catch (err) {
            console.warn('[EChartsPanel] Failed to parse', attr, err);
            return fallback;
        }
    }

    function slugify(text) {
        if (!text) {
            return 'panel';
        }
        return String(text)
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-+|-+$/g, '') || 'panel';
    }

    function normalizeConfig(config) {
        var normalized = {
            panels: [],
            messages: Array.isArray(config && config.messages) ? config.messages.slice() : [],
            meta: config && typeof config.meta === 'object' ? config.meta : {}
        };
        if (!config || typeof config !== 'object') {
            return normalized;
        }

        var panelHeight = config.panelHeight && Number.isFinite(config.panelHeight) ? config.panelHeight : PANEL_DEFAULT_HEIGHT;

        if (Array.isArray(config.oscillators)) {
            config.oscillators.forEach(function (panel) {
                if (!panel || !Array.isArray(panel.series) || !panel.series.length) {
                    return;
                }
                normalized.panels.push({
                    id: panel.id || slugify(panel.title || 'oscillator'),
                    title: panel.title || 'Oscillator',
                    series: panel.series,
                    legend: Array.isArray(panel.legend) ? panel.legend : [],
                    levels: Array.isArray(panel.levels) ? panel.levels : [],
                    height: Number.isFinite(panel.height) ? panel.height : panelHeight,
                    kind: panel.kind || 'oscillator',
                    yAxis: panel.yAxis,
                    grid: panel.grid,
                    xAxis: panel.xAxis,
                    legendConfig: panel.legendConfig,
                    backgroundColor: panel.backgroundColor
                });
            });
        }

        var performance = config.performance && typeof config.performance === 'object' ? config.performance : {};
        function pushPerformance(id, title, entry) {
            if (!entry || !Array.isArray(entry.series) || !entry.series.length) {
                return;
            }
            normalized.panels.push({
                id: entry.id || 'performance-' + id,
                title: entry.title || title,
                series: entry.series,
                legend: Array.isArray(entry.legend) ? entry.legend : [],
                height: Number.isFinite(entry.height) ? entry.height : panelHeight,
                kind: id,
                yAxis: entry.yAxis,
                grid: entry.grid,
                xAxis: entry.xAxis,
                legendConfig: entry.legendConfig,
                backgroundColor: entry.backgroundColor
            });
        }
        pushPerformance('equity', 'Equity Curve', performance.equity);
        pushPerformance('returns', 'Return %', performance.returns);
        pushPerformance('drawdown', 'Drawdown', performance.drawdown);

        if (Array.isArray(config.analytics)) {
            config.analytics.forEach(function (panel) {
                if (!panel || !Array.isArray(panel.series) || !panel.series.length) {
                    return;
                }
                normalized.panels.push({
                    id: panel.id || slugify(panel.title || 'analytics'),
                    title: panel.title || 'Analytics',
                    series: panel.series,
                    legend: Array.isArray(panel.legend) ? panel.legend : [],
                    height: Number.isFinite(panel.height) ? panel.height : panelHeight,
                    kind: panel.kind || 'analytics',
                    yAxis: panel.yAxis,
                    grid: panel.grid,
                    xAxis: panel.xAxis,
                    legendConfig: panel.legendConfig,
                    backgroundColor: panel.backgroundColor
                });
            });
        }

        return normalized;
    }

    function buildSeries(seriesSpec) {
        var base = {
            name: seriesSpec.name || 'Series',
            connectNulls: true
        };
        if (seriesSpec.type === 'bar') {
            var positive = seriesSpec.positiveColor || '#16a34a';
            var negative = seriesSpec.negativeColor || '#dc2626';
            base.type = 'bar';
            base.barWidth = seriesSpec.barWidth || '60%';
            if (seriesSpec.barMinWidth) {
                base.barMinWidth = seriesSpec.barMinWidth;
            }
            if (seriesSpec.stack) {
                base.stack = seriesSpec.stack;
            }
            if (seriesSpec.barGap) {
                base.barGap = seriesSpec.barGap;
            }
            if (seriesSpec.opacity !== undefined) {
                base.itemStyle = base.itemStyle || {};
                base.itemStyle.opacity = seriesSpec.opacity;
            }
            base.data = (seriesSpec.data || []).map(function (point) {
                if (!Array.isArray(point) || point.length < 2) {
                    var simpleStyle = { color: positive };
                    if (seriesSpec.opacity !== undefined) {
                        simpleStyle.opacity = seriesSpec.opacity;
                    }
                    return { value: point, itemStyle: simpleStyle };
                }
                var value = Number(point[1]);
                var color = value >= 0 ? positive : negative;
                var style = { color: color };
                if (seriesSpec.opacity !== undefined) {
                    style.opacity = seriesSpec.opacity;
                }
                return { value: [point[0], value], itemStyle: style };
            });
            if (seriesSpec.markLine) {
                base.markLine = seriesSpec.markLine;
            }
        } else {
            base.type = 'line';
            base.showSymbol = false;
            base.lineStyle = { width: 2 };
            if (typeof seriesSpec.showSymbol === 'boolean') {
                base.showSymbol = seriesSpec.showSymbol;
            }
            if (seriesSpec.symbol) {
                base.symbol = seriesSpec.symbol;
            }
            if (seriesSpec.smooth !== undefined) {
                base.smooth = !!seriesSpec.smooth;
            }
            if (seriesSpec.color) {
                base.lineStyle.color = seriesSpec.color;
                base.itemStyle = { color: seriesSpec.color };
            }
            if (seriesSpec.lineStyle && typeof seriesSpec.lineStyle === 'object') {
                base.lineStyle = Object.assign({}, base.lineStyle, seriesSpec.lineStyle);
            }
            if (seriesSpec.area) {
                base.areaStyle = { opacity: 0.15 };
            }
            if (seriesSpec.yAxisIndex !== undefined) {
                base.yAxisIndex = seriesSpec.yAxisIndex;
            }
            if (seriesSpec.markLine) {
                base.markLine = seriesSpec.markLine;
            }
            base.data = Array.isArray(seriesSpec.data) ? seriesSpec.data : [];
        }
        if (seriesSpec.connectNulls !== undefined) {
            base.connectNulls = !!seriesSpec.connectNulls;
        }
        return base;
    }

    function buildOption(panel) {
        var option = {
            animation: false,
            tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
            legend: { data: [] },
            grid: { left: 50, right: 20, top: 40, bottom: 40, containLabel: true },
            xAxis: { type: 'time', boundaryGap: false, axisLabel: { hideOverlap: true } },
            yAxis: { type: 'value', scale: true },
            series: []
        };

        if (panel.backgroundColor) {
            option.backgroundColor = panel.backgroundColor;
        }

        if (Array.isArray(panel.legend) && panel.legend.length) {
            option.legend.data = panel.legend.map(function (entry) { return entry.label || entry; });
            if (panel.legendConfig && typeof panel.legendConfig === 'object') {
                option.legend = Object.assign({}, option.legend, panel.legendConfig);
            }
        } else {
            option.legend.show = false;
        }

        if (panel.grid && typeof panel.grid === 'object') {
            option.grid = Object.assign({}, option.grid, panel.grid);
        }

        if (panel.xAxis && typeof panel.xAxis === 'object') {
            option.xAxis = Object.assign({}, option.xAxis, panel.xAxis);
        }

        if (Array.isArray(panel.yAxis)) {
            option.yAxis = panel.yAxis.map(function (axis) {
                return Object.assign({ type: 'value', scale: true }, axis);
            });
        } else if (panel.yAxis && typeof panel.yAxis === 'object') {
            option.yAxis = Object.assign({ type: 'value', scale: true }, panel.yAxis);
        }

        option.series = (panel.series || []).map(buildSeries);

        if (panel.markLine && option.series.length) {
            option.series[0].markLine = Object.assign({}, option.series[0].markLine || {}, panel.markLine);
        }

        if (Array.isArray(panel.levels) && panel.levels.length) {
            var targetSeries = option.series.find(function (item) { return item.type === 'line'; });
            if (targetSeries) {
                targetSeries.markLine = {
                    symbol: 'none',
                    silent: true,
                    label: { formatter: function (params) { return params.name || params.value; } },
                    lineStyle: { type: 'dashed', color: '#94a3b8' },
                    data: panel.levels.map(function (level) {
                        return { yAxis: level.value, name: level.label };
                    })
                };
            }
        }

        return option;
    }

    function buildPanelBlocks(element, normalizedConfig) {
        element.innerHTML = '';
        var blocks = [];

        if (normalizedConfig.messages.length) {
            var notice = document.createElement('div');
            notice.className = 'echarts-panel-messages';
            notice.style.margin = '0 0 12px 0';
            normalizedConfig.messages.forEach(function (msg) {
                var row = document.createElement('div');
                row.textContent = msg;
                row.style.fontSize = '12px';
                row.style.color = '#8c6d1f';
                notice.appendChild(row);
            });
            element.appendChild(notice);
        }

        if (!normalizedConfig.panels.length) {
            var empty = document.createElement('div');
            empty.textContent = 'No indicator data available.';
            empty.style.color = '#666';
            empty.style.fontSize = '13px';
            element.appendChild(empty);
            return blocks;
        }

        normalizedConfig.panels.forEach(function (panel) {
            var block = document.createElement('div');
            block.className = 'echarts-panel-block';
            block.style.marginBottom = '18px';

            var title = document.createElement('div');
            title.className = 'echarts-panel-title';
            title.textContent = panel.title || 'Panel';
            title.style.fontWeight = '600';
            title.style.fontSize = '14px';
            title.style.margin = '0 0 6px 0';
            block.appendChild(title);

            var chartHost = document.createElement('div');
            chartHost.className = 'echarts-panel-chart';
            chartHost.style.width = '100%';
            chartHost.style.height = (panel.height || PANEL_DEFAULT_HEIGHT) + 'px';
            block.appendChild(chartHost);

            element.appendChild(block);
            blocks.push({ panel: panel, chartHost: chartHost });
        });

        return blocks;
    }

    function renderPanel(element, entry, config) {
        if (!entry || !entry.library) {
            return;
        }

        if (entry.charts && entry.charts.length) {
            entry.charts.forEach(function (chart) {
                if (chart && typeof chart.dispose === 'function') {
                    chart.dispose();
                }
            });
        }
        entry.charts = [];

        var normalized = normalizeConfig(config);
        var blocks = buildPanelBlocks(element, normalized);
        blocks.forEach(function (block) {
            var chart = entry.library.init(block.chartHost, null, { renderer: 'canvas' });
            var option = buildOption(block.panel);
            chart.setOption(option, true);
            entry.charts.push(chart);
        });

        if (entry.resizeObserver && entry.resizeObserver.disconnect) {
            entry.resizeObserver.disconnect();
        }
        entry.resizeObserver = new ResizeObserver(function () {
            if (entry.charts) {
                entry.charts.forEach(function (chart) {
                    if (chart && typeof chart.resize === 'function') {
                        chart.resize();
                    }
                });
            }
        });
        entry.resizeObserver.observe(element);
    }

    function connectElement(element) {
        if (!element) {
            return;
        }
        var existing = registry.get(element);
        if (existing) {
            var config = parseAttribute(element, 'data-config', {});
            renderPanel(element, existing, config);
            return;
        }

        var entry = { charts: [], resizeObserver: null, library: null };
        registry.set(element, entry);

        ensureLibrary()
            .then(function (lib) {
                entry.library = lib;
                var config = parseAttribute(element, 'data-config', {});
                renderPanel(element, entry, config);
                entry.observer = new MutationObserver(function (mutations) {
                    for (var i = 0; i < mutations.length; i += 1) {
                        var mutation = mutations[i];
                        if (mutation.type === 'attributes' && WATCHED_ATTRS.indexOf(mutation.attributeName) !== -1) {
                            var nextConfig = parseAttribute(element, 'data-config', {});
                            renderPanel(element, entry, nextConfig);
                            break;
                        }
                    }
                });
                entry.observer.observe(element, { attributes: true, attributeFilter: WATCHED_ATTRS });
            })
            .catch(function (err) {
                console.error('[EChartsPanel] Mount error:', err);
                registry.delete(element);
            });
    }

    function disconnectElement(element) {
        var entry = registry.get(element);
        if (!entry) {
            return;
        }
        if (entry.observer && entry.observer.disconnect) {
            entry.observer.disconnect();
        }
        if (entry.resizeObserver && entry.resizeObserver.disconnect) {
            entry.resizeObserver.disconnect();
        }
        if (entry.charts && entry.charts.length) {
            entry.charts.forEach(function (chart) {
                if (chart && typeof chart.dispose === 'function') {
                    chart.dispose();
                }
            });
        }
        registry.delete(element);
    }

    var WATCHED_ATTRS = ['data-config', 'data-last-render'];
    var registry = new Map();

    function scanNode(node, callback) {
        if (!(node instanceof HTMLElement)) {
            return;
        }
        if (node.dataset && node.dataset.component === 'echarts-panel') {
            callback(node);
        }
        var descendants = node.querySelectorAll ? node.querySelectorAll('[data-component="echarts-panel"]') : [];
        for (var i = 0; i < descendants.length; i += 1) {
            callback(descendants[i]);
        }
    }

    function boot() {
        var elements = document.querySelectorAll('[data-component="echarts-panel"]');
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
