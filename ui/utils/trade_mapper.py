import os, json, numpy as np, pandas as pd
from datetime import datetime
from typing import List, Dict, Any

__all__ = ['load_all_json', 'parse_order_events', 'enrich_orders', 'reconstruct_trades', 'build_trade_table', 'build_order_table', 'build_trade_order_table']

ORDER_TYPE_MAP = {
    0: "Market",
    1: "Limit",
    2: "Stop Market",
    3: "Stop Limit",
    4: "Market On Open",
    5: "Market On Close",
    6: "Limit If Touched",
    7: "Stop Market On Close",
    8: "Trailing Stop",
    9: "Combo Market",
    10: "Combo Limit",
    11: "Combo Leg Limit",
}

ORDER_STATUS_MAP = {
    0: "New",
    1: "Submitted",
    2: "Partially Filled",
    3: "Filled",
    4: "Canceled",
    5: "None",
    6: "Invalid",
    7: "Cancel Pending",
    8: "Update Submitted",
    9: "Expired",
}

SECURITY_TYPE_MAP = {
    0: "Base",
    1: "Equity",
    2: "Option",
    3: "Commodity",
    4: "Forex",
    5: "Future",
    6: "CFD",
    7: "Crypto",
    8: "Crypto Future",
    9: "Index",
    10: "Future Option",
    11: "Index Option",
    12: "Commodity Option",
}

ORDER_DIRECTION_MAP = {
    0: "Buy",
    1: "Sell",
    2: "Hold",
}


def _coerce_order_id(value):
    """Convert arbitrary order identifier payloads into integers.

    Parameters
    ----------
    value : Any
        Value extracted from QC order JSON (can be int, float, string, etc.).

    Returns
    -------
    Optional[int]
        Integer representation when coercion succeeds; ``None`` otherwise.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except Exception:
            return None
    return None


def _parse_order_ids(raw_value):
    """Normalize order id fields into a list of integers.

    Parameters
    ----------
    raw_value : Any
        Raw ``orderIds`` payload from trade summaries. Supports single values,
        strings, JSON-encoded lists, or iterables.

    Returns
    -------
    list[int]
        Cleaned list of integer order identifiers.
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        out = []
        for item in raw_value:
            coerced = _coerce_order_id(item)
            if coerced is not None:
                out.append(coerced)
        return out
    if isinstance(raw_value, (int, np.integer, float)):
        coerced = _coerce_order_id(raw_value)
        return [coerced] if coerced is not None else []
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = [part.strip() for part in stripped.split(',') if part.strip()]
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                coerced = _coerce_order_id(item)
                if coerced is not None:
                    out.append(coerced)
            return out
        coerced = _coerce_order_id(parsed)
        return [coerced] if coerced is not None else []
    return []

# --- JSON loading ---

def load_all_json(backtest_folder: str) -> Dict[str, Any]:
    """Read every JSON file in a backtest folder into memory.

    Parameters
    ----------
    backtest_folder : str
        Absolute path to a backtest run directory.

    Returns
    -------
    dict[str, Any]
        Mapping of filename -> parsed JSON object (``None`` on errors).
    """
    out = {}
    for fname in os.listdir(backtest_folder):
        if fname.endswith('.json'):
            path = os.path.join(backtest_folder, fname)
            try:
                out[fname] = json.load(open(path, 'r'))
            except Exception:
                out[fname] = None
    return out

# --- Order events ---

def parse_order_events(json_map: Dict[str, Any]) -> pd.DataFrame:
    """Extract order-event rows from the provided JSON map.

    Parameters
    ----------
    json_map : dict[str, Any]
        Dictionary produced by ``load_all_json``.

    Returns
    -------
    pandas.DataFrame
        DataFrame of order events with an extra ``dt`` column when timestamps
        can be parsed. Empty frame when no order-event file is found.
    """
    # Find first *-order-events.json file
    for name, data in json_map.items():
        if name.endswith('-order-events.json') and isinstance(data, list):
            df = pd.DataFrame(data)
            if 'time' in df.columns:
                try:
                    # assume seconds epoch
                    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
                except Exception:
                    pass
            return df
    return pd.DataFrame()

# --- Orders ---

def enrich_orders(json_map: Dict[str, Any]) -> pd.DataFrame:
    """Load the orders table from summary JSON and expand nested fields.

    Parameters
    ----------
    json_map : dict[str, Any]
        Mapping obtained via ``load_all_json``.

    Returns
    -------
    pandas.DataFrame
        Orders DataFrame sorted by ``orderId``. Empty frame when orders are
        unavailable.
    """
    # Look for file that has 'orders' top-level (summary or main json)
    for name, data in json_map.items():
        if isinstance(data, dict) and 'orders' in data:
            orders = data['orders']
            if isinstance(orders, dict):
                rows = []
                for oid, od in orders.items():
                    r = {'orderId': int(oid)}
                    if isinstance(od, dict):
                        r.update(od)
                    rows.append(r)
                df = pd.DataFrame(rows)
                return df.sort_values('orderId')
    return pd.DataFrame()

# --- Closed trades (original) ---

def reconstruct_trades(json_map: Dict[str, Any]) -> pd.DataFrame:
    """Recover closed trades from QC performance JSON payloads.

    Parameters
    ----------
    json_map : dict[str, Any]
        Result of ``load_all_json``.

    Returns
    -------
    pandas.DataFrame
        Closed trade records with normalized symbol and datetime columns.
    """
    # Prefer summary closedTrades
    trades = []
    for name, data in json_map.items():
        if not isinstance(data, dict):
            continue
        tp = data.get('totalPerformance', {}) or {}
        closed = tp.get('closedTrades') or []
        if closed:
            trades = closed
            break
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    # normalize symbol column
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].apply(lambda v: v.get('permtick') if isinstance(v, dict) and 'permtick' in v else (v.get('value') if isinstance(v, dict) and 'value' in v else v))
    # parse datetimes
    for col in ['entryTime','exitTime']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


def _to_json_safe(value):
    """Convert complex pandas/numpy objects into JSON-serializable values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        try:
            return pd.to_datetime(value).isoformat()
        except Exception:
            return str(value)
    if isinstance(value, (pd.Series, pd.Index)):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, datetime):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    except ValueError:
        try:
            return [_to_json_safe(v) for v in np.asarray(value).tolist()]
        except Exception:
            return str(value)
    return value


def _normalize_table_cell(value):
    """Prepare complex cell values for Dash tables by JSON-encoding them."""
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        safe = _to_json_safe(value)
        return json.dumps(safe)
    if isinstance(value, dict):
        safe = _to_json_safe(value)
        return json.dumps(safe)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        try:
            return pd.to_datetime(value).isoformat()
        except Exception:
            return str(value)
    if isinstance(value, datetime):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


# --- Enriched tables ---

def build_trade_table(trades_df: pd.DataFrame, orders_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a table summarizing trades with aggregated execution context.

    Parameters
    ----------
    trades_df : pandas.DataFrame
        Raw closed trades extracted via ``reconstruct_trades``.
    orders_df : pandas.DataFrame
        Orders DataFrame returned by ``enrich_orders``.
    events_df : pandas.DataFrame
        Order-events DataFrame produced by ``parse_order_events``.

    Returns
    -------
    pandas.DataFrame
        Trade table ready for Dash DataTable consumption.
    """
    if trades_df.empty:
        return trades_df
    trades_df = trades_df.copy()
    # Normalize orderIds so downstream aggregations work reliably
    if 'orderIds' in trades_df.columns:
        normalized_ids = trades_df['orderIds'].apply(_parse_order_ids)
        trades_df['_orderIdsRaw'] = normalized_ids.apply(lambda ids: ids.copy())
        trades_df['orderIds'] = normalized_ids
    # Attach aggregated fees from events if available
    if not events_df.empty:
        # keep only rows with orderId
        if 'orderId' in events_df.columns:
            # determine fill rows (if status exists, use 'filled', else assume all are fills)
            fill_mask = events_df['status'].eq('filled') if 'status' in events_df.columns else pd.Series([True]*len(events_df))
            fills = events_df[fill_mask & events_df['orderId'].notna()].copy()
            # aggregate fee per orderId
            if 'orderFeeAmount' in fills.columns:
                fees_series = fills.groupby('orderId')['orderFeeAmount'].sum()
                fee_map = fees_series.to_dict()
                def compute_total_fees(row):
                    oids = row.get('orderIds', [])
                    return float(sum(fee_map.get(oid, 0) for oid in oids)) if oids else 0.0
                trades_df['eventsFees'] = trades_df.apply(compute_total_fees, axis=1)
            # add per-trade aggregated fill quantity & vwap if possible
            if {'fillQuantity','fillPrice','orderId'} <= set(fills.columns):
                fills['notional'] = fills['fillQuantity'] * fills['fillPrice']
                fill_dict = fills.groupby('orderId').agg(totalQty=('fillQuantity','sum'), totalNotional=('notional','sum'))
                qty_map = fill_dict['totalQty'].to_dict()
                notional_map = fill_dict['totalNotional'].to_dict()
                def compute_fill_stats(row):
                    oids = row.get('orderIds', [])
                    qty = sum(qty_map.get(oid, 0) for oid in oids)
                    notional = sum(notional_map.get(oid, 0) for oid in oids)
                    vwap = (notional / qty) if qty else None
                    return qty, vwap
                stats = trades_df.apply(compute_fill_stats, axis=1)
                trades_df['filledQuantity'] = [s[0] for s in stats]
                trades_df['filledVWAP'] = [s[1] for s in stats]
            # per-side breakdown (buy/sell)
            if 'direction' in fills.columns and 'fillQuantity' in fills.columns and 'fillPrice' in fills.columns:
                def per_side(row):
                    oids = row.get('orderIds', [])
                    sub = fills[fills['orderId'].isin(oids)] if len(oids) else fills.iloc[0:0]
                    if sub.empty:
                        return pd.Series({'sides':'', 'buyQty':0.0, 'sellQty':0.0, 'buyVWAP':None, 'sellVWAP':None})
                    # normalize direction values
                    sub['dir'] = sub['direction'].str.lower()
                    buys = sub[sub['dir'] == 'buy']
                    sells = sub[sub['dir'] == 'sell']
                    buy_qty = buys['fillQuantity'].abs().sum() if not buys.empty else 0.0
                    sell_qty = sells['fillQuantity'].abs().sum() if not sells.empty else 0.0
                    buy_notional = (buys['fillQuantity'].abs() * buys['fillPrice']).sum() if not buys.empty else 0.0
                    sell_notional = (sells['fillQuantity'].abs() * sells['fillPrice']).sum() if not sells.empty else 0.0
                    buy_vwap = (buy_notional / buy_qty) if buy_qty else None
                    sell_vwap = (sell_notional / sell_qty) if sell_qty else None
                    sides = ','.join(sorted(sub['dir'].dropna().unique()))
                    return pd.Series({'sides':sides, 'buyQty':float(buy_qty), 'sellQty':float(sell_qty), 'buyVWAP':buy_vwap, 'sellVWAP':sell_vwap})
                side_df = trades_df.apply(per_side, axis=1)
                for c in side_df.columns:
                    trades_df[c] = side_df[c]
            # realized P&L from fills if not present
            if 'profitLoss' not in trades_df.columns or trades_df['profitLoss'].isna().all():
                def pnl_from_fills(row):
                    oids = row.get('orderIds', [])
                    sub = fills[fills['orderId'].isin(oids)] if len(oids) else fills.iloc[0:0]
                    if sub.empty:
                        return None
                    sub['dir'] = sub['direction'].str.lower() if 'direction' in sub.columns else None
                    # compute signed notional: buy negative, sell positive
                    if 'dir' in sub.columns:
                        sign = sub['dir'].map({'buy':-1,'sell':1}).fillna(0)
                        notional = (sub['fillQuantity'].abs() * sub['fillPrice'] * sign).sum()
                    else:
                        # fallback assume sign from fillQuantity
                        notional = (sub['fillQuantity'] * sub['fillPrice']).sum()
                    fees = row.get('eventsFees', 0) or 0
                    return float(notional) - float(fees)
                trades_df['realizedPnL'] = trades_df.apply(pnl_from_fills, axis=1)
    # Limit prices aggregated from orders if present
    if 'orderIds' in trades_df.columns and not orders_df.empty:
        limit_map = orders_df.set_index('orderId').get('limitPrice') if 'limitPrice' in orders_df.columns else None
        if limit_map is not None:
            def agg_limits(row):
                oids = row.get('orderIds', [])
                vals = [limit_map.get(oid) for oid in oids if oid in limit_map]
                vals = [v for v in vals if pd.notna(v)]
                return ','.join(sorted({str(v) for v in vals})) if vals else ''
            trades_df['limitPrices'] = trades_df.apply(agg_limits, axis=1)
    # Sanitize complex types for Dash DataTable (lists/dicts/etc -> JSON string)
    for col in trades_df.columns:
        trades_df[col] = trades_df[col].apply(_normalize_table_cell)
    if 'orderDetails' in trades_df.columns:
        trades_df = trades_df.drop(columns=['orderDetails'])
    return trades_df


def build_order_table(orders_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Augment the raw orders table with enumerations and event aggregates.

    Parameters
    ----------
    orders_df : pandas.DataFrame
        Orders frame produced by ``enrich_orders``.
    events_df : pandas.DataFrame
        Order events DataFrame for enrichment.

    Returns
    -------
    pandas.DataFrame
        Orders with human-readable enum columns and aggregate metrics.
    """
    if orders_df.empty:
        return orders_df
    orders_df = orders_df.copy()
    # aggregate events per order
    if not events_df.empty:
        # count fills / total quantity filled
        if 'fillQuantity' in events_df.columns:
            fill_qty = events_df.groupby('orderId')['fillQuantity'].sum().rename('filledQuantity')
            orders_df = orders_df.merge(fill_qty, on='orderId', how='left')
        # last fill time
        if 'dt' in events_df.columns:
            last_fill = events_df.groupby('orderId')['dt'].max().rename('lastEventTime')
            orders_df = orders_df.merge(last_fill, on='orderId', how='left')

    def _map_enum(value, mapping):
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        try:
            if isinstance(value, str) and value.isdigit():
                key = int(value)
            else:
                key = int(value)
        except Exception:
            return value
        return mapping.get(key, str(value))

    if 'type' in orders_df.columns:
        orders_df['type'] = orders_df['type'].apply(lambda v: _map_enum(v, ORDER_TYPE_MAP))
    if 'status' in orders_df.columns:
        orders_df['status'] = orders_df['status'].apply(lambda v: _map_enum(v, ORDER_STATUS_MAP))
    if 'securityType' in orders_df.columns:
        orders_df['securityType'] = orders_df['securityType'].apply(lambda v: _map_enum(v, SECURITY_TYPE_MAP))
    if 'direction' in orders_df.columns:
        orders_df['direction'] = orders_df['direction'].apply(lambda v: _map_enum(v, ORDER_DIRECTION_MAP))

    if 'symbol' in orders_df.columns:
        def _symbol_value(val):
            if isinstance(val, dict):
                return val.get('value') or val.get('Value') or val.get('permtick') or json.dumps(val)
            return val
        orders_df['symbol'] = orders_df['symbol'].apply(_symbol_value)

    if 'tag' in orders_df.columns:
        orders_df = orders_df.drop(columns=['tag'])

    for col in orders_df.columns:
        orders_df[col] = orders_df[col].apply(_normalize_table_cell)
    return orders_df


def build_trade_order_table(trades_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """Combine trades and orders into a flattened trade-order join table.

    Parameters
    ----------
    trades_df : pandas.DataFrame
        Trade table returned by ``build_trade_table``.
    orders_df : pandas.DataFrame
        Enriched order table from ``build_order_table``.

    Returns
    -------
    pandas.DataFrame
        Row-per-trade-order pairing, suitable for drilling into execution
        history within Dash tables.
    """
    if trades_df.empty:
        return pd.DataFrame()

    trade_records = trades_df.to_dict('records')
    if not trade_records:
        return pd.DataFrame()

    order_lookup: Dict[Any, Dict[str, Any]] = {}
    if not orders_df.empty:
        for record in orders_df.to_dict('records'):
            key = record.get('orderId')
            candidates = [key]
            coerced = _coerce_order_id(key)
            if coerced is not None:
                candidates.extend([coerced, str(coerced)])
            if isinstance(key, str):
                trimmed = key.strip()
                if trimmed and trimmed not in candidates:
                    candidates.append(trimmed)
            for cand in candidates:
                if cand in (None, ''):
                    continue
                order_lookup[cand] = record

    trade_field_map = [
        ('symbol', 'tradeSymbol'),
        ('direction', 'tradeDirection'),
        ('entryTime', 'tradeEntryTime'),
        ('exitTime', 'tradeExitTime'),
        ('quantity', 'tradeQuantity'),
        ('profitLoss', 'tradeProfitLoss'),
        ('realizedPnL', 'tradeRealizedPnL'),
        ('eventsFees', 'tradeFees'),
        ('filledQuantity', 'tradeFilledQuantity'),
        ('filledVWAP', 'tradeFilledVWAP'),
        ('buyQty', 'tradeBuyQty'),
        ('sellQty', 'tradeSellQty'),
        ('buyVWAP', 'tradeBuyVWAP'),
        ('sellVWAP', 'tradeSellVWAP'),
        ('limitPrices', 'tradeLimitPrices'),
        ('sides', 'tradeSides'),
    ]
    order_field_map = [
        ('status', 'orderStatus'),
        ('type', 'orderType'),
        ('direction', 'orderDirection'),
        ('quantity', 'orderQuantity'),
        ('filledQuantity', 'orderFilledQuantity'),
        ('limitPrice', 'orderLimitPrice'),
        ('stopPrice', 'orderStopPrice'),
        ('lastEventTime', 'orderLastEventTime'),
        ('lastFillTime', 'orderLastFillTime'),
        ('time', 'orderTime'),
        ('createdTime', 'orderCreatedTime'),
        ('securityType', 'orderSecurityType'),
        ('symbol', 'orderSymbol'),
    ]

    combined_rows: List[Dict[str, Any]] = []
    for index, trade in enumerate(trade_records, start=1):
        raw_ids = trade.get('_orderIdsRaw', trade.get('orderIds'))
        order_ids = _parse_order_ids(raw_ids)
        total_orders = len(order_ids)
        if not order_ids:
            row = {
                'tradeIndex': index,
                'orderSequence': None,
                'orderCount': 0,
                'orderId': None
            }
            for source, target in trade_field_map:
                if source in trade:
                    row[target] = trade[source]
            combined_rows.append(row)
            continue
        for sequence, order_id in enumerate(order_ids, start=1):
            order_data = order_lookup.get(order_id) or order_lookup.get(str(order_id))
            row = {
                'tradeIndex': index,
                'orderSequence': sequence,
                'orderCount': total_orders,
            }
            for source, target in trade_field_map:
                if source in trade:
                    row[target] = trade[source]
            if order_data:
                row['orderId'] = order_data.get('orderId', order_id)
                for source, target in order_field_map:
                    if source in order_data:
                        row[target] = order_data[source]
            else:
                row['orderId'] = order_id
            combined_rows.append(row)

    if not combined_rows:
        return pd.DataFrame()

    combined_df = pd.DataFrame(combined_rows)
    if not combined_df.empty:
        combined_df = combined_df.sort_values(['tradeIndex', 'orderSequence'], na_position='last').reset_index(drop=True)
        for col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(_normalize_table_cell)
        preferred_order = [
            'tradeIndex',
            'orderSequence',
            'orderCount',
            'tradeSymbol',
            'tradeDirection',
            'tradeEntryTime',
            'tradeExitTime',
            'tradeQuantity',
            'tradeProfitLoss',
            'tradeRealizedPnL',
            'tradeFees',
            'tradeFilledQuantity',
            'tradeFilledVWAP',
            'tradeLimitPrices',
            'tradeSides',
            'tradeBuyQty',
            'tradeBuyVWAP',
            'tradeSellQty',
            'tradeSellVWAP',
            'orderId',
            'orderStatus',
            'orderType',
            'orderDirection',
            'orderQuantity',
            'orderFilledQuantity',
            'orderLimitPrice',
            'orderStopPrice',
            'orderLastEventTime',
            'orderLastFillTime',
            'orderTime',
            'orderCreatedTime',
            'orderSecurityType',
            'orderSymbol',
        ]
        existing = [c for c in preferred_order if c in combined_df.columns]
        remaining = [c for c in combined_df.columns if c not in existing]
        combined_df = combined_df[existing + remaining]
    return combined_df
