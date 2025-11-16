import os, json, pandas as pd
from datetime import datetime
from typing import List, Dict, Any

__all__ = ['load_all_json', 'parse_order_events', 'enrich_orders', 'reconstruct_trades', 'build_trade_table', 'build_order_table']

# --- JSON loading ---

def load_all_json(backtest_folder: str) -> Dict[str, Any]:
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

# --- Enriched tables ---

def build_trade_table(trades_df: pd.DataFrame, orders_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    # Expand trades with orderIds details if present
    if 'orderIds' in trades_df.columns:
        # explode orderIds to link orders
        trades_df['orderIds'] = trades_df['orderIds'].apply(lambda x: x if isinstance(x, list) else [])
        exploded = trades_df[['orderIds']].explode('orderIds').rename(columns={'orderIds':'orderId'})
        exploded['orderId'] = exploded['orderId'].astype('Int64')
        merged_orders = exploded.merge(orders_df, on='orderId', how='left', suffixes=('','_order'))
        agg_cols = [c for c in merged_orders.columns if c not in ['orderId']]
        # group back by index (trade row) collecting order detail dicts
        order_details = merged_orders.groupby(level=0).apply(lambda g: g.to_dict('records'))
        trades_df['orderDetails'] = order_details
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
        trades_df[col] = trades_df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict, set, tuple)) else x)
    return trades_df


def build_order_table(orders_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    if orders_df.empty:
        return orders_df
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
    for col in orders_df.columns:
        orders_df[col] = orders_df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict, set, tuple)) else x)
    return orders_df
