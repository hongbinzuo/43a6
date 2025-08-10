import ccxt
import pandas as pd
import numpy as np
import time
from scipy.signal import argrelextrema

# 初始化ccxt连接
exchange = ccxt.gateio()  # 使用Gate.io交易所
symbols = exchange.load_markets()

# Filter for USDT spot pairs
usdt_spot_symbols = [
    symbol for symbol in symbols.keys()
    if symbol.endswith('/USDT') and symbols[symbol].get('spot', False)
    # Exclude leveraged tokens which often end with UP/DOWN and are not standard spot
    and not any(token in symbol for token in ['UP/USDT', 'DOWN/USDT', 'BULL/USDT', 'BEAR/USDT'])
]

# Take the top 500 USDT spot symbols, if available
top_500_symbols = usdt_spot_symbols[:500]

def get_ohlcv(symbol, timeframe='1d', limit=1000):
    """获取OHLCV数据，默认返回1天K线数据"""
    try:
        # Gate.io uses lowercase intervals like '1d', '4h'
        gateio_timeframe = timeframe.lower()
        ohlcv = exchange.fetch_ohlcv(symbol, gateio_timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()

        ohlcv = np.array(ohlcv)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except (ccxt.BadSymbol, ccxt.NetworkError, ccxt.ExchangeError) as e:
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()


def find_swing_points(df, window=5):
    """
    找出价格的摆动高点和低点
    window: 用于确定局部极值的窗口大小
    """
    if df is None or df.empty or len(df) < window * 2:
        return [], []
    
    highs = df['high'].values
    lows = df['low'].values
    
    # 找出局部高点和低点的索引
    high_indices = argrelextrema(highs, np.greater, order=window)[0]
    low_indices = argrelextrema(lows, np.less, order=window)[0]
    
    # 创建摆动点列表，包含索引和值
    swing_highs = [(idx, highs[idx]) for idx in high_indices]
    swing_lows = [(idx, lows[idx]) for idx in low_indices]
    
    return swing_highs, swing_lows


def is_dow_theory_bullish(df, lookback_days=100, min_swing_points=3):
    """
    判断是否符合道氏多头的条件
    需要从最近的低点开始，形成2个更高的高点和2个更高的低点
    
    lookback_days: 回看的天数，用于寻找最近的低点
    min_swing_points: 最少需要的摆动点数量
    """
    if df is None or df.empty or len(df) < 20:
        return False, None

    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN
    df.dropna(subset=['high', 'low'], inplace=True)

    if len(df) < 20:
        return False, None

    # 只分析最近的数据
    recent_df = df.tail(lookback_days).reset_index(drop=True)
    
    # 找出摆动高点和低点
    swing_highs, swing_lows = find_swing_points(recent_df, window=3)
    
    if len(swing_lows) < min_swing_points or len(swing_highs) < min_swing_points:
        return False, None
    
    # 找出最近一段时间内的最低点作为起始点
    # 从后往前找，确保有足够的后续点来形成模式
    valid_start_found = False
    pattern_info = None
    
    # 尝试从不同的低点开始
    for start_idx in range(len(swing_lows) - min_swing_points, -1, -1):
        start_low_idx, start_low_val = swing_lows[start_idx]
        
        # 收集在起始低点之后的所有摆动点
        subsequent_highs = [(idx, val) for idx, val in swing_highs if idx > start_low_idx]
        subsequent_lows = [(idx, val) for idx, val in swing_lows if idx > start_low_idx]
        
        # 需要至少2个后续高点和2个后续低点
        if len(subsequent_highs) >= 2 and len(subsequent_lows) >= 2:
            # 检查是否形成更高的高点序列
            higher_highs = True
            for i in range(1, min(len(subsequent_highs), 3)):
                if subsequent_highs[i][1] <= subsequent_highs[i-1][1]:
                    higher_highs = False
                    break
            
            # 检查是否形成更高的低点序列
            higher_lows = True
            for i in range(min(len(subsequent_lows), 2)):
                if subsequent_lows[i][1] <= start_low_val:
                    higher_lows = False
                    break
            for i in range(1, min(len(subsequent_lows), 2)):
                if subsequent_lows[i][1] <= subsequent_lows[i-1][1]:
                    higher_lows = False
                    break
            
            # 如果找到符合条件的模式
            if higher_highs and higher_lows:
                pattern_info = {
                    'start_low': (start_low_idx, start_low_val),
                    'highs': subsequent_highs[:2],
                    'lows': subsequent_lows[:2],
                    'start_date': recent_df.iloc[start_low_idx]['timestamp'] if 'timestamp' in recent_df.columns else start_low_idx
                }
                valid_start_found = True
                break
    
    return valid_start_found, pattern_info


def scan_top_500_for_bullish():
    """扫描前500个币种，输出符合道氏多头条件的币种"""
    bullish_symbols = []
    total_symbols = len(top_500_symbols)

    for i, symbol in enumerate(top_500_symbols):
        print(f"Scanning {symbol} ({i+1}/{total_symbols})...")
        try:
            # Fetch OHLCV data
            df = get_ohlcv(symbol, timeframe='1d', limit=200)  # 获取更多数据以便分析

            # Ensure DataFrame is valid before proceeding
            if df is None or df.empty or len(df) < 20:
                print(f"  Skipping {symbol}: Not enough valid data or error fetching.")
                continue

            # Check for Dow Theory bullish pattern
            is_bullish, pattern_info = is_dow_theory_bullish(df, lookback_days=100)
            
            if is_bullish:
                bullish_symbols.append(symbol)
                print(f"✅ Found Bullish Pattern: {symbol}")
                if pattern_info:
                    print(f"   Pattern details:")
                    print(f"   - Starting low at index {pattern_info['start_low'][0]}: {pattern_info['start_low'][1]:.4f}")
                    print(f"   - Higher highs: {[f'{h[1]:.4f}' for h in pattern_info['highs']]}")
                    print(f"   - Higher lows: {[f'{l[1]:.4f}' for l in pattern_info['lows']]}")

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

        # Add a delay to avoid hitting rate limits
        time.sleep(0.5)

    return bullish_symbols


# 执行扫描
print("Starting scan for Dow Theory Bullish patterns on Gate.io...")
print("Looking for patterns with 2 higher highs and 2 higher lows from a recent low point...\n")
bullish_symbols = scan_top_500_for_bullish()
print("\nScan complete.")

# Remove duplicates
unique_bullish_symbols = list(set(bullish_symbols))

print("\n符合道氏多头条件的币种（从最近低点产生2高2低）：")
for symbol in unique_bullish_symbols:
    print(f"  - {symbol}")
print(f"\n总计找到 {len(unique_bullish_symbols)} 个符合条件的币种")