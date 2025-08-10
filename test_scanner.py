import ccxt
import pandas as pd
import numpy as np
import time

# 初始化ccxt连接
print("Initializing Gate.io connection...")
exchange = ccxt.gateio()  # 使用Gate.io交易所
symbols = exchange.load_markets()

# Filter for USDT spot pairs
usdt_spot_symbols = [
    symbol for symbol in symbols.keys()
    if symbol.endswith('/USDT') and symbols[symbol].get('spot', False)
    # Exclude leveraged tokens which often end with UP/DOWN and are not standard spot
    and not any(token in symbol for token in ['UP/USDT', 'DOWN/USDT', 'BULL/USDT', 'BEAR/USDT'])
]

print(f"Found {len(usdt_spot_symbols)} USDT spot pairs on Gate.io")

# Select specific symbols for testing
test_symbols = ['DENT/USDT', 'BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'SHIB/USDT']
# Filter to only include symbols that exist
test_symbols = [s for s in test_symbols if s in usdt_spot_symbols]

print(f"Testing with {len(test_symbols)} symbols: {test_symbols}")

def get_ohlcv(symbol, timeframe='1d', limit=1000):
    """获取OHLCV数据，默认返回1天K线数据"""
    try:
        # Gate.io uses lowercase intervals like '1d', '4h'
        gateio_timeframe = timeframe.lower()
        ohlcv = exchange.fetch_ohlcv(symbol, gateio_timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()

        ohlcv = np.array(ohlcv)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime for better readability
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except (ccxt.BadSymbol, ccxt.NetworkError, ccxt.ExchangeError) as e:
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()


def find_swing_points_simple(df, window=5):
    """
    找出价格的摆动高点和低点（不依赖scipy）
    window: 用于确定局部极值的窗口大小
    """
    if df is None or df.empty or len(df) < window * 2 + 1:
        return [], []
    
    highs = df['high'].values
    lows = df['low'].values
    
    swing_highs = []
    swing_lows = []
    
    # 找局部高点
    for i in range(window, len(highs) - window):
        # 检查是否是局部最高点
        is_high = True
        for j in range(i - window, i + window + 1):
            if j != i and highs[j] >= highs[i]:
                is_high = False
                break
        if is_high:
            swing_highs.append((i, highs[i]))
    
    # 找局部低点
    for i in range(window, len(lows) - window):
        # 检查是否是局部最低点
        is_low = True
        for j in range(i - window, i + window + 1):
            if j != i and lows[j] <= lows[i]:
                is_low = False
                break
        if is_low:
            swing_lows.append((i, lows[i]))
    
    return swing_highs, swing_lows


def is_dow_theory_bullish_strict(df, lookback_days=150):
    """
    严格判断是否符合道氏多头的条件
    需要从最近的一个明显低点开始，形成至少2个更高的高点和2个更高的低点
    
    lookback_days: 回看的天数
    """
    if df is None or df.empty or len(df) < 30:
        return False, None

    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN
    df = df.dropna(subset=['high', 'low'])

    if len(df) < 30:
        return False, None

    # 只分析最近的数据
    recent_df = df.tail(lookback_days).reset_index(drop=True)
    
    # 使用较小的窗口找出更多的潜在转折点
    swing_highs, swing_lows = find_swing_points_simple(recent_df, window=3)
    
    if len(swing_lows) < 3 or len(swing_highs) < 2:
        return False, None
    
    # 寻找符合条件的模式
    best_pattern = None
    
    # 从倒数第3个低点开始往前找（确保后面有足够的点）
    for start_idx in range(max(0, len(swing_lows) - 5), len(swing_lows) - 2):
        start_low_idx, start_low_val = swing_lows[start_idx]
        
        # 收集在起始低点之后的摆动点
        subsequent_highs = [(idx, val) for idx, val in swing_highs if idx > start_low_idx]
        subsequent_lows = [(idx, val) for idx, val in swing_lows if idx > start_low_idx]
        
        # 需要至少2个高点和2个低点
        if len(subsequent_highs) >= 2 and len(subsequent_lows) >= 2:
            # 取前2个高点和前2个低点
            check_highs = subsequent_highs[:2]
            check_lows = subsequent_lows[:2]
            
            # 检查高点是否递增（形成更高的高点）
            highs_ascending = check_highs[1][1] > check_highs[0][1]
            
            # 检查低点是否递增（形成更高的低点）
            lows_ascending = (check_lows[0][1] > start_low_val and 
                            check_lows[1][1] > check_lows[0][1])
            
            # 额外检查：确保第二个低点也高于起始低点
            second_low_higher = check_lows[1][1] > start_low_val
            
            if highs_ascending and lows_ascending and second_low_higher:
                # 计算涨幅以选择最佳模式
                rise_pct = ((check_highs[-1][1] - start_low_val) / start_low_val) * 100
                
                pattern_info = {
                    'start_low': (start_low_idx, start_low_val),
                    'highs': check_highs,
                    'lows': check_lows,
                    'rise_pct': rise_pct,
                    'start_date': recent_df.iloc[start_low_idx]['timestamp'] if start_low_idx < len(recent_df) else None
                }
                
                # 选择涨幅最大或最近的模式
                if best_pattern is None or pattern_info['start_low'][0] > best_pattern['start_low'][0]:
                    best_pattern = pattern_info
    
    return best_pattern is not None, best_pattern


def test_specific_symbol(symbol):
    """测试特定币种是否符合条件"""
    print(f"\n{'='*60}")
    print(f"Testing {symbol}...")
    print("-" * 60)
    
    df = get_ohlcv(symbol, timeframe='1d', limit=200)
    
    if df is None or df.empty:
        print(f"Failed to fetch data for {symbol}")
        return False
    
    print(f"Fetched {len(df)} days of data")
    
    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get current price
    current_price = df['close'].iloc[-1]
    print(f"Current price: {current_price:.6f}")
    
    # Test with detailed output
    is_bullish, pattern_info = is_dow_theory_bullish_strict(df, lookback_days=150)
    
    if is_bullish:
        print(f"\n✅ {symbol} shows a BULLISH Dow Theory pattern!")
        if pattern_info:
            print(f"\nPattern Details:")
            print(f"  Starting Low: {pattern_info['start_low'][1]:.6f} at index {pattern_info['start_low'][0]}")
            print(f"  Date: {pattern_info['start_date'].strftime('%Y-%m-%d') if pattern_info['start_date'] is not None else 'N/A'}")
            print(f"\n  2 Higher Highs:")
            for i, (idx, val) in enumerate(pattern_info['highs'], 1):
                print(f"    High {i}: {val:.6f} at index {idx}")
            print(f"\n  2 Higher Lows (after starting low):")
            for i, (idx, val) in enumerate(pattern_info['lows'], 1):
                print(f"    Low {i}: {val:.6f} at index {idx}")
            print(f"\n  Total Rise from starting low: {pattern_info['rise_pct']:.2f}%")
            print(f"  Current price vs starting low: {((current_price - pattern_info['start_low'][1]) / pattern_info['start_low'][1] * 100):.2f}%")
        return True
    else:
        print(f"\n❌ {symbol} does NOT show a clear Dow Theory bullish pattern")
        print(f"   (Need 2 higher highs and 2 higher lows from a recent low point)")
        
        # Show the swing points found for debugging
        recent_df = df.tail(150).reset_index(drop=True)
        swing_highs, swing_lows = find_swing_points_simple(recent_df, window=3)
        
        print(f"\n  Debug Info:")
        print(f"    Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
        
        if len(swing_lows) > 0:
            print(f"    Last 3 swing lows: ")
            for idx, val in swing_lows[-3:]:
                print(f"      Index {idx}: {val:.6f}")
        if len(swing_highs) > 0:
            print(f"    Last 3 swing highs: ")
            for idx, val in swing_highs[-3:]:
                print(f"      Index {idx}: {val:.6f}")
        return False


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("Dow Theory Bullish Pattern Scanner for Gate.io")
    print("Looking for: 2 Higher Highs + 2 Higher Lows from recent low")
    print("=" * 60)
    
    bullish_symbols = []
    
    # Test each symbol
    for symbol in test_symbols:
        try:
            is_bullish = test_specific_symbol(symbol)
            if is_bullish:
                bullish_symbols.append(symbol)
            # Add delay to avoid rate limits
            time.sleep(0.5)
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SCAN SUMMARY")
    print("=" * 60)
    
    if bullish_symbols:
        print(f"\n✅ Found {len(bullish_symbols)} symbols with Dow Theory Bullish Pattern:")
        for symbol in bullish_symbols:
            print(f"  - {symbol}")
    else:
        print("\n❌ No symbols found with clear Dow Theory Bullish Pattern")
    
    print(f"\nTotal symbols tested: {len(test_symbols)}")
    print(f"Bullish patterns found: {len(bullish_symbols)}")
    print(f"Success rate: {(len(bullish_symbols)/len(test_symbols)*100):.1f}%")
    
    print("\n✨ Test complete!")