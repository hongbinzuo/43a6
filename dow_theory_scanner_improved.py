import ccxt
import pandas as pd
import numpy as np
import time

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


def find_pivot_points(df, left_bars=5, right_bars=5):
    """
    使用更严格的方法找出关键的转折点（枢轴点）
    left_bars: 左侧需要比较的K线数量
    right_bars: 右侧需要比较的K线数量
    """
    if df is None or df.empty or len(df) < left_bars + right_bars + 1:
        return [], []
    
    highs = df['high'].values
    lows = df['low'].values
    
    pivot_highs = []
    pivot_lows = []
    
    for i in range(left_bars, len(df) - right_bars):
        # 检查是否为枢轴高点
        is_pivot_high = True
        pivot_high_value = highs[i]
        
        # 检查左侧
        for j in range(i - left_bars, i):
            if highs[j] > pivot_high_value:
                is_pivot_high = False
                break
        
        # 检查右侧
        if is_pivot_high:
            for j in range(i + 1, i + right_bars + 1):
                if j < len(highs) and highs[j] > pivot_high_value:
                    is_pivot_high = False
                    break
        
        if is_pivot_high:
            pivot_highs.append((i, pivot_high_value))
        
        # 检查是否为枢轴低点
        is_pivot_low = True
        pivot_low_value = lows[i]
        
        # 检查左侧
        for j in range(i - left_bars, i):
            if lows[j] < pivot_low_value:
                is_pivot_low = False
                break
        
        # 检查右侧
        if is_pivot_low:
            for j in range(i + 1, i + right_bars + 1):
                if j < len(lows) and lows[j] < pivot_low_value:
                    is_pivot_low = False
                    break
        
        if is_pivot_low:
            pivot_lows.append((i, pivot_low_value))
    
    return pivot_highs, pivot_lows


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
    
    # 使用不同的方法找出摆动点
    # 方法1: 使用较小的窗口找出更多的潜在转折点
    swing_highs_small, swing_lows_small = find_swing_points_simple(recent_df, window=3)
    
    # 方法2: 使用枢轴点方法
    pivot_highs, pivot_lows = find_pivot_points(recent_df, left_bars=5, right_bars=5)
    
    # 合并两种方法的结果，去重
    all_highs = list(set(swing_highs_small + pivot_highs))
    all_lows = list(set(swing_lows_small + pivot_lows))
    
    # 按索引排序
    all_highs.sort(key=lambda x: x[0])
    all_lows.sort(key=lambda x: x[0])
    
    if len(all_lows) < 3 or len(all_highs) < 2:
        return False, None
    
    # 寻找符合条件的模式
    best_pattern = None
    
    # 从倒数第3个低点开始往前找（确保后面有足够的点）
    for start_idx in range(max(0, len(all_lows) - 5), len(all_lows) - 2):
        start_low_idx, start_low_val = all_lows[start_idx]
        
        # 收集在起始低点之后的摆动点
        subsequent_highs = [(idx, val) for idx, val in all_highs if idx > start_low_idx]
        subsequent_lows = [(idx, val) for idx, val in all_lows if idx > start_low_idx]
        
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


def scan_top_500_for_bullish(max_symbols=None):
    """扫描币种，输出符合道氏多头条件的币种"""
    bullish_symbols = []
    symbols_to_scan = top_500_symbols[:max_symbols] if max_symbols else top_500_symbols
    total_symbols = len(symbols_to_scan)

    print(f"Will scan {total_symbols} symbols...")
    print("-" * 60)

    for i, symbol in enumerate(symbols_to_scan):
        print(f"[{i+1}/{total_symbols}] Scanning {symbol}...", end='')
        
        try:
            # Fetch OHLCV data
            df = get_ohlcv(symbol, timeframe='1d', limit=200)

            # Ensure DataFrame is valid
            if df is None or df.empty or len(df) < 30:
                print(" ⏭️  Skipped (insufficient data)")
                continue

            # Check for Dow Theory bullish pattern
            is_bullish, pattern_info = is_dow_theory_bullish_strict(df, lookback_days=150)
            
            if is_bullish:
                bullish_symbols.append({
                    'symbol': symbol,
                    'pattern': pattern_info
                })
                print(f" ✅ BULLISH!")
                if pattern_info:
                    print(f"     Start date: {pattern_info['start_date'].strftime('%Y-%m-%d') if pattern_info['start_date'] is not None else 'N/A'}")
                    print(f"     Start low: {pattern_info['start_low'][1]:.6f}")
                    print(f"     High 1: {pattern_info['highs'][0][1]:.6f}, High 2: {pattern_info['highs'][1][1]:.6f}")
                    print(f"     Low 1: {pattern_info['lows'][0][1]:.6f}, Low 2: {pattern_info['lows'][1][1]:.6f}")
                    print(f"     Rise: {pattern_info['rise_pct']:.2f}%")
            else:
                print(" ❌")

        except Exception as e:
            print(f" ⚠️  Error: {e}")

        # Rate limiting
        time.sleep(0.3)

    return bullish_symbols


def test_specific_symbol(symbol):
    """测试特定币种是否符合条件"""
    print(f"\nTesting {symbol} specifically...")
    print("-" * 60)
    
    df = get_ohlcv(symbol, timeframe='1d', limit=200)
    
    if df is None or df.empty:
        print(f"Failed to fetch data for {symbol}")
        return
    
    print(f"Fetched {len(df)} days of data")
    
    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Test with detailed output
    is_bullish, pattern_info = is_dow_theory_bullish_strict(df, lookback_days=150)
    
    if is_bullish:
        print(f"✅ {symbol} shows a BULLISH Dow Theory pattern!")
        if pattern_info:
            print(f"\nPattern Details:")
            print(f"  Starting Low: {pattern_info['start_low'][1]:.6f} at index {pattern_info['start_low'][0]}")
            print(f"  Date: {pattern_info['start_date'].strftime('%Y-%m-%d') if pattern_info['start_date'] is not None else 'N/A'}")
            print(f"\n  Higher Highs:")
            for i, (idx, val) in enumerate(pattern_info['highs'], 1):
                print(f"    High {i}: {val:.6f} at index {idx}")
            print(f"\n  Higher Lows:")
            for i, (idx, val) in enumerate(pattern_info['lows'], 1):
                print(f"    Low {i}: {val:.6f} at index {idx}")
            print(f"\n  Total Rise: {pattern_info['rise_pct']:.2f}%")
    else:
        print(f"❌ {symbol} does NOT show a clear Dow Theory bullish pattern")
        
        # Show the swing points found for debugging
        recent_df = df.tail(150).reset_index(drop=True)
        swing_highs, swing_lows = find_swing_points_simple(recent_df, window=3)
        pivot_highs, pivot_lows = find_pivot_points(recent_df, left_bars=5, right_bars=5)
        
        print(f"\n  Debug Info:")
        print(f"    Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
        print(f"    Found {len(pivot_highs)} pivot highs, {len(pivot_lows)} pivot lows")
        
        if len(swing_lows) > 0:
            print(f"    Last 3 swing lows: {swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows}")
        if len(swing_highs) > 0:
            print(f"    Last 3 swing highs: {swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs}")


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("Dow Theory Bullish Pattern Scanner for Gate.io")
    print("Looking for: 2 Higher Highs + 2 Higher Lows from recent low")
    print("=" * 60)
    
    # 先测试DENT以验证程序是否正确
    print("\n1. Testing DENT/USDT first...")
    test_specific_symbol('DENT/USDT')
    
    # 询问是否继续扫描
    print("\n" + "=" * 60)
    print("2. Starting full scan...")
    print("=" * 60)
    
    # 执行扫描（可以先扫描前50个进行测试）
    bullish_results = scan_top_500_for_bullish(max_symbols=100)  # 先扫描前100个
    
    print("\n" + "=" * 60)
    print("SCAN RESULTS")
    print("=" * 60)
    
    if bullish_results:
        print(f"\n找到 {len(bullish_results)} 个符合道氏多头条件的币种：\n")
        
        # 按涨幅排序
        bullish_results.sort(key=lambda x: x['pattern']['rise_pct'] if x['pattern'] else 0, reverse=True)
        
        for i, result in enumerate(bullish_results, 1):
            symbol = result['symbol']
            pattern = result['pattern']
            if pattern:
                print(f"{i}. {symbol}")
                print(f"   起始日期: {pattern['start_date'].strftime('%Y-%m-%d') if pattern['start_date'] is not None else 'N/A'}")
                print(f"   涨幅: {pattern['rise_pct']:.2f}%")
                print()
    else:
        print("\n未找到符合条件的币种")
    
    print("\n扫描完成！")