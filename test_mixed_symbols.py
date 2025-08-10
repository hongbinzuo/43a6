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
    and not any(token in symbol for token in ['UP/USDT', 'DOWN/USDT', 'BULL/USDT', 'BEAR/USDT'])
]

print(f"Found {len(usdt_spot_symbols)} USDT spot pairs on Gate.io")

# Select more diverse symbols for testing - including some that might not have bullish patterns
test_symbols = [
    'DENT/USDT',   # You mentioned this one specifically
    'BTC/USDT',    # Major crypto
    'ETH/USDT',    # Major crypto
    'XRP/USDT',    # Different market behavior
    'ADA/USDT',    # Different pattern
    'SOL/USDT',    # High volatility
    'MATIC/USDT',  # Mid cap
    'LINK/USDT',   # Oracle token
    'DOT/USDT',    # Different ecosystem
    'UNI/USDT'     # DeFi token
]

# Filter to only include symbols that exist
test_symbols = [s for s in test_symbols if s in usdt_spot_symbols][:10]  # Limit to 10 for quick test

print(f"Testing with {len(test_symbols)} symbols")

def get_ohlcv(symbol, timeframe='1d', limit=1000):
    """获取OHLCV数据，默认返回1天K线数据"""
    try:
        gateio_timeframe = timeframe.lower()
        ohlcv = exchange.fetch_ohlcv(symbol, gateio_timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()

        ohlcv = np.array(ohlcv)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()


def find_swing_points_simple(df, window=5):
    """找出价格的摆动高点和低点"""
    if df is None or df.empty or len(df) < window * 2 + 1:
        return [], []
    
    highs = df['high'].values
    lows = df['low'].values
    
    swing_highs = []
    swing_lows = []
    
    # 找局部高点
    for i in range(window, len(highs) - window):
        is_high = True
        for j in range(i - window, i + window + 1):
            if j != i and highs[j] >= highs[i]:
                is_high = False
                break
        if is_high:
            swing_highs.append((i, highs[i]))
    
    # 找局部低点
    for i in range(window, len(lows) - window):
        is_low = True
        for j in range(i - window, i + window + 1):
            if j != i and lows[j] <= lows[i]:
                is_low = False
                break
        if is_low:
            swing_lows.append((i, lows[i]))
    
    return swing_highs, swing_lows


def is_dow_theory_bullish_strict(df, lookback_days=150):
    """严格判断是否符合道氏多头的条件"""
    if df is None or df.empty or len(df) < 30:
        return False, None

    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['high', 'low'])

    if len(df) < 30:
        return False, None

    recent_df = df.tail(lookback_days).reset_index(drop=True)
    swing_highs, swing_lows = find_swing_points_simple(recent_df, window=3)
    
    if len(swing_lows) < 3 or len(swing_highs) < 2:
        return False, None
    
    best_pattern = None
    
    # 从倒数第3个低点开始往前找
    for start_idx in range(max(0, len(swing_lows) - 5), len(swing_lows) - 2):
        start_low_idx, start_low_val = swing_lows[start_idx]
        
        subsequent_highs = [(idx, val) for idx, val in swing_highs if idx > start_low_idx]
        subsequent_lows = [(idx, val) for idx, val in swing_lows if idx > start_low_idx]
        
        if len(subsequent_highs) >= 2 and len(subsequent_lows) >= 2:
            check_highs = subsequent_highs[:2]
            check_lows = subsequent_lows[:2]
            
            # Check for 2 higher highs
            highs_ascending = check_highs[1][1] > check_highs[0][1]
            
            # Check for 2 higher lows
            lows_ascending = (check_lows[0][1] > start_low_val and 
                            check_lows[1][1] > check_lows[0][1])
            
            second_low_higher = check_lows[1][1] > start_low_val
            
            if highs_ascending and lows_ascending and second_low_higher:
                rise_pct = ((check_highs[-1][1] - start_low_val) / start_low_val) * 100
                
                pattern_info = {
                    'start_low': (start_low_idx, start_low_val),
                    'highs': check_highs,
                    'lows': check_lows,
                    'rise_pct': rise_pct,
                    'start_date': recent_df.iloc[start_low_idx]['timestamp'] if start_low_idx < len(recent_df) else None
                }
                
                if best_pattern is None or pattern_info['start_low'][0] > best_pattern['start_low'][0]:
                    best_pattern = pattern_info
    
    return best_pattern is not None, best_pattern


def test_symbol_compact(symbol):
    """Compact testing for each symbol"""
    df = get_ohlcv(symbol, timeframe='1d', limit=200)
    
    if df is None or df.empty:
        return False, "No data"
    
    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    current_price = df['close'].iloc[-1]
    is_bullish, pattern_info = is_dow_theory_bullish_strict(df, lookback_days=150)
    
    if is_bullish and pattern_info:
        days_since = (df['timestamp'].iloc[-1] - pattern_info['start_date']).days if pattern_info['start_date'] is not None else 0
        return True, {
            'current_price': current_price,
            'start_low': pattern_info['start_low'][1],
            'start_date': pattern_info['start_date'],
            'days_since': days_since,
            'rise_pct': pattern_info['rise_pct'],
            'current_vs_start': ((current_price - pattern_info['start_low'][1]) / pattern_info['start_low'][1] * 100)
        }
    else:
        return False, "No pattern"


# Main program
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DOW THEORY BULLISH PATTERN SCANNER - COMPREHENSIVE TEST")
    print("=" * 70)
    print("Pattern Requirements: 2 Higher Highs + 2 Higher Lows from recent low")
    print("-" * 70)
    
    results = []
    
    print(f"\nScanning {len(test_symbols)} symbols...\n")
    print(f"{'Symbol':<12} {'Status':<10} {'Days Since':<12} {'Rise %':<10} {'Current vs Start'}")
    print("-" * 70)
    
    for symbol in test_symbols:
        try:
            is_bullish, info = test_symbol_compact(symbol)
            
            if is_bullish and isinstance(info, dict):
                status = "✅ BULLISH"
                days = f"{info['days_since']} days"
                rise = f"{info['rise_pct']:.1f}%"
                current = f"{info['current_vs_start']:.1f}%"
                results.append((symbol, True, info))
            else:
                status = "❌ NO"
                days = "-"
                rise = "-"
                current = info
                results.append((symbol, False, info))
            
            print(f"{symbol:<12} {status:<10} {days:<12} {rise:<10} {current}")
            
            # Small delay to avoid rate limits
            time.sleep(0.3)
            
        except Exception as e:
            print(f"{symbol:<12} {'⚠️ ERROR':<10} {str(e)[:50]}")
            results.append((symbol, False, f"Error: {e}"))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    bullish_count = sum(1 for _, is_bullish, _ in results if is_bullish)
    total_count = len(results)
    
    print(f"\nTotal symbols scanned: {total_count}")
    print(f"Bullish patterns found: {bullish_count}")
    print(f"No pattern found: {total_count - bullish_count}")
    print(f"Success rate: {(bullish_count/total_count*100):.1f}%")
    
    if bullish_count > 0:
        print(f"\n📈 Symbols with Dow Theory Bullish Pattern (2高2低):")
        for symbol, is_bullish, info in results:
            if is_bullish and isinstance(info, dict):
                print(f"  • {symbol}: Started {info['start_date'].strftime('%Y-%m-%d') if info['start_date'] else 'N/A'}, "
                      f"Current gain: {info['current_vs_start']:.1f}%")
    
    print("\n✨ Test complete!")