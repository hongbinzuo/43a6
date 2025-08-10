from flask import Flask, render_template, jsonify
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import json

app = Flask(__name__)

# Global variable to store scanning progress
scanning_status = {
    'is_scanning': False,
    'progress': 0,
    'total': 0,
    'current_symbol': '',
    'results': [],
    'error': None
}

def find_swing_points(prices, window=3):
    """Find swing highs and lows in price data"""
    highs = []
    lows = []
    
    for i in range(window, len(prices) - window):
        # Check for swing high
        is_high = True
        for j in range(i - window, i + window + 1):
            if j != i and prices[j] >= prices[i]:
                is_high = False
                break
        if is_high:
            highs.append(i)
        
        # Check for swing low
        is_low = True
        for j in range(i - window, i + window + 1):
            if j != i and prices[j] <= prices[i]:
                is_low = False
                break
        if is_low:
            lows.append(i)
    
    return highs, lows

def check_bullish_pattern(df):
    """Check if the price data shows a bullish Dow Theory pattern"""
    if len(df) < 20:
        return False, {}
    
    prices = df['close'].values
    dates = df.index
    
    # Find swing points
    highs, lows = find_swing_points(prices, window=3)
    
    if len(lows) < 3 or len(highs) < 2:
        return False, {}
    
    # Look for pattern starting from recent lows
    for start_idx in range(len(lows) - 3, -1, -1):
        low_start = lows[start_idx]
        
        # Find subsequent highs and lows
        higher_highs = []
        higher_lows = []
        
        # Find highs after the starting low
        for high_idx in highs:
            if high_idx > low_start:
                if not higher_highs or prices[high_idx] > prices[higher_highs[-1]]:
                    higher_highs.append(high_idx)
        
        # Find lows after the starting low
        for low_idx in lows:
            if low_idx > low_start and prices[low_idx] > prices[low_start]:
                if not higher_lows or prices[low_idx] > prices[higher_lows[-1]]:
                    higher_lows.append(low_idx)
        
        # Check if we have the required pattern
        if len(higher_highs) >= 2 and len(higher_lows) >= 2:
            pattern_details = {
                'start_date': dates[low_start].strftime('%Y-%m-%d'),
                'start_low': float(prices[low_start]),
                'high1': float(prices[higher_highs[0]]),
                'high2': float(prices[higher_highs[1]]),
                'low1': float(prices[higher_lows[0]]),
                'low2': float(prices[higher_lows[1]]),
                'rise_pct': float(((prices[higher_highs[-1]] - prices[low_start]) / prices[low_start]) * 100)
            }
            return True, pattern_details
    
    return False, {}

def scan_symbols():
    """Scan cryptocurrency symbols for Dow Theory patterns"""
    global scanning_status
    
    try:
        scanning_status['is_scanning'] = True
        scanning_status['results'] = []
        scanning_status['error'] = None
        
        # Initialize exchange
        exchange = ccxt.gateio({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        # Load markets
        markets = exchange.load_markets()
        
        # Filter USDT pairs
        usdt_symbols = []
        for symbol in markets:
            if symbol.endswith('/USDT') and markets[symbol]['active']:
                base = symbol.split('/')[0]
                # Skip leveraged tokens
                if not any(base.endswith(suffix) for suffix in ['UP', 'DOWN', 'BULL', 'BEAR', '3L', '3S', '5L', '5S']):
                    usdt_symbols.append(symbol)
        
        # Limit to first 50 symbols for faster demo
        max_symbols = min(50, len(usdt_symbols))
        usdt_symbols = usdt_symbols[:max_symbols]
        
        scanning_status['total'] = len(usdt_symbols)
        
        # Scan each symbol
        for idx, symbol in enumerate(usdt_symbols):
            if not scanning_status['is_scanning']:
                break
                
            scanning_status['progress'] = idx + 1
            scanning_status['current_symbol'] = symbol
            
            try:
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=150)
                
                if len(ohlcv) < 50:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Check for bullish pattern
                is_bullish, details = check_bullish_pattern(df)
                
                if is_bullish:
                    result = {
                        'symbol': symbol,
                        'start_date': details['start_date'],
                        'rise_pct': round(details['rise_pct'], 2),
                        'start_low': details['start_low'],
                        'high1': details['high1'],
                        'high2': details['high2'],
                        'low1': details['low1'],
                        'low2': details['low2']
                    }
                    scanning_status['results'].append(result)
                
                # Small delay to avoid rate limits
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Error scanning {symbol}: {str(e)}")
                continue
        
        # Sort results by rise percentage
        scanning_status['results'].sort(key=lambda x: x['rise_pct'], reverse=True)
        
    except Exception as e:
        scanning_status['error'] = str(e)
    finally:
        scanning_status['is_scanning'] = False
        scanning_status['current_symbol'] = ''

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/start_scan', methods=['POST'])
def start_scan():
    """Start the scanning process"""
    global scanning_status
    
    if scanning_status['is_scanning']:
        return jsonify({'error': 'Scan already in progress'}), 400
    
    # Start scanning in a background thread
    thread = threading.Thread(target=scan_symbols)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Scan started'})

@app.route('/scan_status')
def scan_status():
    """Get the current scanning status"""
    return jsonify(scanning_status)

@app.route('/stop_scan', methods=['POST'])
def stop_scan():
    """Stop the scanning process"""
    global scanning_status
    scanning_status['is_scanning'] = False
    return jsonify({'message': 'Scan stopped'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)