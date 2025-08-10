# Dow Theory Bullish Pattern Scanner for Gate.io

这是一个用于扫描Gate.io交易所USDT现货交易对的道氏理论多头模式扫描器。

## 功能说明

该程序会扫描符合以下道氏理论多头条件的币种：
- 从最近的一个低点开始
- 形成至少2个更高的高点（Higher Highs）
- 形成至少2个更高的低点（Higher Lows）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 文件说明

1. **dow_theory_scanner.py** - 基础版本，使用scipy库进行摆动点检测
2. **dow_theory_scanner_improved.py** - 改进版本，不依赖scipy，提供更详细的模式识别

## 使用方法

### 运行改进版本（推荐）

```bash
python dow_theory_scanner_improved.py
```

程序会：
1. 首先测试DENT/USDT以验证算法正确性
2. 然后扫描前100个USDT交易对
3. 输出所有符合条件的币种及其模式详情

### 运行基础版本

```bash
python dow_theory_scanner.py
```

## 主要改进

相比原始代码，修正后的版本有以下改进：

1. **正确的模式识别**：不再只看最后3个数据点，而是寻找完整的摆动点序列
2. **多种检测方法**：结合局部极值和枢轴点两种方法找出关键转折点
3. **严格的验证**：确保从一个低点开始，后续有2个递增的高点和2个递增的低点
4. **详细的调试信息**：提供模式的具体细节，包括起始日期、各个高低点的值等
5. **特定币种测试**：可以单独测试某个币种（如DENT）来验证算法

## 参数调整

在`dow_theory_scanner_improved.py`中可以调整以下参数：

- `lookback_days`: 回看天数，默认150天
- `window`: 摆动点检测窗口，默认3-5根K线
- `max_symbols`: 扫描的最大币种数量，默认100个

## 注意事项

1. 程序包含了0.3-0.5秒的延迟以避免触发交易所的频率限制
2. Gate.io可能没有500个USDT现货交易对，实际扫描数量会受限
3. 程序会自动过滤掉杠杆代币（如UP/DOWN/BULL/BEAR结尾的代币）

## 输出示例

```
Testing DENT/USDT specifically...
✅ DENT/USDT shows a BULLISH Dow Theory pattern!

Pattern Details:
  Starting Low: 0.001234 at index 45
  Date: 2024-01-15
  
  Higher Highs:
    High 1: 0.001456 at index 52
    High 2: 0.001678 at index 68
  
  Higher Lows:
    Low 1: 0.001345 at index 58
    Low 2: 0.001423 at index 75
  
  Total Rise: 36.12%
```
