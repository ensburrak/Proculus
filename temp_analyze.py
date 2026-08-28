import os
import ast

def analyze_file(filepath):
    if not os.path.exists(filepath):
        return f"{filepath} not found."
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.split('\n')
    size = len(lines)
    
    has_docstring = False
    try:
        tree = ast.parse(content)
        has_docstring = ast.get_docstring(tree) is not None
    except:
        pass
        
    return f"{os.path.basename(filepath)}: {size} lines, Docstring: {bool(has_docstring)}"

files = [
    "analyzer.py", "market_metrics.py", "cross_asset_signals.py", "mtf_alignment.py", 
    "entry_optimizer.py", "portfolio_optimizer.py", "monte_carlo.py", "pair_trading.py", 
    "orderbook_analyzer.py", "session_filter.py", "timing_filters.py", "signal_service.py", 
    "signal_logger.py", "whale_alert_provider.py", "onchain_analytics.py", "social_scanner.py", 
    "macro_filter.py", "anomaly_detector.py", "performance_analyzer.py", "report_generator.py"
]

print("=== SPECIFIC FILES ===")
for f in files:
    print(analyze_file(f))

directories = ['analysis', 'backtesting', 'decision', 'tools', 'quality_gate', 'paper_trading', 'tests']
print("\n=== DIRECTORIES ===")
for d in directories:
    if not os.path.exists(d):
        continue
    for root, _, filenames in os.walk(d):
        for name in filenames:
            if name.endswith('.py'):
                print(analyze_file(os.path.join(root, name)))
