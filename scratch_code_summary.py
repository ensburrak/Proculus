import os
import ast
import json

def analyze_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        classes = []
        functions = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
        return {'size': len(content), 'classes': classes, 'functions': functions}
    except Exception as e:
        return {'error': str(e)}

dirs_to_scan = [
    'analysis',
    'decision/strategy_experts',
    'decision',
    'core/engine'
]

results = {}
for d in dirs_to_scan:
    if os.path.exists(d):
        for file in os.listdir(d):
            if file.endswith('.py') and file != '__init__.py':
                path = os.path.join(d, file)
                if not os.path.isdir(path):
                    results[path.replace('\\', '/')] = analyze_file(path)

for file in ['analyzer.py', 'meta_strategy_selector.py', 'chatgpt_decision_layer.py']:
    if os.path.exists(file):
        results[file] = analyze_file(file)

with open('scratch_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Analysis complete.')
