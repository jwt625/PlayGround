#!/usr/bin/env python3
import hashlib
import re

def compute_hash(text, length=16):
    return hashlib.sha256(text.encode()).hexdigest()[:length]

def normalize_command(command):
    if not command:
        return ''
    normalized = re.sub(r'\s*2>/dev/null\s*', ' ', command)
    normalized = re.sub(r'\s*2>&1\s*', ' ', normalized)
    normalized = re.sub(r'\s*\|[^|]*$', '', normalized)
    normalized = normalized.replace('"', '')
    normalized = normalized.replace("'", '')
    normalized = ' '.join(normalized.split())
    return normalized.strip()

# Parent command from request 45
parent_cmd = 'find /Users/wentaojiang/Documents/GitHub/PlayGround/20260108_GLM_claude_code/proxy -maxdepth 1 -name "*.md" -o -name "*.txt" -o -name "*.json" -o -name "*.py" 2>/dev/null | head -20'

# Child command from request 58
child_cmd = 'find /Users/wentaojiang/Documents/GitHub/PlayGround/20260108_GLM_claude_code/proxy -maxdepth 1 -name *.md -o -name *.txt -o -name *.json -o -name *.py'

parent_norm = normalize_command(parent_cmd)
child_norm = normalize_command(child_cmd)

print('Parent normalized:', parent_norm)
print('Parent hash:', compute_hash(parent_norm, 16))
print()
print('Child normalized:', child_norm)
print('Child hash:', compute_hash(child_norm, 16))
print()
print('Match:', parent_norm == child_norm)

