# Extracted from features.py for portable use. See manifests/code_origins.csv.
import re
import unicodedata

def norm_name(text):return ' '.join(re.findall('[a-z0-9]+',unicodedata.normalize('NFKC',text).casefold()))

def jaccard(left,right):return len(left&right)/len(left|right) if left|right else 0.

def parent_context(bank,uri):
    """Exact N25/N27 operation: BFS three levels, all named ancestors per level."""
    seen={uri};front=bank[uri]['parents'];names=[]
    for _level in range(3):
        next_front=[]
        for parent in sorted(set(front)):
            if parent in seen:continue
            seen.add(parent)
            if parent in bank:
                if bank[parent]['name']:names.append(bank[parent]['name'])
                next_front.extend(bank[parent]['parents'])
        front=next_front
    return ' | '.join(names)
