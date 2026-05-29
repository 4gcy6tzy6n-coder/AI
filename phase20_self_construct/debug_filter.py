import json

with open('candidates/stage5e_relation_candidates.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
    candidates = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            candidates.append(data)
        except:
            continue

mappable_categories = [
    'identity', 'project_state', 'technical_stack',
    'history_recall', 'causal', 'temporal'
]

print("筛选详情:")
for c in candidates:
    score = c.get('metadata', {}).get('quality_score', 0)
    subcategory = c.get('subcategory', '')
    entities = c.get('entities', {})
    
    checks = []
    if score >= 0.90:
        checks.append("score>=0.90")
    if subcategory in mappable_categories:
        checks.append("mappable")
    if entities.get('a') and entities.get('b'):
        checks.append("has_entities")
    
    if score >= 0.90:
        print(f"{c['candidate_id']}: score={score:.3f}, subcat={subcategory}, entities=({entities.get('a', 'N/A')}, {entities.get('b', 'N/A')}), checks={checks}")
