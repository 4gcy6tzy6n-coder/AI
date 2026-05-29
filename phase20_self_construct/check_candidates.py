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

print(f'总候选数: {len(candidates)}')
scores = [c['metadata']['quality_score'] for c in candidates]
print(f'分数范围: {min(scores):.3f} - {max(scores):.3f}')
print(f'≥0.90: {sum(1 for s in scores if s >= 0.90)}')
print(f'≥0.85: {sum(1 for s in scores if s >= 0.85)}')

# 显示高分候选
print('\n高分候选:')
for c in sorted(candidates, key=lambda x: x['metadata']['quality_score'], reverse=True)[:10]:
    print(f"  {c['candidate_id']}: {c['metadata']['quality_score']:.3f} ({c.get('subcategory', 'unknown')})")
