"""检查Stage 14.2 tokenizer"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from stage14_2_tokenization import SimpleBigramTokenizer

tokenizer = SimpleBigramTokenizer()
tokenizer.fitted = True
tokenizer.word2id = {}
tokenizer.id2word = {}

ckpt = torch.load('stage8_dataset/stage14_2_decoder.pt', map_location='cpu', weights_only=False)
tokenizer.word2id = ckpt['tokenizer_word2id']
tokenizer.id2word = ckpt['tokenizer_id2word']

print('词表大小:', len(tokenizer.word2id))
print('BOS_ID:', 2, 'EOS_ID:', 3, 'PAD_ID:', 0, 'UNK_ID:', 1)

test = '你好'
ids = tokenizer.encode(test, 20)
print(f'编码"{test}":', ids)

decoded = tokenizer.decode(ids)
print(f'解码:', decoded)

print('\n词表样本:')
for i in range(10):
    if i in tokenizer.id2word:
        print(f'  {i}: {repr(tokenizer.id2word[i])}')

print('\n解码生成结果测试:')
for tid in [4, 5, 6, 7, 8, 9, 10]:
    if tid in tokenizer.id2word:
        print(f'  {tid}: {repr(tokenizer.id2word[tid])}')
