import torch
from sentence_transformers import SentenceTransformer
import time
import warnings
warnings.filterwarnings('ignore')

print('Testing LaBSE GPU execution (CPU-first approach)...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Target device: {device}')

torch.cuda.reset_peak_memory_stats()

# Load model ON CPU first
print('Loading LaBSE model on CPU...')
model = SentenceTransformer('sentence-transformers/LaBSE')
model = model.to(device)
model.eval()

print(f'Model loaded and moved to: {device}')

# Test sentences
sentences = [
    'Hello, this is a test sentence.',
    'Write a Python function to calculate factorial.'
] * 25

print(f'Encoding {len(sentences)} sentences on {device}...')
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    embeddings = model.encode(sentences, batch_size=32, convert_to_tensor=True, device=device)

torch.cuda.synchronize()
elapsed = time.time() - start

peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
print(f'\nResults:')
print(f'  Encoding time: {elapsed:.2f}s')
print(f'  Peak GPU memory: {peak_mem:.2f} GB')
print(f'  Embeddings device: {embeddings.device}')
print(f'  Embeddings shape: {embeddings.shape}')

if peak_mem > 0.05:
    print('\n✓ SUCCESS: LaBSE is running on GPU!')
else:
    print(f'\n⚠ Embeddings are on {embeddings.device}')
