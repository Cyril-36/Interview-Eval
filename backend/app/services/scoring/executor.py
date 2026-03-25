from concurrent.futures import ThreadPoolExecutor

# Shared thread pool for all CPU-bound scoring work.
# Two workers is enough: NLP scorers are sequential per-request,
# and the pool just keeps them off the async event loop.
executor = ThreadPoolExecutor(max_workers=2)
