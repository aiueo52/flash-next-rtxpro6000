"""The exact BN1 streaming warmup contract initializes matching metric labels."""
import sys
sys.path.insert(0,'/home/user/tools/flash-next-bench')
from fnbench.http_client import OpenAIStreamClient
from fnbench.models import Sampling
client=OpenAIStreamClient('http://127.0.0.1:8001/v1')
try:
    client.complete(prompt='Write a detailed practical guide to maintaining a small community bicycle workshop. Discuss tools, storage, inspection and staff handover in connected paragraphs.',max_tokens=1024,sampling=Sampling(0.0),model='flash-next')
    print('BN1 streaming warmup complete',flush=True)
finally:client.session.close()
