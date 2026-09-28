#!/usr/bin/env python3
"""CPU-only transport guard: HTTP chunks must arrive without per-byte backlog."""
if __name__ != "__main__":  # collected by pytest: this is a GPU experiment script, not a unit test
    import pytest
    pytest.skip("GPU experiment script: run it directly (see the docstring)", allow_module_level=True)
import json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import requests
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def do_GET(self):
  self.send_response(200);self.send_header('Transfer-Encoding','chunked');self.send_header('Content-Type','text/event-stream');self.end_headers()
  start=time.perf_counter()
  for i in range(20):
   payload=b'data: '+json.dumps({'i':i,'server_elapsed':time.perf_counter()-start,'text':'x'*32768}).encode()+b'\n\n'
   self.wfile.write(f'{len(payload):x}\r\n'.encode()+payload+b'\r\n');self.wfile.flush();time.sleep(.02)
  self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
results=[]
try:
 for size in (1,None):
  times=[];start=time.perf_counter()
  with requests.get(f'http://127.0.0.1:{server.server_port}',stream=True) as response:
   for line in response.iter_lines(chunk_size=size):
    if line.startswith(b'data: '):
     d=json.loads(line[6:]);times.append((time.perf_counter()-start,d['server_elapsed']))
  results.append(dict(chunk_size=size,records=len(times),elapsed=times[-1][0],server_elapsed=times[-1][1],max_delivery_delay=max(a-b for a,b in times)))
 assert results[1]['records']==20
 assert results[1]['max_delivery_delay']<.05,results
finally:server.shutdown();server.server_close()
path=Path(__file__).resolve().parents[1]/'specs/wa6/sse-transport-test.json';path.write_text(json.dumps(results,indent=2)+'\n');print(path.read_text())
