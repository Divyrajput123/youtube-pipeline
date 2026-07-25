import sys, json, base64

d = json.load(sys.stdin)
print('status:', d.get('status'))
e = d.get('error', '')
if e:
    print('error:', e[:300])
out = d.get('output', {})
if 'mp4_b64' in out:
    data = base64.b64decode(out['mp4_b64'])
    open('test_output.mp4', 'wb').write(data)
    print('VIDEO SAVED: test_output.mp4 —', len(data)//1024, 'KB')
else:
    print('output keys:', list(out.keys()))
