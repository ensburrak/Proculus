"""V9 local Proculus HTTP acceptance: synthetic data, no exchange or personal keys."""
import json,math,os,shutil,socket,subprocess,sys,tempfile,time,unittest,urllib.error,urllib.request
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TOKEN='synthetic-temporary-ci-research-token-000123456789'
def sample():
    t=datetime(2025,1,1,tzinfo=timezone.utc);rows=['timestamp,open,high,low,close']
    for i in range(135):
        p=130+i*.19+2*math.sin(i*.18)
        rows.append(f"{(t+timedelta(minutes=i)).strftime('%Y-%m-%dT%H:%M:%SZ')},{p:.5f},{p+1:.5f},{p-1:.5f},{p:.5f}")
    return '\n'.join(rows)+'\n'
def req(url,method='GET',payload=None,headers=None):
    body=None if payload is None else json.dumps(payload).encode()
    h=headers or {}
    if body is not None:h={'Content-Type':'application/json',**h}
    r=urllib.request.Request(url,data=body,headers=h,method=method)
    try:
        with urllib.request.urlopen(r,timeout=3) as res:return res.status,json.load(res)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
class OperatorV9HTTP(unittest.TestCase):
    def test_loopback_token_origin_real_jobs(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)/'fake';root.mkdir()
            (root/'config.json').write_text('{"paper_trading":{"enabled":true}}')
            (root/'studio_control').mkdir()
            shutil.copy2(ROOT/'studio_control/research_jobs.py',root/'studio_control/research_jobs.py')
            (root/'web').mkdir()
            (root/'web/operator.html').write_text('<html><body>ci only</body></html>')
            (root/'web/index.html').write_text('<html>test</html>')
            with socket.socket() as s:
                s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            env={**os.environ,'STUDIO_RESEARCH_TOKEN':TOKEN,'STUDIO_RESEARCH_HOME':str(Path(work)/'private')}
            proc=subprocess.Popen([sys.executable,str(ROOT/'web/paper_operator_v9.py'),'--repo',str(root),'--site',str(root/'web'),'--port',str(port)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            url=f'http://127.0.0.1:{port}'
            auth={'X-Research-Token':TOKEN,'X-Ops-Reason':'Synthetic CI offline research','X-Research-Confirm':'OFFLINE'}
            try:
                for _ in range(60):
                    try:
                        status,_=req(url+'/api/studio/v1/health')
                        if status==200:break
                    except Exception:time.sleep(.06)
                else:self.fail('Observer failed to start')
                code,caps=req(url+'/api/studio/research/capabilities')
                self.assertEqual(code,200)
                self.assertFalse(caps['real_orders'])
                self.assertEqual(req(url+'/api/studio/research/jobs')[0],401)
                self.assertEqual(req(url+'/api/studio/research/jobs',headers={'X-Research-Token':TOKEN,'Host':'evil.invalid'})[0],403)
                self.assertEqual(req(url+'/api/studio/research/jobs',headers={'X-Research-Token':TOKEN,'Origin':'https://evil.invalid'})[0],403)
                self.assertEqual(req(url+'/api/studio/v1/health','POST',{})[0],405)
                self.assertEqual(req(url+'/api/studio/research/jobs','PUT',{})[0],405)
                self.assertEqual(req(url+'/api/studio/research/datasets','POST',{'csv_text':sample()},headers={'X-Research-Token':TOKEN})[0],400)
                code,data=req(url+'/api/studio/research/datasets','POST',{'csv_text':sample()},auth)
                self.assertEqual(code,201)
                payload={'kind':'backtest','dataset_id':data['dataset_id'],'params':{'fast':7,'slow':24}}
                key={**auth,'Idempotency-Key':'ci-v9-idempotent-job-001'}
                code,job=req(url+'/api/studio/research/jobs','POST',payload,key)
                self.assertEqual(code,201)
                self.assertEqual(req(url+'/api/studio/research/jobs','POST',payload,key)[1]['id'],job['id'])
                for _ in range(100):
                    _,now=req(url+'/api/studio/research/jobs/'+job['id'],headers={'X-Research-Token':TOKEN})
                    if now['state'] not in ('queued','running'):break
                    time.sleep(.04)
                self.assertEqual(now['state'],'completed')
                code,artifact=req(url+'/api/studio/research/jobs/'+job['id']+'/artifact',headers={'X-Research-Token':TOKEN})
                self.assertEqual(code,200)
                self.assertTrue(artifact['no_exchange_orders'])
                self.assertEqual(artifact['dataset_sha256'],data['sha256'])
            finally:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':unittest.main()
