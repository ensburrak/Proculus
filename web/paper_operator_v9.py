#!/usr/bin/env python3
"""Proculus V9 local-only, token-guarded OFFLINE research API. No exchange use.

Run: STUDIO_RESEARCH_TOKEN=<32+ random chars> python web/paper_operator_v9.py
Only binds 127.0.0.1; NEVER proxy/publish directly to the public internet.
"""
from __future__ import annotations
import argparse,hmac,json,os,sys
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
from paper_observer_v7 import json_report,recent_events,ALLOW
MAX_BODY=3*1024*1024+16384

def serve(repo:Path,site:Path,port:int,secret:str):
    repo=repo.resolve(strict=True);site=site.resolve(strict=True)
    sys.path.insert(0,str(repo))
    from studio_control.research_jobs import JobStore,JobWorker,InputError
    enabled=bool(secret and len(secret)>=32)
    store=JobStore(Path(os.getenv('STUDIO_RESEARCH_HOME',str(repo/'runtime_studio_v9')))) if enabled else None
    worker=JobWorker(store) if enabled else None
    if worker:worker.start()
    hosts={f'127.0.0.1:{port}',f'localhost:{port}'}
    origins={'http://'+v for v in hosts}

    class Handler(BaseHTTPRequestHandler):
        server_version='ProculusOfflineResearch/0.9'
        def log_message(self,fmt,*args):
            # Do not log supplied token, authorization data or request body.
            print('[paper-only]',fmt%args)
        def respond(self,data,status=200):
            raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status)
            for k,v in {'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Content-Security-Policy':"default-src 'none'; frame-ancestors 'none'",'Content-Length':str(len(raw))}.items():self.send_header(k,v)
            self.end_headers();self.wfile.write(raw)
        def guard(self):
            if self.headers.get('Host') not in hosts:return False
            origin=self.headers.get('Origin','')
            if origin and origin not in origins:return False
            return self.headers.get('Sec-Fetch-Site')!='cross-site'
        def route_path(self):
            parsed=urlsplit(self.path)
            if parsed.query or parsed.fragment:raise ValueError('Query strings denied')
            return unquote(parsed.path)
        def owner(self):
            if not enabled:self.respond({'error':'STUDIO_RESEARCH_TOKEN>=32 chars required'},503);return None
            value=str(self.headers.get('X-Research-Token',''))
            if not hmac.compare_digest(value,secret):
                self.respond({'error':'Research token required'},401);return None
            return 'local_operator'
        def require_write(self):
            account=self.owner()
            if not account:return None
            reason=str(self.headers.get('X-Ops-Reason',''))
            if len(reason.strip())<12 or self.headers.get('X-Research-Confirm','')!='OFFLINE':
                self.respond({'error':'Reason >=12 chars and OFFLINE confirmation required'},400);return None
            return account
        def serve_file(self,name):
            target=site/name
            if not target.is_file():return self.respond({'error':'Optional V9 HTML not installed'},404)
            raw=target.read_bytes()
            self.send_response(200)
            for k,v in {'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer',
                'Content-Security-Policy':"default-src 'self' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; connect-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",'Content-Length':str(len(raw))}.items():self.send_header(k,v)
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.guard():return self.respond({'error':'Host/origin denied'},403)
            try:path=self.route_path()
            except ValueError:return self.respond({'error':'Query denied'},400)
            if path in ('/','/index.html'):return self.serve_file('index.html')
            if path=='/operator.html':return self.serve_file('operator.html')
            if path=='/Proculus_Immersive_v9.html':return self.serve_file('Proculus_Immersive_v9.html')
            if path=='/Proculus_Immersive_v10.html':return self.serve_file('Proculus_Immersive_v10.html')
            if path.startswith('/api/studio/v1/'):
                key=path[len('/api/studio/v1/'):]
                if key=='production-readiness':
                    from live_readiness_probe import inspect
                    return self.respond(inspect(repo))
                if key=='health':return self.respond({'status':'ok','mode':'local_paper_only','utc':datetime.now(timezone.utc).isoformat(),'worker_enabled':enabled})
                if key=='capabilities':return self.respond({'mode':'local_paper_only','legacy_sources':list(ALLOW),'real_orders':False,'model_deployment':False})
                if key=='config':
                    data=json_report(repo,'config.json')
                    if data.get('available') and isinstance(data.get('data'),dict):
                        from paper_observer_v7 import mask
                        data['data']=mask({k:data['data'].get(k) for k in ('paper_trading','edge_learning','live_readiness','performance_evidence')})
                    return self.respond(data)
                if key=='snapshot':return self.respond({'mode':'local_paper_only','real_balance':None,
                    'reports':{k:json_report(repo,ALLOW[k]) for k in ('setup_edge_stats','edge_calibration_proposals')}})
                if key=='edge_events_tail':return self.respond(recent_events(repo))
                if key in ALLOW and key!='edge_events_tail':return self.respond(json_report(repo,ALLOW[key]))
                return self.respond({'error':'Not allowlisted'},404)
            if path=='/api/studio/research/capabilities':
                return self.respond({'enabled':enabled,'mode':'local_offline_only','real_orders':False,'model_auto_deploy':False,
                    'worker':bool(worker and worker.thread and worker.thread.is_alive()),
                    'engines':['offline_ema_cross_v1','offline_logistic_research_v1']})
            if path.startswith('/api/studio/research/'):
                user=self.owner()
                if not user:return
                tail=path[len('/api/studio/research/'):];parts=tail.split('/')
                try:
                    if tail=='jobs':return self.respond({'jobs':store.list(user)})
                    if len(parts)==2 and parts[0]=='jobs':return self.respond(store.get(user,parts[1]))
                    if len(parts)==3 and parts[0]=='jobs' and parts[2]=='artifact':return self.respond(store.artifact(user,parts[1]))
                except (InputError,OSError):return self.respond({'error':'Job or artifact not available'},404)
            return self.respond({'error':'Not found'},404)
        def do_POST(self):
            if not self.guard():return self.respond({'error':'Host/origin denied'},403)
            try:path=self.route_path()
            except ValueError:return self.respond({'error':'Query denied'},400)
            if not path.startswith('/api/studio/research/'):return self.respond({'error':'Writes not implemented outside offline research'},405)
            user=self.require_write()
            if not user:return
            try:size=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.respond({'error':'Content length invalid'},400)
            if size<=0 or size>MAX_BODY:return self.respond({'error':'Body too large'},413)
            if not str(self.headers.get('Content-Type','')).startswith('application/json'):return self.respond({'error':'JSON required'},415)
            try:body=json.loads(self.rfile.read(size))
            except (UnicodeError,ValueError):return self.respond({'error':'Invalid JSON'},400)
            if not isinstance(body,dict):return self.respond({'error':'JSON object required'},400)
            tail=path[len('/api/studio/research/'):]
            try:
                if tail=='datasets':
                    text=body.get('csv_text')
                    if not isinstance(text,str):raise InputError('csv_text string required')
                    return self.respond(store.upload(user,text),201)
                if tail=='jobs':
                    key=str(self.headers.get('Idempotency-Key',''))
                    return self.respond(store.create(user,str(body.get('kind','')),str(body.get('dataset_id','')),body.get('params',{}),key),201)
                parts=tail.split('/')
                if len(parts)==3 and parts[0]=='jobs' and parts[2]=='cancel':
                    return self.respond(store.cancel(user,parts[1]))
            except InputError as error:return self.respond({'error':str(error)},422)
            return self.respond({'error':'Unsupported write'},405)
        def deny(self):self.respond({'error':'Unsupported HTTP method'},405)
        do_PUT=do_PATCH=do_DELETE=do_OPTIONS=deny
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    try:
        print(f'Offline operator: http://127.0.0.1:{port}/operator.html; worker enabled: {enabled}')
        server.serve_forever(poll_interval=.4)
    finally:
        server.server_close()
        if worker:worker.stop()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--site',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--port',type=int,default=8096)
    opts=parser.parse_args()
    if not (opts.repo/'config.json').is_file():parser.error('Valid Proculus checkout with config.json required')
    serve(opts.repo,opts.site,opts.port,os.getenv('STUDIO_RESEARCH_TOKEN',''))
