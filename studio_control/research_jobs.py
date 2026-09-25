"""Durable OFFLINE-ONLY research jobs. No exchange, live execution or model promotion."""
from __future__ import annotations
import csv,hashlib,io,json,math,os,sqlite3,threading,time,uuid
from pathlib import Path
from datetime import datetime,timezone
MAX_BYTES=3145728
MAX_ROWS=20000
class InputError(ValueError):pass
def now():return datetime.now(timezone.utc).isoformat()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def dumps(o):return json.dumps(o,separators=(',',':'),allow_nan=False,ensure_ascii=False)
def parsed_csv(text):
    raw=text.encode('utf-8')
    if not raw or len(raw)>MAX_BYTES:raise InputError('CSV 1 bayt ile 3 MB arasinda olmali')
    reader=csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {'timestamp','open','high','low','close'}.issubset({str(x).strip().lower() for x in reader.fieldnames}):raise InputError('Eksik OHLC sutunu')
    rows=[];last=''
    for item in reader:
        if len(rows)>=MAX_ROWS:raise InputError('Maksimum 20.000 mum')
        row={str(k or '').lower().strip():v for k,v in item.items()}
        ts=str(row.get('timestamp') or '')
        try:
            if ts.isdecimal():
                epoch=int(ts);dt=datetime.fromtimestamp(epoch/1000 if epoch>1000000000000 else epoch,timezone.utc)
            else:
                if not ts.endswith('Z'):raise ValueError()
                dt=datetime.fromisoformat(ts.replace('Z','+00:00'))
            t=dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            if t<=last:raise ValueError()
            prices={k:float(row[k]) for k in ('open','high','low','close')}
            if any(not math.isfinite(v) or v<=0 or v>1e12 for v in prices.values()):raise ValueError()
            if prices['high']<max(prices['open'],prices['close']) or prices['low']>min(prices['open'],prices['close']) or prices['low']>prices['high']:raise ValueError()
        except (ValueError,TypeError,OverflowError,OSError,KeyError):
            raise InputError('Gecersiz OHLC veya sirali UTC zaman bilgisi') from None
        last=t;rows.append({'timestamp':t,**prices})
    if len(rows)<100:raise InputError('En az 100 mum gerekli')
    return rows
def ema(vals,span):
    alpha=2/(span+1);e=vals[0];out=[]
    for v in vals:e=e+alpha*(v-e);out.append(e)
    return out
def backtest(bars,p):
    try:
        fast=int(p.get('fast',12));slow=int(p.get('slow',30));fee=float(p.get('fee_bps',4));slip=float(p.get('slippage_bps',3));capital=float(p.get('capital',10000))
    except (TypeError,ValueError,OverflowError):raise InputError('Invalid backtest parameters') from None
    if not(2<=fast<=30 and fast+2<=slow<=100 and 0<=fee<=100 and 0<=slip<=200 and 100<=capital<=1000000):raise InputError('Backtest parametreleri sinir disi')
    close=[b['close'] for b in bars];a=ema(close,fast);b=ema(close,slow);cash=capital;qty=0.;basis=0.;fills=[];curve=[];fee/=10000;slip/=10000
    for i in range(2,len(bars)):
        prev=i-1;before=i-2
        up=a[prev]>b[prev] and a[before]<=b[before]
        down=a[prev]<b[prev] and a[before]>=b[before]
        if up and not qty:
            price=bars[i]['open']*(1+slip);qty=cash/(price*(1+fee));basis=cash;cash=0
            fills.append({'signal_at':bars[prev]['timestamp'],'at':bars[i]['timestamp'],'side':'BUY','price':price})
        elif down and qty:
            price=bars[i]['open']*(1-slip);received=qty*price*(1-fee)
            fills.append({'signal_at':bars[prev]['timestamp'],'at':bars[i]['timestamp'],'side':'SELL','price':price,'net_trade':received-basis})
            qty=0;cash=received
        curve.append((bars[i]['timestamp'],cash+qty*bars[i]['close']))
    peak=capital;dd=0
    for _,eq in curve:peak=max(peak,eq);dd=max(dd,1-eq/peak)
    final=curve[-1][1];real=[x['net_trade'] for x in fills if 'net_trade' in x];gain=sum(x for x in real if x>0);loss=-sum(x for x in real if x<0)
    return {'engine':'offline_ema_cross_v1','no_exchange_orders':True,'bars':len(bars),'fast':fast,'slow':slow,'return_pct':(final/capital-1)*100,'max_drawdown_pct':dd*100,'realized_trades':len(real),'profit_factor':gain/loss if loss else None,'fills':fills[:2000],'equity_curve':curve[::max(1,len(curve)//350)]}
def training(bars,p):
    try:epochs=int(p.get('epochs',70));lr=float(p.get('learning_rate',.08))
    except (TypeError,ValueError,OverflowError):raise InputError('Invalid training parameters') from None
    if not(10<=epochs<=120 and .001<=lr<=.2):raise InputError('Training params out of range')
    close=[b['close'] for b in bars];samples=[]
    for i in range(16,len(close)-1):
        features=[close[i]/close[i-k]-1 for k in (1,3,8,16)]
        if all(math.isfinite(x) for x in features):samples.append((features,int(close[i+1]>close[i])))
    samples=samples[-5000:]
    if len(samples)<70:raise InputError('Insufficient train/OOS sample')
    split=int(.8*len(samples));train=samples[:split];test=samples[split:]
    means=[sum(f[j] for f,_ in train)/len(train) for j in range(4)]
    std=[max((sum((f[j]-means[j])**2 for f,_ in train)/len(train))**.5,1e-8) for j in range(4)]
    norm=lambda f:[(f[j]-means[j])/std[j] for j in range(4)]
    sig=lambda z:1/(1+math.exp(-max(-35,min(35,z))))
    w=[0.]*4;bias=0.
    for _ in range(epochs):
        gradient=[0.]*4;gb=0.
        for f,y in train:
            x=norm(f);delta=sig(bias+sum(i*j for i,j in zip(w,x)))-y;gb+=delta
            for j in range(4):gradient[j]+=delta*x[j]
        w=[max(-20,min(20,w[j]-lr*(gradient[j]/len(train)+.002*w[j]))) for j in range(4)]
        bias=max(-20,min(20,bias-lr*gb/len(train)))
    y=[label for _,label in test];prob=[sig(bias+sum(i*j for i,j in zip(w,norm(f)))) for f,_ in test];pred=[int(v>=.5) for v in prob]
    accuracy=sum(a==b for a,b in zip(pred,y))/len(y)
    f1=[]
    for klass in (0,1):
        tp=sum(a==klass and b==klass for a,b in zip(pred,y));fp=sum(a==klass and b!=klass for a,b in zip(pred,y));fn=sum(a!=klass and b==klass for a,b in zip(pred,y))
        precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0;f1.append(2*precision*recall/(precision+recall) if precision+recall else 0)
    ece=0.
    for k in range(10):
        b=[(q,t) for q,t in zip(prob,y) if k/10<=q<(k+1)/10 or k==9 and q==1]
        if b:ece+=len(b)/len(y)*abs(sum(q for q,_ in b)/len(b)-sum(t for _,t in b)/len(b))
    return {'engine':'offline_logistic_research_v1','never_auto_deploy':True,'samples':len(samples),'train_samples':len(train),'oos_samples':len(test),'accuracy':accuracy,'macro_f1':sum(f1)/2,'ece_10bins':ece,'research_model':{'weights':w,'bias':bias,'mean':means,'std':std},'disclaimer':'OOS alone is NOT deployment or financial-edge proof'}
class JobStore:
    def __init__(self,home):
        self.home=Path(home).resolve();self.home.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.datasets=self.home/'datasets';self.artifacts=self.home/'artifacts'
        for p in (self.datasets,self.artifacts):p.mkdir(mode=0o700,exist_ok=True)
        self.db=self.home/'jobs.sqlite3'
        with self.db_conn() as db:
            db.execute('CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, owner TEXT,sha TEXT,created TEXT,rows INTEGER)')
            db.execute("""CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,owner TEXT,kind TEXT,dataset_id TEXT,params TEXT,state TEXT,created TEXT,started TEXT,finished TEXT,result_sha TEXT,error TEXT,idempotency TEXT,cancel INTEGER DEFAULT 0,UNIQUE(owner,idempotency))""")
    def db_conn(self):
        conn=sqlite3.connect(str(self.db),timeout=10);conn.row_factory=sqlite3.Row;conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA busy_timeout=10000');return conn
    def upload(self,owner,text):
        bars=parsed_csv(text);raw=text.encode('utf-8');id_=str(uuid.uuid4());sha=digest(raw)
        fd=os.open(self.datasets/(id_+'.csv'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        with self.db_conn() as db:db.execute('INSERT INTO datasets VALUES(?,?,?,?,?)',(id_,owner,sha,now(),len(bars)))
        return {'dataset_id':id_,'sha256':sha,'rows':len(bars)}
    def create(self,owner,kind,dataset_id,params,key):
        if kind not in ('backtest','training') or not isinstance(params,dict) or not isinstance(key,str) or not(10<=len(key)<=120):raise InputError('Invalid job request/idempotency')
        # Validate before adding to durable queue.
        if kind=='backtest':backtest([{'timestamp':'x','open':1,'high':1,'low':1,'close':1}]*3,params)
        else:
            try:e=int(params.get('epochs',70));lr=float(params.get('learning_rate',.08))
            except (ValueError,TypeError,OverflowError):raise InputError('Invalid train params') from None
            if not(10<=e<=120 and .001<=lr<=.2):raise InputError('Invalid train params')
        with self.db_conn() as db:
            db.execute('BEGIN IMMEDIATE')
            existing=db.execute('SELECT * FROM jobs WHERE owner=? AND idempotency=?',(owner,key)).fetchone()
            if existing:
                if (existing['kind'],existing['dataset_id'],existing['params'])!=(kind,dataset_id,dumps(params)):raise InputError('Idempotency replay mismatch')
                return dict(existing)
            if not db.execute('SELECT id FROM datasets WHERE owner=? AND id=?',(owner,dataset_id)).fetchone():raise InputError('Unknown dataset')
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]>=12:raise InputError('Queue full')
            id_=str(uuid.uuid4())
            db.execute('INSERT INTO jobs(id,owner,kind,dataset_id,params,state,created,idempotency) VALUES(?,?,?,?,?,?,?,?)',(id_,owner,kind,dataset_id,dumps(params),'queued',now(),key))
            return dict(db.execute('SELECT * FROM jobs WHERE id=?',(id_,)).fetchone())
    def list(self,owner):
        with self.db_conn() as db:return [dict(x) for x in db.execute('SELECT * FROM jobs WHERE owner=? ORDER BY created DESC LIMIT 40',(owner,))]
    def get(self,owner,id_):
        with self.db_conn() as db:row=db.execute('SELECT * FROM jobs WHERE owner=? AND id=?',(owner,id_)).fetchone()
        if not row:raise InputError('Job missing/unauthorized')
        return dict(row)
    def cancel(self,owner,id_):
        self.get(owner,id_)
        with self.db_conn() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE jobs SET cancel=1,state=CASE WHEN state='queued' THEN 'cancelled' ELSE state END WHERE owner=? AND id=? AND state IN ('queued','running')",(owner,id_))
        return self.get(owner,id_)
    def recover(self):
        with self.db_conn() as db:db.execute("UPDATE jobs SET state='interrupted',finished=?,error='Worker restarted, manual retry' WHERE state='running'",(now(),))
    def claim(self):
        with self.db_conn() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if not row:return None
            db.execute("UPDATE jobs SET state='running',started=? WHERE id=?",(now(),row['id']))
            return dict(db.execute('SELECT * FROM jobs WHERE id=?',(row['id'],)).fetchone())
    def finish(self,id_,state,data=None,error=''):
        sha=None
        if data is not None:
            raw=dumps(data).encode('utf-8');sha=digest(raw);tmp=self.artifacts/(id_+'.tmp')
            with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            tmp.replace(self.artifacts/(id_+'.json'))
        with self.db_conn() as db:db.execute('UPDATE jobs SET state=?,finished=?,result_sha=?,error=? WHERE id=?',(state,now(),sha,error[:1000],id_))
    def artifact(self,owner,id_):
        j=self.get(owner,id_)
        if j['state']!='completed' or not j['result_sha']:raise InputError('Artifact unavailable')
        raw=(self.artifacts/(id_+'.json')).read_bytes()
        if digest(raw)!=j['result_sha']:raise InputError('Artifact tamper detected')
        return json.loads(raw)
class JobWorker:
    def __init__(self,store,interval=.3):
        self.store=store;self.interval=interval;self.stop_event=threading.Event();self.thread=None;self.lease=None
    def start(self):
        if self.thread and self.thread.is_alive():return
        lock=(self.store.home/'worker.lock').open('a+b')
        try:
            if os.name=='nt':
                import msvcrt
                lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            lock.close();raise RuntimeError('A worker already owns the job store')
        self.lease=lock;self.store.recover();self.stop_event.clear()
        self.thread=threading.Thread(target=self.loop,name='studio-offline',daemon=True);self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=15)
        if not self.thread or not self.thread.is_alive():
            if self.lease:self.lease.close();self.lease=None
    def loop(self):
        while not self.stop_event.wait(self.interval):
            job=self.store.claim()
            if not job:continue
            try:
                with self.store.db_conn() as db:c=db.execute('SELECT cancel FROM jobs WHERE id=?',(job['id'],)).fetchone()[0]
                if c:self.store.finish(job['id'],'cancelled');continue
                path=self.store.datasets/(job['dataset_id']+'.csv');raw=path.read_bytes();bars=parsed_csv(raw.decode('utf-8'))
                if job['kind']=='backtest':out=backtest(bars,json.loads(job['params']))
                else:out=training(bars,json.loads(job['params']))
                out.update({'job_id':job['id'],'dataset_sha256':digest(raw),'completed_utc':now()})
                with self.store.db_conn() as db:c=db.execute('SELECT cancel FROM jobs WHERE id=?',(job['id'],)).fetchone()[0]
                self.store.finish(job['id'],'cancelled' if c else 'completed',None if c else out)
            except Exception:
                self.store.finish(job['id'],'failed',error='Offline job failed: inspect private logs')