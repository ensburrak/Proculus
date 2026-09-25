"""No-exchange V9 worker security and arithmetic regression checks (stdlib unittest)."""
from __future__ import annotations
import math,tempfile,time,unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path
from studio_control.research_jobs import JobStore,JobWorker,InputError,parsed_csv

def fixture(count=185):
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    rows=['timestamp,open,high,low,close']
    for i in range(count):
        p=125+i*.19+3*math.sin(i*.14)
        rows.append(f"{(start+timedelta(minutes=i)).strftime('%Y-%m-%dT%H:%M:%SZ')},{p:.7f},{p+2:.7f},{p-2:.7f},{p:.7f}")
    return '\n'.join(rows)+'\n'
def wait(store,owner,id_,limit=12):
    until=time.monotonic()+limit
    while time.monotonic()<until:
        item=store.get(owner,id_)
        if item['state'] not in ('queued','running'):return item
        time.sleep(.02)
    raise AssertionError('offline worker did not settle')
class OfflineV9Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=JobStore(Path(self.temp.name)/'private')
        self.dataset=self.store.upload('alice',fixture())
    def tearDown(self):self.temp.cleanup()
    def test_csv_no_duplicates_and_boundaries(self):
        self.assertEqual(len(parsed_csv(fixture())),185)
        with self.assertRaises(InputError):
            parsed_csv(fixture().replace('2025-01-01T00:01:00Z','2025-01-01T00:00:00Z'))
        with self.assertRaises(InputError):
            parsed_csv('timestamp,open,high,low,close\n')
    def test_auth_and_idempotency(self):
        self.assertEqual(self.dataset['rows'],185)
        job=self.store.create('alice','backtest',self.dataset['dataset_id'],{'fast':7,'slow':22},'job-once-idemp-v9')
        self.assertEqual(self.store.create('alice','backtest',self.dataset['dataset_id'],{'fast':7,'slow':22},'job-once-idemp-v9')['id'],job['id'])
        with self.assertRaises(InputError):self.store.create('alice','training',self.dataset['dataset_id'],{},'job-once-idemp-v9')
        with self.assertRaises(InputError):self.store.get('bob',job['id'])
        with self.assertRaises(InputError):self.store.create('bob','backtest',self.dataset['dataset_id'],{},'bob-unknown-dataset')
    def test_durable_backtest_and_training(self):
        worker=JobWorker(self.store,.01);worker.start()
        try:
            a=self.store.create('alice','backtest',self.dataset['dataset_id'],{'fast':7,'slow':22},'job-bt-v9-safe-id')
            self.assertEqual(wait(self.store,'alice',a['id'])['state'],'completed')
            output=self.store.artifact('alice',a['id'])
            self.assertTrue(output['no_exchange_orders'])
            self.assertTrue(all(f['signal_at']<f['at'] for f in output['fills']))
            self.assertEqual(output['dataset_sha256'],self.dataset['sha256'])
            b=self.store.create('alice','training',self.dataset['dataset_id'],{'epochs':15},'job-ml-v9-safe-id')
            self.assertEqual(wait(self.store,'alice',b['id'])['state'],'completed')
            m=self.store.artifact('alice',b['id'])
            self.assertTrue(m['never_auto_deploy'])
            self.assertGreater(m['oos_samples'],0)
        finally:worker.stop()
        reread=JobStore(self.store.home)
        self.assertEqual(reread.get('alice',a['id'])['state'],'completed')
        self.assertEqual(reread.artifact('alice',a['id'])['dataset_sha256'],self.dataset['sha256'])
        path=self.store.artifacts/(a['id']+'.json')
        path.write_text('{}',encoding='utf-8')
        with self.assertRaises(InputError):self.store.artifact('alice',a['id'])
    def test_worker_lease_and_restart(self):
        a=self.store.create('alice','backtest',self.dataset['dataset_id'],{},'job-lease-v9-id')
        w=JobWorker(self.store,4);w.start()
        try:
            with self.assertRaises(RuntimeError):JobWorker(JobStore(self.store.home),4).start()
        finally:w.stop()
        # An in-progress task is marked interrupted on explicit recovery, never auto-marked successful.
        another=JobStore(self.store.home)
        self.assertIn(another.get('alice',a['id'])['state'],('queued','running','interrupted'))
        another.recover()
        self.assertNotEqual(another.get('alice',a['id'])['state'],'completed')
    def test_malformed_params_no_queue_write(self):
        for params in ({'fast':'oops'},{'fast':100},{'capital':float('inf')}):
            with self.assertRaises(InputError):self.store.create('alice','backtest',self.dataset['dataset_id'],params,'bad-bt-'+str(params))
        for params in ({'epochs':'nan'},{'learning_rate':5}):
            with self.assertRaises(InputError):self.store.create('alice','training',self.dataset['dataset_id'],params,'bad-ml-'+str(params))
        self.assertFalse(self.store.list('alice'))
if __name__=='__main__':unittest.main()
