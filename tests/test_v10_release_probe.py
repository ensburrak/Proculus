"""V10: preflight can never approve live from repository files alone."""
import unittest
import tempfile
import json
from pathlib import Path
from web.live_readiness_probe import inspect

class ProductionGateTests(unittest.TestCase):
    def test_missing_canonical_runtime_is_visible_and_live_denied(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'config.json').write_text(json.dumps({
              'paper_trading':{'enabled':True},'live_readiness':{},
              'live_safety':{'canary':{'enabled':True},
                             'independent_watchdog':{'required':True},
                             'capital_isolation':{'required':True}}
            }),encoding='utf8')
            report=inspect(root)
            self.assertFalse(report['production_authorized'])
            self.assertFalse(report['live_exchange_order_endpoint'])
            self.assertIn('canonical_backtesting',report['blockers'])
            self.assertIn('canonical_ml_training',report['blockers'])
            self.assertIn('actual_cloud_https_iam_mfa_unverified',report['blockers'])
    def test_all_fabricated_file_presence_does_not_equal_production_authority(self):
        from web.live_readiness_probe import FILES
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for relative in FILES.values():
                path=root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('placeholder')
            (root/'config.json').write_text('{}')
            report=inspect(root)
            self.assertTrue(all(report['repo_module_presence'].values()))
            self.assertFalse(report['production_authorized'])
            self.assertIn('config:canary_configured',report['blockers'])
if __name__=='__main__':unittest.main()
