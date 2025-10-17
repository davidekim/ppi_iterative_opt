import unittest
from icecream import ic
import subprocess
from pathlib import Path

from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

import test_utils
import run_inference

REWRITE = False
def infer(overrides):
    run_inference.make_deterministic()
    initialize(version_base=None, config_path="config/inference", job_name="test_app")
    conf = compose(config_name='base.yaml', overrides=overrides, return_hydra_config=True)
    # This is necessary so that when the model_runner is picking up the overrides, it finds them set on HydraConfig.
    HydraConfig.instance().set_config(conf)
    conf = compose(config_name='base.yaml', overrides=overrides)
    ic(conf.contigmap)
    run_inference.main(conf)
    p = Path(conf.inference.output_prefix + '_0.pdb')
    return p.read_text()

class TestRegression(unittest.TestCase):
    
    # Example regression test.
    def test_t2(self):
        pdb_contents = infer([
            'diffuser.T=2',
            'inference.num_designs=1',
            'inference.cautious=False',
            'inference.output_prefix=tmp/test',
            "contigmap.contigs=['20', 'A3-23', '30']",
        ])
        test_utils.assert_matches_golden(self, 'T2', pdb_contents, rewrite=REWRITE)

if __name__ == '__main__':
        unittest.main()
