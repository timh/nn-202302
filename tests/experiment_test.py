import unittest
import sys

sys.path.append("..")
from experiment import Experiment

usually_same = set(('sched_type sched_warmup_epochs do_compile loss_type skip '
                    'batch_size optim_type max_epochs').split())
# same_fields = 'sched_warmup_epochs optim_type '
class TestExperiment(unittest.TestCase):
    def test_simple_same(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="foo")

        same, fields_same, fields_diff = exp1.is_same(exp2)
        self.assertEqual(True, same)
        self.assertEqual({'label'} | usually_same, fields_same)
        self.assertEqual(set(), fields_diff)

    def test_simple_notsame(self):
        exp1 = Experiment(label="foo")
        exp2 = Experiment(label="bar")

        same, fields_same, fields_diff = exp1.is_same(exp2)
        self.assertEqual(False, same)
        self.assertEqual(usually_same, fields_same)
        self.assertEqual({'label'}, fields_diff)
