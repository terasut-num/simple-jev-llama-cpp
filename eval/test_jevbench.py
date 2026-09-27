"""JevBench accuracy and native request regression checks, no GPU required."""
import copy
import json
from pathlib import Path
import unittest
from adapters import jevbench
from suites import load_suite

ROOT = Path(__file__).parent


class JevBenchTests(unittest.TestCase):
    def setUp(self):
        self.rows = load_suite(ROOT/'suites/jevbench-public.json')[3]

    def row(self, kind):
        return next(r for r in self.rows if r['question']['type'] == kind)

    def test_public_dataset_and_perfect_answers(self):
        self.assertEqual(len(self.rows), 231)
        records = []
        for row in self.rows:
            body = jevbench.request_for(row, 'test')
            self.assertEqual(set(body), {'model', 'state', 'questions'})
            self.assertEqual(body['questions']['decision'], row['question'])
            answer = {'type': row['question']['type']}
            if answer['type'] == 'noul':
                answer['noul'] = float(row['expected'] == 'yes')
            else:
                answer['probabilities'] = {k: float(k == str(row['expected'])) for k in row['labels']}
            record = jevbench.parse_response(row, {'answers': {'decision': answer}})
            records.append(dict(record, id=row['id']))
        summary = jevbench.summarize(self.rows, records)
        self.assertEqual(summary['correct'], 231)
        self.assertEqual(summary['accuracy'], 1)
        self.assertEqual(summary['failed_rows'], 0)
        self.assertEqual({k:v['rows'] for k,v in summary['by_tier'].items()},
                         {'easy':48, 'original':72, 'hard':111})

    def test_score_uses_argmax_not_rounded_expectation(self):
        row = copy.deepcopy(self.row('score'))
        row['expected'] = 0
        p = dict(zip(row['labels'], [.4, .05, .2, .35]))
        record = jevbench.parse_response(row, {'answers': {'decision': {'type':'score', 'probabilities':p, 'score':1.5}}})
        self.assertEqual(record['predicted'], '0')
        self.assertTrue(record['correct'])

    def test_noul_tie_matches_upstream_lexical_order(self):
        row = self.row('noul')
        record = jevbench.parse_response(row, {'answers': {'decision': {'type':'noul','noul':.5}}})
        self.assertEqual(record['predicted'], 'no')
        for bad in (True, float('nan'), float('inf'), -1, 2):
            with self.assertRaises(ValueError):
                jevbench.parse_response(row, {'answers': {'decision': {'type':'noul','noul':bad}}})

    def test_invalid_distribution_and_rounding(self):
        row = self.row('choice')
        n = len(row['labels'])
        for p in ({}, {k:float('nan') for k in row['labels']}, {k:.01 for k in row['labels']}):
            with self.assertRaises(ValueError):
                jevbench.parse_response(row, {'answers': {'decision': {'type':'choice','probabilities':p}}})
        p = {k:1.01/n for k in row['labels']}
        r = jevbench.parse_response(row, {'answers': {'decision': {'type':'choice','probabilities':p}}})
        self.assertTrue(r['renormalized'])
        self.assertAlmostEqual(sum(r['probabilities'].values()),1)

    def test_failures_count_wrong_and_null_gold_is_unscorable(self):
        rows = [copy.deepcopy(self.row('choice')) for _ in range(3)]
        for i,r in enumerate(rows):r['id']=str(i)
        rows[-1]['expected']=None
        result = jevbench.summarize(rows,[{'id':'0','correct':True,'probabilities':{}},
                                        {'id':'1','error':'HTTP 422'}, {'id':'2','error':'HTTP 500'}])
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['failed_rows'],2)
        self.assertEqual(result['unscorable_rows'],1)

    def test_invalid_fixture_before_requests(self):
        row=copy.deepcopy(self.row('choice'));row['expected']='not-a-label'
        with self.assertRaises(ValueError):jevbench.validate([row])
        with self.assertRaises(ValueError):jevbench.validate([self.rows[0],self.rows[0]])

    def test_published_reference_matches_exact_ids(self):
        ref=json.loads((ROOT/'vendor/jevbench/jev-public-reference.json').read_text(encoding='utf-8'))
        self.assertEqual(set(ref['outcomes']),{r['id'] for r in self.rows})
        self.assertEqual(sum(v=='c' for v in ref['outcomes'].values()),200)
