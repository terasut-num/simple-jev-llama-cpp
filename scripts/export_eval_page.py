#!/usr/bin/env python3
"""Export a frozen, auditable website snapshot; reads experiment checkouts only.

Usage: python3 scripts/export_eval_page.py --workspace /workspace/open-jev
No model calls. No writes outside website/assets/evaluations/.
"""
import argparse
import hashlib
import json
from pathlib import Path

MODELS = [
    ('qwen27b', 'Qwen3.8-27B', 'examples_binary', 'qwen27b-examples95', 214),
    ('gemma-moe', 'Gemma4-26B-A4B', 'strict_mix_repeat2', 'gemma-moe-repeat95-ready', 207),
    ('qwen-moe', 'Qwen3.6-35B-A3B', 'repeat_state', 'qwen-moe-context_repeat95', 204),
    ('jev', 'Jev 1.13', 'Native', None, 200),
    ('gemma12b', 'Gemma4-12B', 'strict_mix_repeat2', 'gemma12b-family-repeat95', 199),
    ('qwen4b', 'Qwen3.5-4B', 'strict_mix_repeat2', 'qwen4b-family-repeat95', 178),
]
DESCRIPTIONS = {
    'CodeComplex': 'Choose the time-complexity class of a program.',
    'CodeMMLU': 'Multiple-choice programming decisions across the valid-choice release, including completion, repair and execution prediction.',
    'ContractNLI': 'Decide whether a contract entails, contradicts, or does not mention a proposed statement.',
    'Jev Korean benchmark': 'English–English arm only: passage comprehension and sentence equivalence; not a Korean-language result.',
    'Jev Phishing Bench': 'Identify phishing from message evidence.',
    'Jev Sec Bench': 'Classify security properties of code using the labelled binary questions.',
    'JevBench': 'Native Choice, Score and Noul decisions, grouped here by difficulty and domain. This is the separate full-run evaluation, not the prompt-selection run above.',
    'Jevtest': 'A small customer-support subset testing policy-grounded decisions.',
    'LexGLUE': 'Legal understanding: select a matching holding (casehold), or identify unfair terms of service (unfair-tos).',
    'MetaTool': 'Decide whether a request needs an external tool.',
    'SemIf': 'Conditional decisions: authored scenarios, meaning-preserving perturbations, TypeSafe business rules, or WANLI inference.',
    'wondertwins/jev-benchmark': 'Identify which game characters a player is addressing. Clean speech, speech-to-text, and misheard speech are separate configurations.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.workspace.resolve()
    out = Path(__file__).resolve().parents[1] / 'website/assets/evaluations'
    sources = {}

    def read(path):
        path = root / path
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return raw

    def load(path):
        return json.loads(read(path))

    def lines(path):
        return [json.loads(line) for line in read(path).splitlines() if line.strip()]

    ev = Path('simple-jev-eval/eval')
    lab = Path('simple-jev-prompt-lab')
    quick = lab / 'experiments/prompt_lab/results'
    final = load('reports/FULL_EVAL_FINAL_COMPARISON.json')
    public = lines(ev / 'data/jevbench-public.jsonl')
    published = load(ev / 'vendor/jevbench/jev-public-reference.json')
    assert len(public) == len({r['id'] for r in public}) == 231
    assert set(published['outcomes']) == {r['id'] for r in public}
    assert published['correct'] == 200
    predictions = {}
    models = []
    for key, name, policy, run, expected in MODELS:
        if key == 'jev':
            scores = {r['id']: {'correct': published['outcomes'][r['id']] == 'c', 'predicted': None} for r in public}
        else:
            folder = quick / run / policy / 'jevbench-public'
            summary = load(folder / 'summary.json')
            assert summary['correct'] == expected and summary['failed_rows'] == 0
            records = lines(folder / 'predictions.jsonl')
            assert len(records) == 231 and {r['id'] for r in records} == set(published['outcomes'])
            scores = {r['id']: {'correct': r['correct'], 'predicted': r['predicted']} for r in records}
        assert sum(r['correct'] for r in scores.values()) == expected
        predictions[key] = scores
        items = final['reference']['items'] if key == 'jev' else final['text'][key]['items']
        decision = [r for r in items if r['task'] == 'classification-decision']
        assert len(decision) == 26
        models.append({'id': key, 'name': name, 'policy': policy, 'publicCorrect': expected,
                       'decisionScore': sum(r['score'] for r in decision) / len(decision)})

    breakdowns = {}
    for dimension in ['tier', 'family', 'primitive']:
        field = lambda row: row['question']['type'] if dimension == 'primitive' else row[dimension]
        groups = []
        for value in sorted({field(r) for r in public}):
            rows = [r for r in public if field(r) == value]
            groups.append({'name': value, 'rows': len(rows), 'scores': {
                key: sum(predictions[key][r['id']]['correct'] for r in rows) for key, *_ in MODELS}})
        breakdowns[dimension] = groups
    examples = [{**{k: r[k] for k in ['id', 'tier', 'family', 'state', 'question', 'expected']},
                 'answers': {key: predictions[key][r['id']] for key, *_ in MODELS}} for r in public]

    def item_key(row):
        return tuple(row[k] for k in ['project', 'configuration', 'category', 'task'])

    full_report = load(lab / 'experiments/full_eval/results_merged/qwen27b/text/report.json')
    decision_items = []
    for ref in final['reference']['items']:
        if ref['task'] != 'classification-decision':
            continue
        matches = [r for r in full_report['by_category'] if
                   (r['project'], r['configuration'], r['placement'][2], r['placement'][3]) == item_key(ref)
                   and r['placement'][:2] == ['text', 'english']]
        assert len(matches) == 1 and matches[0]['coverage'] == 'complete'
        suite_ids = matches[0]['suites']
        # Show one authentic, short case from the first constituent suite.
        suite_id = suite_ids[0]
        manifest_path = lab / 'eval/suites/english' / (suite_id + '.json')
        manifest = load(manifest_path)
        # Cross-check the evaluator checkout explicitly, not just the prompt-lab copy.
        canonical = load(ev / 'suites/english' / (suite_id + '.json'))
        assert canonical == manifest, suite_id
        dataset = (root / manifest_path).parent / manifest['dataset']
        rows = lines(dataset.resolve().relative_to(root))
        selection = manifest.get('row_filter')
        if selection:
            rows = [r for r in rows if r[selection['field']] in selection['values']]
        row = min(rows, key=lambda r: (len(json.dumps(r['state'])), r['id']))
        if manifest['adapter'] == 'binary-battery-v1':
            questions = {k: row['questions'][k] for k in row['targets']}
            gold = {k: {'label': bool(v['label']), 'positive_option': v.get('positive', 'yes')} for k, v in row['targets'].items()}
            scoring = 'Accuracy over labelled binary targets, using P(positive) ≥ 0.5. One input row can contain several targets; rows are not necessarily decisions.'
        else:
            questions = {'decision': row.get('native_question') or {
                'type': 'choice', 'instructions': row['question'],
                'criteria': {o['id']: o['description'] for o in row['options']}}}
            gold = {'decision': row['options'][row['label']]['id']}
            scoring = 'Accuracy from the highest-probability label; ties follow fixture order. Native Noul uses P(yes), and Score uses its level distribution, not a rounded expected score.'
        scores = {}
        for key, *_ in MODELS:
            items = final['reference']['items'] if key == 'jev' else final['text'][key]['items']
            found = [r for r in items if item_key(r) == item_key(ref)]
            assert len(found) == 1 and found[0]['rows'] == ref['rows']
            scores[key] = found[0]['score']
        decision_items.append({
            'id': suite_id, **{k: ref[k] for k in ['project', 'configuration', 'category', 'rows']},
            'questions': matches[0]['metrics'].get('targets', ref['rows']),
            'description': DESCRIPTIONS[ref['project']], 'scoring': scoring,
            'suites': suite_ids, 'scores': scores, 'source': manifest.get('source', {}),
            'scopeNote': manifest.get('scope_note', ''),
            'example': {'id': row['id'], 'suite': suite_id, 'state': row['state'],
                        'questions': questions, 'gold': gold},
        })
    assert len(decision_items) == 26
    for model in models:
        assert abs(sum(r['scores'][model['id']] for r in decision_items)/26-model['decisionScore']) < 1e-12
    # Use accuracy throughout the new vision headline, never average F1/MME points
    # with accuracy. Retain native benchmark metrics alongside it in drill-downs.
    vision_definitions = [
        ('vision-cifar10', 'CIFAR-10', 'test', 'Recognize the main object among ten classes.'),
        ('vision-oxford-pets', 'Oxford-IIIT Pets', 'test', 'Identify one of 37 cat or dog breeds, not just cat versus dog.'),
        ('vision-mme-perception', 'MME', 'perception', 'Yes/no visual perception across ten categories. The native MME score sums accuracy and paired-question accuracy across categories (maximum 2,000).'),
        ('vision-pope-adversarial', 'POPE', 'adversarial', 'Detect whether an object is present, with challenging absent-object distractors. Native headline metric: F1.'),
        ('vision-pope-popular', 'POPE', 'popular', 'Detect object presence using popular-object distractors. Native headline metric: F1.'),
        ('vision-pope-random', 'POPE', 'random', 'Detect object presence using randomly selected distractors. Native headline metric: F1.'),
        ('vision-tallyqa', 'TallyQA', 'test', 'Count objects in images, including simple and complex counting questions.'),
    ]
    vision_reports = {key: load(lab / 'experiments/full_eval/results_merged' / key / 'image/report.json')
                      for key, *_ in MODELS if key != 'jev'}
    vision_items = []
    images = {}
    for suite_id, project, configuration, description in vision_definitions:
        manifest_path = lab / 'eval/suites/english' / (suite_id + '.json')
        manifest = load(manifest_path)
        assert manifest == load(ev / 'suites/english' / (suite_id + '.json')), suite_id
        dataset = ((root / manifest_path).parent / manifest['dataset']).resolve()
        rows = lines(dataset.relative_to(root))
        # Select source examples without consulting any model's answers.
        # POPE variants share images: use distinct absent-object cases to make
        # their different negative-question sampling easier to illustrate.
        example_ids = {'vision-cifar10': 'image-000010', 'vision-pope-adversarial': '2',
                       'vision-pope-popular': '8', 'vision-pope-random': '14'}
        row = next(r for r in rows if r['id'] == example_ids[suite_id]) if suite_id in example_ids else rows[0]
        selection = ('Selected for visual clarity, not model performance.' if suite_id == 'vision-cifar10'
                     else 'An absent-object case on a distinct image, selected to illustrate this POPE variant, not model performance.' if project == 'POPE'
                     else 'First source case, not selected for model performance.')
        image_path = (dataset.parent / row['image_path']).resolve()
        image = read(image_path.relative_to(root))
        assert hashlib.sha256(image).hexdigest() == row['image_sha256']
        image_name = row['image_sha256'] + '.png'
        images[image_name] = image
        gold = row['options'][row['label']]['description'] if 'options' in row else row['answer']
        scores, native_scores = {'jev': None}, {'jev': None}
        for key, report in vision_reports.items():
            found = [r for r in report['by_project'] if (r['project'], r['configuration']) == (project, configuration)]
            assert len(found) == 1
            item = found[0]
            assert item['coverage'] == 'complete' and not item['missing_suites']
            assert item['metrics']['rows'] == len(rows) and item['metrics']['failed_rows'] == 0
            assert item['score'] == final['vision'][key][project + '/' + configuration]
            scores[key] = item['metrics']['accuracy']
            native_scores[key] = item['score']
        vision_items.append({
            'id': suite_id, 'project': project, 'configuration': configuration,
            'description': description, 'rows': len(rows), 'scores': scores,
            'nativeMetric': manifest['headline_metric'], 'nativeScores': native_scores,
            'source': manifest['source'],
            'example': {'id': row['id'], 'question': row['question'], 'gold': gold,
                        'selection': selection,
                        'options': row.get('options'), 'imageSha256': row['image_sha256'],
                        'image': 'assets/evaluations/images/' + image_name},
        })
    assert sum(item['rows'] for item in vision_items) == 63372
    for model in models:
        model['visionScore'] = None if model['id'] == 'jev' else sum(item['scores'][model['id']] for item in vision_items) / 7

    decision_rows = sum(item['rows'] for item in decision_items)
    decision_questions = sum(item['questions'] for item in decision_items)
    assert (decision_rows, decision_questions) == (21364, 33099)
    data = {
        'schemaVersion': 1, 'models': models, 'publicRows': 231,
        'decisionRows': decision_rows, 'decisionQuestions': decision_questions,
        'visionRows': 63372, 'visionItems': vision_items,
        'publishedReference': {k: published[k] for k in ['model', 'source', 'source_sha256', 'rows', 'correct', 'accuracy']},
        'publicRevision': '83831807458d7df424a1e53e5724f3a3ffe2cf89',
        'publicBreakdowns': breakdowns, 'decisionItems': decision_items,
        'sources': sources,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    for name, image in images.items():
        (out / 'images' / name).write_bytes(image)
    (out / 'results.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    (out / 'public-examples.json').write_text(json.dumps(examples, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    (out / 'LICENSE-jevbench.txt').write_bytes((root / ev / 'vendor/jevbench/LICENSE').read_bytes())
    print(f'Exported 6 models, 26 decision items, 7 vision configurations, 231 public examples to {out}')


if __name__ == '__main__':
    main()
