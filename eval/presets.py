"""Portable suite selections; no downloads, inference, or model-name policy selection."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('quick', 'decision', 'full-text', 'vision', 'full')


def suite_paths(name):
    if name not in NAMES:
        raise ValueError(f'Unknown preset: {name}')
    if name == 'quick':
        return [ROOT / 'suites' / p for p in (
            'jevbench-public.json', 'english/semif-authored.json',
            'english/semif-typesafe.json')]
    selected = json.loads((ROOT / 'full-suites.json').read_text(encoding='utf-8'))
    paths = selected['text'] if name in ('decision', 'full-text') else selected['image']
    if name == 'full':
        paths = selected['text'] + selected['image']
    paths = [ROOT.parent / p for p in paths]
    if name == 'decision':
        from catalog import entries
        catalog = {suite['id']: suite for _, suite in entries()}
        def contains_decision(suite):
            children = suite.get('aggregate_children')
            return (any(contains_decision(catalog[child]) for child in children)
                    if children else _decision(suite))
        # Keep mixed parents whole: category reports retain native project-wide
        # coverage. CodeMMLU's decision children must not be missed merely because
        # its historical parent metadata says model-knowledge.
        paths = [p for p in paths if contains_decision(json.loads(p.read_text(encoding='utf-8')))]
    return paths


def _decision(suite):
    return (suite['language_group'] == 'english'
            and suite['subcategory'] == 'classification-decision')


def describe(paths):
    result = []
    for path in paths:
        suite = json.loads(path.read_text(encoding='utf-8'))
        dataset = (path.parent / suite['dataset']).resolve()
        result.append({'suite': suite['id'], 'manifest': str(path),
                       'dataset': str(dataset), 'dataset_present': dataset.is_file()})
    return result
