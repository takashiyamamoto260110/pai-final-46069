"""Summarize recorded robot experiments without requiring a GPU or simulator.

Input: directories containing result.json and trace.json.
Output: task results and contact windows; no absolute source paths are exported.
"""
import argparse
import json
from pathlib import Path


def summarize(source):
    reports = []
    for path in sorted(source.glob('*/*/result.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        trace = json.loads(path.with_name('trace.json').read_text(encoding='utf-8'))
        windows = {}
        for obj, step in result.get('first_crossing', {}).items():
            windows[obj] = [dict(
                step=row['step'], displacement_mm=row['displacement_l1'][obj]*1000,
                eef_z_m=row['eef_pos'][2],
                contacts=sorted({tuple(pair) for pair in row['contacts']
                    if any(obj in (name or '') for name in pair)}),
            ) for row in trace if step-4 <= row['step'] <= step+3]
        reports.append(dict(task=result['task'], mode=result['mode'],
            initial_state_index=result['initial_state_index'],
            goal_reached=result['done'], success_with_displacement_check=result['success'],
            steps=result['steps'], crossing_windows=windows,
            max_displacement_l1_m=result['max_displacement_l1_m']))
    if not reports:
        raise ValueError('No matching result.json files')
    return dict(displacement_threshold_m=0.001,
        limitation='Displacement proxy, not a complete physical collision detector; selected demonstrations, not a benchmark success rate.',
        episodes=reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=summarize(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Analyzed {len(report['episodes'])} episodes")
