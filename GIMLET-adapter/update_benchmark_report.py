"""Merge the completed six-dataset report and an independent extension queue.

Only complete, supervisor-validated dataset means are published. Incomplete
datasets show counts, not a potentially misleading partial mean.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time

STRATEGIES = ('shared', 'graph', 'text', 'separate')
GROUPS = ('shared_post',) + STRATEGIES
FIELDS = ['dataset', 'metric', 'original_shared_post'] + [s + '_pre' for s in STRATEGIES]
LABELS = dict(bace='BACE', hiv='HIV', bbbp='BBBP', tox21='Tox21', esol='ESOL',
              freesolv='FreeSolv', muv='MUV', cyp450='CYP450', toxcast='ToxCast')


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, text):
    if path.exists() and path.read_text() == text:
        return
    temporary = path.with_name(path.name + '.combined.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def progress_rows(path, job):
    if not path.exists():
        return 0
    allowed = {m['task']: m['protocol_sha256'] for m in job['manifests']}
    completed = set()
    for row in read_csv(path):
        try:
            task = int(row['split'])
            valid = (allowed.get(task) == row.get('protocol_sha256')
                     and row.get('adapter_strategy') == job['adapter']['adapter_strategy']
                     and row.get('adapter_placement') == job['adapter']['adapter_placement']
                     and math.isfinite(float(row['test_best'])))
            if valid:
                completed.add(task)
        except (KeyError, ValueError, TypeError):
            pass  # A writer may still be appending the final row.
    return len(completed)


def refresh(base, extension):
    original = read_json(base / 'legacy_shared_post.json')
    base_rows = {r['dataset']: r for r in read_csv(base / 'summary.csv')}
    config = read_json(extension / 'run_config.json')
    status = read_json(extension / 'status.json')
    audit = read_json(Path(config['protocol_dir']) / 'audit.json')
    scores = {r['dataset']: r for r in read_csv(extension / 'summary.csv')}
    jobs = {j['job_id']: j for j in config['jobs']}
    rows, displays = [], []
    for legacy in original['results']:
        dataset = legacy['dataset']
        row = dict(dataset=dataset, metric=legacy['metric'], original_shared_post=legacy['test_best'])
        row.update({s + '_pre': base_rows[dataset][s] for s in STRATEGIES})
        rows.append(row)
        displays.append([LABELS[dataset], '残差标准差 ↓' if dataset in ('esol', 'freesolv') else 'ROC-AUC ↑']
                        + ['{:.6f}'.format(float(row[f])) for f in FIELDS[2:]])
    for dataset in config['datasets']:
        data_audit = audit['datasets'][dataset]
        row = dict(dataset=dataset, metric='roc_auc')
        values = []
        for group, field in zip(GROUPS, FIELDS[2:]):
            job_id = group + '/' + dataset
            entry = status['jobs'][job_id]
            score = scores[dataset].get(group, '') if entry['state'] == 'completed' else ''
            row[field] = score
            if score != '':
                values.append('{:.6f}'.format(float(score)))
            elif entry['state'] == 'pending':
                values.append('待运行')
            else:
                n = progress_rows(extension / group / (dataset + '.csv'), jobs[job_id])
                state = '运行中' if entry['state'] == 'running' else entry['state']
                values.append('{} {}/{}'.format(state, n, len(data_audit['included'])))
        rows.append(row)
        displays.append(['{}（{}/{} 项）'.format(LABELS[dataset], len(data_audit['included']), data_audit['total_tasks']),
                         'ROC-AUC ↑'] + values)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader(); writer.writerows(rows)
    write(base / 'comparison_with_legacy.csv', stream.getvalue())
    counts = {state: sum(j['state'] == state for j in status['jobs'].values())
              for state in ('completed', 'running', 'pending')}
    lines = ['# Adapter 适配策略实验与原配置对比', '',
        '原六数据集：24/24 组已完成；新增三数据集：{}。'.format(status['state']),
        '新增 15 组：已完成 {completed}，运行中 {running}，待运行 {pending}。'.format(**counts), '',
        'shared/post 为原来的层后共享 Adapter；另外四组均为层前插入。原六数据集的 shared/post 引用历史结果，新增三数据集的 shared/post 重新训练。',
        '各组训练 Adapter 和绑定的 lm_head/input embedding；Adapter 参数：共享/图专用/文本专用 402,816，分离 399,732（少 0.77%）。',
        '各组保持 runseed=6、seed=42、scaffold 80/10/10、60 epochs、目标有效 batch=128、Adam lr=0.01；新增协议的末批按实际样本数更新。',
        '分类为 ROC-AUC ↑，回归为残差标准差 ddof=1 ↓（不是 RMSE）。所有多任务数据集在固定有效任务集合上等权平均。', '',
        '| 数据集 | 指标 | 原配置 shared/post | shared/pre | graph/pre | text/pre | separate/pre |',
        '|---|---|---:|---:|---:|---:|---:|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in displays]
    rel = os.path.relpath(extension, base)
    protocol_rel = os.path.relpath(config['protocol_dir'], base)
    lines += ['', '**新增数据集的抽样与任务范围**', '',
        '原六数据集沿用旧的逐任务缩小训练池协议，原始分数不变。新增三数据集采用 independent_labeled_v1：每个任务从完整 scaffold 训练池独立抽样，每类最多 128 个有标签样本，采样 seed=6+task_id；验证/测试仅过滤缺失标签，不改变有标签分子的归属。',
        '全部五组共享同一份任务清单、样本、prompt 和切分哈希。仅纳入有 prompt 且 train/valid/test 均有正负类的任务；排除规则在训练前固定，未根据模型分数选择任务。',
        'CYP450 已按原始列顺序校正 CYP2C19 与 CYP2C9 的 prompt 对应关系。MUV/ToxCast 的有效任务子集不能冒充全任务基准。']
    lines.append('新增协议对梯度累积末批进行更新并按实际样本数归一化，避免不足 128 个训练样本的任务没有参数更新；原六数据集的训练循环和结果不变。')
    for dataset, info in audit['datasets'].items():
        reasons = {}
        for excluded in info['excluded']:
            for reason in excluded['reasons']:
                reasons[reason] = reasons.get(reason, 0) + 1
        lines.append('- {}：纳入 {}/{} 项；排除原因计数 {}（同一任务可能有多个原因）。'.format(
            LABELS[dataset], len(info['included']), info['total_tasks'], reasons or '无'))
    lines += ['', '任务清单与排除明细：[audit.json]({}/audit.json)。'.format(protocol_rel),
        '新增任务状态：[status.json]({}/status.json)；新增原始结果和日志位于 `{}/{{shared_post,shared,graph,text,separate}}/`。'.format(rel, rel),
        '完整对比表：[comparison_with_legacy.csv](comparison_with_legacy.csv)。历史六数据集来源：[legacy_shared_post.json](legacy_shared_post.json)。',
        '运行中单元格显示已写入任务数，不报告部分任务的平均分；完整均值只在队列校验通过后发布。',
        '这是单种子实验。shared/post 与其他新策略的比较同时涉及插入位置变化，不能将差异完全归因于适配策略。']
    if status.get('error'):
        lines += ['', '新增队列停止原因：' + status['error']]
    write(base / 'RESULTS.md', '\n'.join(lines) + '\n')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_root', type=Path, required=True)
    parser.add_argument('--extension_root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    with (args.base_root / 'comparison_report.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.base_root / 'comparison_report.pid').write_text(str(os.getpid()) + '\n')
        while True:
            status = refresh(args.base_root, args.extension_root)
            proc = Path('/proc/{}/cmdline'.format(status['launcher_pid']))
            try:
                alive = str(args.extension_root.resolve()).encode() in proc.read_bytes()
            except FileNotFoundError:
                alive = False
            if not args.watch or not alive:
                refresh(args.base_root, args.extension_root)
                print('Combined report written; updater finished.', flush=True)
                return
            time.sleep(10)


if __name__ == '__main__':
    main()
