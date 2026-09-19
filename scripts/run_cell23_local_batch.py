"""Run a restartable two-phase cell2/cell3 batch on one 64-thread host."""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PAIRS = list(itertools.product(('cell2', 'cell3'), ('basal', 'apical'), range(3), range(1, 51)))
REQUIRED = (
    'soma_v_array.npy',
    'apic_v_array.npy',
    'apic_ica_array.npy',
    'trunk_v_array.npy',
    'basal_v_array.npy',
    'tuft_v_array.npy',
    'section_synapse_df.csv',
    'simulation_params.json',
)


def now():
    return datetime.now(timezone.utc).isoformat()


class Batch:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.batch_root = self.config_path.parent
        self.results_root = Path(self.config['results_root'])
        self.repo_root = Path(self.config['repo_root'])
        self.logs = self.batch_root / 'logs'
        self.recovery = self.batch_root / 'recovery' / 'partial'
        self.state_path = self.batch_root / 'state.json'
        self.lock = threading.Lock()
        self.active = {}
        previous_state = {}
        if self.state_path.exists():
            try:
                previous_state = json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                previous_state = {}
        self.attempts = previous_state.get('attempts', {})
        self.logs.mkdir(parents=True, exist_ok=True)

    def leaf(self, index, phase):
        cell, section, range_index, epoch = PAIRS[index]
        folder = f"{section}_range{range_index}_{phase}_invivo_{self.config['channel_suffix']}"
        return self.results_root / cell / folder / '1' / str(epoch)

    def expected_root(self, index):
        cell, _, _, epoch = PAIRS[index]
        if cell != 'cell3':
            return None
        roots = self.config['cell3_apical_roots']
        return roots[(epoch - 1) % len(roots)]

    def valid(self, index, phase):
        path = self.leaf(index, phase)
        try:
            if not all((path / name).is_file() and (path / name).stat().st_size for name in REQUIRED):
                return False
            if np.load(path / 'soma_v_array.npy', mmap_mode='r').shape != (40001, 1, 37, 1):
                return False
            params = json.loads((path / 'simulation_params.json').read_text())
            cell = PAIRS[index][0]
            counts = self.config['synapse_counts'][cell]
            if params.get('with_ap') is not False or params.get('effective_iter_step') != 2:
                return False
            for key, expected in counts.items():
                if params.get(key) != expected:
                    return False
            expected_root = self.expected_root(index)
            if expected_root is not None:
                policy_path = path / 'cell3_apical_policy.json'
                if not policy_path.is_file():
                    return False
                policy = json.loads(policy_path.read_text())
                if policy.get('selected_root') != expected_root:
                    return False
                if policy.get('eligible_exc_count', 0) < 3:
                    return False
                if sum(policy.get('exc_counts_by_range', [])) != policy['eligible_exc_count']:
                    return False
            return True
        except Exception:
            return False

    def save_state(self, **updates):
        with self.lock:
            state = {}
            if self.state_path.exists():
                try:
                    state = json.loads(self.state_path.read_text())
                except json.JSONDecodeError:
                    state = {}
            state.update(updates)
            state['updated_at'] = now()
            state['active'] = dict(self.active)
            state['attempts'] = dict(self.attempts)
            temporary = self.state_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(self.state_path)

    def archive_partial(self, index, phase, attempt):
        leaf = self.leaf(index, phase)
        if not leaf.exists():
            return
        destination = self.recovery / f'{phase}-{index}-attempt{attempt}-{time.time_ns()}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(leaf, destination)

    def command(self, index, phase):
        cell, section, range_index, epoch = PAIRS[index]
        counts = self.config['synapse_counts'][cell]
        return [
            self.config['python'], '-u', 'L5b_simulation.py',
            '--morphology', f'./model/{cell}.asc',
            '--results_root', str(self.results_root / cell),
            '--sec_type', section,
            '--dis_to_root', str(range_index),
            '--start_epoch', str(epoch),
            '--num_epochs', '1',
            '--num_syn_basal_exc', str(counts['NUM_SYN_BASAL_EXC']),
            '--num_syn_apic_exc', str(counts['NUM_SYN_APIC_EXC']),
            '--num_syn_basal_inh', str(counts['NUM_SYN_BASAL_INH']),
            '--num_syn_apic_inh', str(counts['NUM_SYN_APIC_INH']),
            '--num_syn_soma_inh', str(counts['NUM_SYN_SOMA_INH']),
            *self.config['cli'],
            '--spat_cond', phase,
        ]

    def run_unit(self, index, phase):
        attempt_key = f'{phase}-{index}'
        with self.lock:
            attempt = self.attempts.get(attempt_key, 0) + 1
            self.attempts[attempt_key] = attempt
            self.active[attempt_key] = {'attempt': attempt, 'started_at': now(), 'pair': PAIRS[index]}
        self.save_state(phase=phase, status=f'running_{phase}')
        self.archive_partial(index, phase, attempt)
        log_path = self.logs / f'{phase}-{index}-attempt{attempt}.log'
        env = os.environ.copy()
        env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        return_code = None
        timed_out = False
        with log_path.open('w') as log:
            log.write(f'[START] {now()} pair={PAIRS[index]} attempt={attempt}\n')
            log.write('[RUN] ' + ' '.join(self.command(index, phase)) + '\n')
            log.flush()
            process = subprocess.Popen(
                self.command(index, phase), cwd=self.repo_root, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=int(self.config['unit_timeout_seconds']))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.write(f'[END] {now()} return_code={return_code} timed_out={timed_out}\n')
        ok = return_code == 0 and not timed_out and self.valid(index, phase)
        with self.lock:
            self.active.pop(attempt_key, None)
        self.save_state(phase=phase, status=f'running_{phase}')
        return index, ok, attempt, timed_out, return_code

    def audit(self, phase):
        missing = [index for index in range(len(PAIRS)) if not self.valid(index, phase)]
        report = {
            'checked_at': now(), 'phase': phase, 'expected': len(PAIRS),
            'complete': len(PAIRS) - len(missing), 'missing': missing,
        }
        path = self.batch_root / f'audit-{phase}.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)
        return report

    def run_phase(self, phase):
        max_attempts = int(self.config['max_attempts_per_unit'])
        while True:
            report = self.audit(phase)
            if not report['missing']:
                self.save_state(phase=phase, status=f'{phase}_complete', **{f'{phase}_complete': len(PAIRS)})
                return
            pending = [
                index for index in report['missing']
                if self.attempts.get(f'{phase}-{index}', 0) < max_attempts
            ]
            if not pending:
                blocked = {
                    'blocked_at': now(), 'phase': phase, 'missing': report['missing'],
                    'attempts': self.attempts,
                }
                (self.batch_root / 'BLOCKED.json').write_text(json.dumps(blocked, indent=2) + '\n')
                self.save_state(phase=phase, status='blocked')
                raise RuntimeError(f'{phase} has {len(report["missing"])} invalid units after retries')

            max_workers = int(self.config['max_concurrency'])
            stagger = float(self.config['launch_stagger_seconds'])
            iterator = iter(pending)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = set()
                for _ in range(min(max_workers, len(pending))):
                    futures.add(executor.submit(self.run_unit, next(iterator), phase))
                    time.sleep(stagger)
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, ok, attempt, timed_out, return_code = future.result()
                        print(
                            f'[UNIT] phase={phase} index={index} pair={PAIRS[index]} '
                            f'attempt={attempt} ok={ok} timed_out={timed_out} rc={return_code}',
                            flush=True,
                        )
                        try:
                            next_index = next(iterator)
                        except StopIteration:
                            continue
                        time.sleep(stagger)
                        futures.add(executor.submit(self.run_unit, next_index, phase))

    def run(self):
        lock_path = self.batch_root / 'controller.lock'
        with lock_path.open('w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(str(os.getpid()) + '\n')
            lock_file.flush()
            self.save_state(status='starting', started_at=now(), pid=os.getpid())
            self.run_phase('clus')
            clus_report = self.audit('clus')
            if clus_report['complete'] != len(PAIRS):
                raise RuntimeError('global clus barrier failed')
            self.run_phase('distr')
            distr_report = self.audit('distr')
            if distr_report['complete'] != len(PAIRS):
                raise RuntimeError('distr final audit failed')
            complete = {'completed_at': now(), 'clus': clus_report, 'distr': distr_report}
            (self.batch_root / 'COMPLETE').write_text(json.dumps(complete, indent=2) + '\n')
            self.save_state(status='complete', completed_at=complete['completed_at'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'status', 'audit'))
    parser.add_argument('config')
    args = parser.parse_args()
    batch = Batch(args.config)
    if args.mode == 'run':
        batch.run()
    elif args.mode == 'status':
        state = json.loads(batch.state_path.read_text()) if batch.state_path.exists() else {}
        print(json.dumps({'state': state, 'clus': batch.audit('clus'), 'distr': batch.audit('distr')}, indent=2))
    else:
        print(json.dumps({'clus': batch.audit('clus'), 'distr': batch.audit('distr')}, indent=2))


if __name__ == '__main__':
    main()
