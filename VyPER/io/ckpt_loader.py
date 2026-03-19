import os
import re
import warnings
import numpy as np
import pandas as pd

from pathlib import Path


def ckpt_loader(cfg, key: str='predicting'):
    model_directory = Path(cfg[key]['model_directory'])
    model_choice    = cfg[key]['model_choice'].split('-')

    if len(model_choice) != 2:
        warnings.warn(
            f"Invalid `model_choice`: {cfg[key]['model_choice']}, "
            "use the last checkpoint instead.", UserWarning)
        ckpt_file = model_directory / 'checkpoints' / 'last.ckpt'
    else:
        ckpt_dir = model_directory / 'checkpoints'
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory '{ckpt_dir}' does not exist.")

        # Regex to capture: epoch=5-<metric_name>=<value>
        pattern = re.compile(r'^epoch=(?P<epoch>\d+)-(?P<metric>[^=]+)=(?P<value>[^}]+)\.ckpt$')

        rows = []
        for fname in os.listdir(ckpt_dir):
            m = pattern.match(fname)
            if not m:
                continue

            epoch   = int(m.group('epoch'))
            metric  = m.group('metric')
            value   = float(m.group('value'))

            rows.append({'epoch': epoch, 'metric': metric, 'value': value})

        if not rows:
            raise FileNotFoundError(f"No valid checkpoints found in '{ckpt_dir}'")

        ckpt_db = pd.DataFrame(rows)

        # Find the best checkpoint according to user choice
        mode, suffix = model_choice

        # Find all metric names that end with the supplied suffix
        candidate_metrics = ckpt_db['metric'].unique()
        matching = [m for m in candidate_metrics if m.endswith(suffix)]

        if not matching:
            raise ValueError(
                f"No metric ending with '{suffix}' found in checkpoints.\n"
                f"Available metrics: {sorted(candidate_metrics)}"
            )

        if len(matching) > 1:
            raise ValueError(
                f"Ambiguous metric suffix '{suffix}'. Matches: {matching}\n"
                "Please specify the full metric name or choose a more specific suffix."
            )

        target_metric = matching[0]

        # Filter to that metric and pick best according to mode
        metric_df = ckpt_db[ckpt_db['metric'] == target_metric]
        idx_best  = getattr(np, 'arg' + mode)(metric_df['value'])
        best_row  = metric_df.iloc[idx_best]

        # Build the filename again (to be safe)
        best_fname = f"epoch={best_row['epoch']}-{target_metric}={best_row['value']:.3f}.ckpt"
        ckpt_file  = ckpt_dir / best_fname

        print(f"Loading checkpoint: {ckpt_file.name}")

        return ckpt_file

