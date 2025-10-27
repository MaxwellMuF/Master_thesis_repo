from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from agent.helper import configs
from agent.helper import json_io


# ---------------------------------------------------------------------------
# Logger: setup logger for agent
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    rotate: bool = False,
    max_bytes: int = 2_000_000,
    backup_count: int = 2,
) -> logging.Logger:
    """
    Console logger
    """
    logger = logging.getLogger("Agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = "%(asctime)sZ | %(levelname)s | %(name)s: %(message)s"

    class _UTCFormatter(logging.Formatter):
        converter = time.gmtime

    formatter = _UTCFormatter(fmt)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    if log_file is not None:
        from logging.handlers import RotatingFileHandler
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = (RotatingFileHandler(str(log_file), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
              if rotate else logging.FileHandler(str(log_file), encoding="utf-8"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _secs_to_hms(s: float) -> str:
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"


# ---------------------------------------------------------------------------
# Logging output: write log file and txt file for evaluation report
# ---------------------------------------------------------------------------

def pipeline_data_log_and_report(
    p_data_config: configs.PipelineDataConfig,
    p_task_config:configs.PipelineTaskConfig,
    data: pd.DataFrame,
    path_folder_output: Path,
    num_prompts: int,
    # timestamps
    time_start: float,
    time_prompts: float,
    time_predictions: float,
    time_evaluation: float,
    iteration:int,
    log: logging.Logger,
    log_level: str = "INFO",
    metrics_types: Iterable[str] = ("LONG_CODES", "SHORT_CODES"),
    text_column: str = "TEXT",
) -> None:
    """
    Console logs + write run_meta.json and evaluation_report.txt.
    """

    file_stem = Path(p_data_config.name_data).stem
    out_dir = path_folder_output / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- timings
    dur_prompts = time_prompts - time_start
    dur_predict = time_predictions - time_prompts
    dur_eval = time_evaluation - time_predictions
    dur_total = time.time() - time_start

    # save time of prediction
    path_file_times_pred_dict = out_dir / "time_prompts_pred.json"
    try:
        times_pred_dict = json_io.load_json(path=path_file_times_pred_dict)
    except:
        times_pred_dict = {}
    times_pred_dict = {**times_pred_dict, 
                       str(iteration) : _secs_to_hms(dur_predict)
    }
    json_io.save_json(times_pred_dict, path=path_file_times_pred_dict)
    
    # meta JSON
    meta = {
        "run": {
            "date": str(datetime.now().date()),
            "started_at": _now_iso_utc(),
            "finished_at": _now_iso_utc(),
            "status": "ok",
        },
        "data": {
            "name": p_data_config.name_data,
            "rows": int(len(data)),
            "shape": list(data.shape),
        },
        "model": {"name": p_data_config.name_model},
        "prompts": {"set": p_data_config.name_prompts_data, "num_prompts": num_prompts},
        "p_data_config": asdict(p_data_config),
        "p_task_config": asdict(p_task_config),
        "timings_seconds": {
            "make_prompts": round(dur_prompts, 3),
            "predictions": round(dur_predict, 3),
            "evaluation": round(dur_eval, 3),
            "total": round(dur_total, 3),
        },
        "timings_hms": {
            "make_prompts": _secs_to_hms(dur_prompts),
            "predictions": _secs_to_hms(dur_predict),
            "evaluation": _secs_to_hms(dur_eval),
            "total": _secs_to_hms(dur_total),
        },
        "artifacts": {},
    }
    meta_path = out_dir / f"{file_stem}_run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


    # ---- evaluation text (compact)
    eval_path = path_folder_output / f"{file_stem}_report.txt"
    with eval_path.open("w", encoding="utf-8") as f:
        f.write("## General ##\n")
        f.write(f"Date: {meta['run']['date']}\n")
        f.write(f"Used GPU: {p_data_config.type_gpu}\n\n")

        f.write("## Data and Model ##\n")
        f.write(f"Dataset: {p_data_config.name_data}, Shape: {tuple(meta['data']['shape'])}\n")
        f.write(f"Prompt set: {p_data_config.name_prompts_data}, #prompts: {num_prompts}\n")
        f.write(f"Model: {p_data_config.name_model}\n\n")

        f.write("## p_data_config ##\n")
        f.write(f"Temperature: {p_data_config.temperature}\n")
        f.write(f"Max tokens: {p_data_config.max_tokens}\n")
        f.write(f"Role with patient note: {p_data_config.role_to_add_note}\n")
        f.write(f"Enable thinking: {p_data_config.enable_thinking}\n")
        f.write(f"Continue final message: {p_data_config.continue_final_message}\n")
        f.write(f"Add generation prompt: {p_data_config.add_generation_prompt}\n")
        f.write(f"Improve prompt: {p_task_config.improve_prompt}\n\n")

        f.write("## Time last prompt##\n")
        f.write(f"Make prompts: {_secs_to_hms(dur_prompts)}\n")
        f.write(f"Predictions: {_secs_to_hms(dur_predict)}\n")
        f.write(f"Evaluation: {_secs_to_hms(dur_eval)}\n")
        f.write(f"Script total: {_secs_to_hms(dur_total)}\n\n")

        f.write("## Evaluation ##\n")
        for mt in metrics_types:
            # ground-truth column
            gt_col = mt if mt in data.columns else ("LONG_CODES" if "LONG_CODES" in data.columns else None)
            mean_gt = f"{data[gt_col].apply(len).mean():.1f}" if gt_col else "NA"

            f.write(f"Evaluation dataset: {p_data_config.name_data}, length: {len(data)}, "
                    f"metric: {mt}, mean codes GT: {mean_gt}\n")
            f.write(f"|{'Prompt':^10}|{'Precision':^11}|{'Recall':^10}|{'F1':^10}|{'F1 = 0':^10}|"
                    f"{'Prompt len':^12}|{'Output len':^12}|{'Duration':^12}|{'Codes Pred':^12}|\n")
            f.write("-" * 109 + "\n")

            for i in range(iteration+1):
                # check if column exist
                def _mean(col: str, digits: int = 4) -> str:
                    return f"{data[col].mean():.{digits}f}" if col in data.columns else "NA"

                def _count_zero(col: str) -> str:
                    return f"{int((data[col] == 0.0).sum())}" if col in data.columns else "NA"

                def _mean_len(col: str) -> str:
                    return f"{data[col].astype(str).str.len().mean():.0f}" if col in data.columns else "NA"

                precision = _mean(f"precision_{i}_{mt}")
                recall = _mean(f"recall_{i}_{mt}")
                f1 = _mean(f"f1_{i}_{mt}")
                f1_0 = _count_zero(f"f1_{i}_{mt}")

                # prompt length
                prompt_len = "NA"
                if f"prompt_{i}" in data.columns and text_column in data.columns:
                    prompt_len = f"{(data[f'prompt_{i}'].astype(str).str.len().mean() - data[text_column].astype(str).str.len().mean()):.0f}"

                output_len = _mean_len(f"prediction_{i}")
                codes_pred = f"{data[f'predicted_code_{i}'].apply(len).mean():.0f}" if f"predicted_code_{i}" in data.columns else "NA"

                f.write(f"|{i:^10}|{precision:^11}|{recall:^10}|{f1:^10}|{f1_0:^10}|"
                        f"{prompt_len:^12}|{output_len:^12}|{times_pred_dict[str(i)]:^12}|{codes_pred:^12}|\n")
            f.write("\n")


