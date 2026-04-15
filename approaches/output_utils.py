from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


STANDARD_FINAL_COLUMNS = [
    "audio_name",
    "customer_index",
    "start",
    "end",
    "duration",
    "start_hms",
    "end_hms",
    "method",
    "score",
    "source_conversation_ids",
]

STANDARD_TAGGED_COLUMNS = [
    "audio_name",
    "conversation_id",
    "start",
    "end",
    "duration",
    "num_regions",
    "start_hms",
    "end_hms",
    "assigned_customer_index",
    "is_selected_candidate",
    "source_final_start",
    "source_final_end",
    "assignment_reason",
]


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "start" in df.columns:
        df["start_hms"] = df["start"].astype(float).map(format_time)
    if "end" in df.columns:
        df["end_hms"] = df["end"].astype(float).map(format_time)
    return df


def load_ground_truth(project_root: Path) -> dict:
    path = project_root / "ground_truth.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def source_ids_from_row(row: pd.Series | dict) -> list[int]:
    value = row.get("source_conversation_ids")
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        value = row.get("conversation_id")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    ids = []
    for part in str(value).replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(float(part)))
        except ValueError:
            continue
    return ids


def choose_final_two_from_candidates(
    candidates: pd.DataFrame,
    audio_name: str,
    method: str,
    expected_count: int = 2,
) -> pd.DataFrame:
    """Fallback final selector for baselines that only produce candidates.

    Selects the longest candidate windows and returns them in chronological order.
    This is intentionally simple so baseline approaches remain honest.
    """
    if candidates.empty:
        return pd.DataFrame(columns=STANDARD_FINAL_COLUMNS)

    df = candidates.copy()
    df["duration"] = df["duration"].astype(float)
    selected = df.sort_values(["duration", "start"], ascending=[False, True]).head(expected_count)
    selected = selected.sort_values("start").reset_index(drop=True)

    rows = []
    for idx, row in selected.iterrows():
        source_id = int(row["conversation_id"]) if "conversation_id" in row else idx + 1
        rows.append(
            {
                "audio_name": audio_name,
                "customer_index": idx + 1,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["end"]) - float(row["start"]),
                "method": method,
                "score": float(row.get("duration", 0.0)),
                "source_conversation_ids": str(source_id),
            }
        )

    return standardize_final_two(pd.DataFrame(rows), audio_name, method)


def standardize_final_two(final_rows: pd.DataFrame | list[dict], audio_name: str, method: str) -> pd.DataFrame:
    df = pd.DataFrame(final_rows).copy()
    if df.empty:
        return pd.DataFrame(columns=STANDARD_FINAL_COLUMNS)

    if "audio_name" not in df.columns:
        df["audio_name"] = audio_name
    if "customer_index" not in df.columns:
        df = df.sort_values("start").reset_index(drop=True)
        df["customer_index"] = range(1, len(df) + 1)
    if "method" not in df.columns:
        df["method"] = method
    if "score" not in df.columns:
        score_col = "conversation_score" if "conversation_score" in df.columns else None
        df["score"] = df[score_col] if score_col else pd.NA
    if "source_conversation_ids" not in df.columns:
        if "conversation_id" in df.columns:
            df["source_conversation_ids"] = df["conversation_id"].astype(str)
        else:
            df["source_conversation_ids"] = pd.NA

    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["duration"] = df["end"] - df["start"]
    df = add_time_columns(df)

    for col in STANDARD_FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[STANDARD_FINAL_COLUMNS].sort_values("customer_index").reset_index(drop=True)


def build_tagged_conversation_candidates(
    candidates: pd.DataFrame | list[dict],
    final_two: pd.DataFrame | list[dict],
    audio_name: str,
) -> pd.DataFrame:
    candidates_df = pd.DataFrame(candidates).copy()
    final_df = pd.DataFrame(final_two).copy()
    if candidates_df.empty:
        return pd.DataFrame(columns=STANDARD_TAGGED_COLUMNS)

    candidates_df = add_time_columns(candidates_df)
    final_df = standardize_final_two(final_df, audio_name, method="final")

    assignments = {}
    for _, row in final_df.iterrows():
        for source_id in source_ids_from_row(row):
            assignments[source_id] = row

    rows = []
    for _, row in candidates_df.iterrows():
        conversation_id = int(row["conversation_id"])
        assigned = assignments.get(conversation_id)
        assignment_reason = None
        if assigned is None:
            candidate_start = float(row["start"])
            candidate_end = float(row["end"])
            best_overlap = 0.0
            best_final = None
            for _, final_row in final_df.iterrows():
                overlap = max(
                    0.0,
                    min(candidate_end, float(final_row["end"])) - max(candidate_start, float(final_row["start"])),
                )
                coverage = overlap / max(candidate_end - candidate_start, 1e-6)
                if coverage > best_overlap:
                    best_overlap = coverage
                    best_final = final_row
            if best_final is not None and best_overlap >= 0.5:
                assigned = best_final
                assignment_reason = f"overlaps_customer_{int(assigned['customer_index'])}"
        out = row.to_dict()
        out["audio_name"] = audio_name
        if assigned is None:
            out["assigned_customer_index"] = pd.NA
            out["is_selected_candidate"] = False
            out["source_final_start"] = pd.NA
            out["source_final_end"] = pd.NA
            out["assignment_reason"] = "not_selected"
        else:
            out["assigned_customer_index"] = int(assigned["customer_index"])
            out["is_selected_candidate"] = True
            out["source_final_start"] = float(assigned["start"])
            out["source_final_end"] = float(assigned["end"])
            source_ids = source_ids_from_row(assigned)
            if assignment_reason:
                out["assignment_reason"] = assignment_reason
            else:
                out["assignment_reason"] = (
                    f"merged_into_customer_{int(assigned['customer_index'])}"
                    if len(source_ids) > 1
                    else f"selected_as_customer_{int(assigned['customer_index'])}"
                )
        rows.append(out)

    tagged = pd.DataFrame(rows)
    for col in STANDARD_TAGGED_COLUMNS:
        if col not in tagged.columns:
            tagged[col] = pd.NA
    return tagged[STANDARD_TAGGED_COLUMNS].sort_values("conversation_id").reset_index(drop=True)


def interval_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def evaluate_final_two(final_two: pd.DataFrame, ground_truth: dict, approach_name: str) -> pd.DataFrame:
    rows = []
    for _, row in final_two.iterrows():
        audio_name = row["audio_name"]
        idx = int(row["customer_index"])
        gt = ground_truth.get(audio_name, {}).get(f"Conversation {idx}")
        if not gt:
            continue
        pred_start = float(row["start"])
        pred_end = float(row["end"])
        rows.append(
            {
                "approach": approach_name,
                "audio_name": audio_name,
                "customer_index": idx,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "gt_start": float(gt["start"]),
                "gt_end": float(gt["end"]),
                "start_error_s": pred_start - float(gt["start"]),
                "end_error_s": pred_end - float(gt["end"]),
                "abs_start_error_s": abs(pred_start - float(gt["start"])),
                "abs_end_error_s": abs(pred_end - float(gt["end"])),
                "iou": interval_iou(pred_start, pred_end, float(gt["start"]), float(gt["end"])),
            }
        )
    return pd.DataFrame(rows)


def summarize_evaluation(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    return (
        eval_df.groupby(["approach", "audio_name"], as_index=False)
        .agg(
            mean_abs_start_error_s=("abs_start_error_s", "mean"),
            mean_abs_end_error_s=("abs_end_error_s", "mean"),
            mean_iou=("iou", "mean"),
        )
        .sort_values(["approach", "audio_name"])
    )


def export_uniform_outputs(
    output_dir: Path,
    audio_name: str,
    approach_name: str,
    conversation_candidates: pd.DataFrame | list[dict],
    final_two: pd.DataFrame | list[dict] | None = None,
    ground_truth: dict | None = None,
    final_selection_method: str | None = None,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_name).stem
    candidates_df = add_time_columns(pd.DataFrame(conversation_candidates).copy())
    method = final_selection_method or approach_name
    if final_two is None or pd.DataFrame(final_two).empty:
        final_df = choose_final_two_from_candidates(candidates_df, audio_name, method=method)
    else:
        final_df = standardize_final_two(final_two, audio_name, method=method)
    tagged_df = build_tagged_conversation_candidates(candidates_df, final_df, audio_name)

    candidates_df.to_csv(output_dir / f"{stem}_conversation_candidates.csv", index=False)
    final_df.to_csv(output_dir / f"{stem}_final_two_conversations.csv", index=False)
    tagged_df.to_csv(output_dir / f"{stem}_tagged_conversation_candidates.csv", index=False)

    eval_df = pd.DataFrame()
    if ground_truth:
        eval_df = evaluate_final_two(final_df, ground_truth, approach_name)

    return {
        "conversation_candidates": candidates_df,
        "final_two_conversations": final_df,
        "tagged_conversation_candidates": tagged_df,
        "evaluation": eval_df,
    }


def export_combined_outputs(output_dir: Path, exported: Iterable[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = list(exported)
    combined_final = pd.concat(
        [item["final_two_conversations"] for item in exported if not item["final_two_conversations"].empty],
        ignore_index=True,
    ) if exported else pd.DataFrame(columns=STANDARD_FINAL_COLUMNS)
    combined_eval = pd.concat(
        [item["evaluation"] for item in exported if not item["evaluation"].empty],
        ignore_index=True,
    ) if exported else pd.DataFrame()
    eval_summary = summarize_evaluation(combined_eval)

    combined_final.to_csv(output_dir / "all_files_final_two_conversations.csv", index=False)
    if not combined_eval.empty:
        combined_eval.to_csv(output_dir / "evaluation_against_ground_truth.csv", index=False)
    if not eval_summary.empty:
        eval_summary.to_csv(output_dir / "evaluation_summary.csv", index=False)

    return {
        "all_files_final_two_conversations": combined_final,
        "evaluation_against_ground_truth": combined_eval,
        "evaluation_summary": eval_summary,
    }
