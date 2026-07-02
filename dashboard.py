"""
Phase 6 — Streamlit Dashboard.

Run from the project root:
    streamlit run dashboard.py

Optional flag to point at a non-default database:
    streamlit run dashboard.py -- --db /path/to/results.db
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so storage/ and models/ can be imported
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from storage.db import ResultsStore

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Evaluation Benchmark",
    page_icon="📊",
    layout="wide",
)

st.title("📊 LLM Evaluation Benchmark Dashboard")

# ---------------------------------------------------------------------------
# Sidebar — database path + global filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    db_path = st.text_input(
        "Database path",
        value="results.db",
        help="Path to the SQLite results database (relative to project root or absolute).",
    )

    try:
        store = ResultsStore(db_path)
        runs = store.list_runs()
        agg_data = store.agg_by_provider_and_type()
        st.success(f"Connected · {len(runs)} run(s)")
    except Exception as exc:
        st.error(f"Cannot open database: {exc}")
        st.stop()

    st.divider()

    # Derive unique filter values from the pre-aggregated view (avoids a full table scan)
    all_providers = sorted({r["provider"] for r in agg_data})
    all_types = sorted({r["task_type"] for r in agg_data})
    all_diffs = ["easy", "medium", "hard"]

    if all_providers:
        st.subheader("Filters")
        sel_providers: list[str] = st.multiselect(
            "Providers", all_providers, default=all_providers
        )
        sel_types: list[str] = st.multiselect(
            "Task types", all_types, default=all_types
        )
        sel_diffs: list[str] = st.multiselect(
            "Difficulty", all_diffs, default=all_diffs
        )
    else:
        sel_providers = sel_types = sel_diffs = []

# ---------------------------------------------------------------------------
# Global empty-state guard
# ---------------------------------------------------------------------------

if not runs:
    st.info(
        "No benchmark runs found in the database.\n\n"
        "Run a benchmark using the runner module and save results via "
        "`ResultsStore`, then refresh this page."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_history, tab_scores, tab_heatmap, tab_cost, tab_viewer = st.tabs([
    "📋 Run History",
    "📈 Score Breakdown",
    "🔥 Heatmap",
    "💰 Cost vs Quality",
    "🔍 Response Viewer",
])

# ---------------------------------------------------------------------------
# Tab 1 — Run History
# ---------------------------------------------------------------------------

with tab_history:
    st.header("Benchmark Runs")

    runs_df = pd.DataFrame(runs).rename(columns={
        "run_id": "Run ID",
        "created_at": "Created At",
        "provider": "Provider",
        "model": "Model",
        "task_count": "Tasks",
        "notes": "Notes",
    })
    st.dataframe(runs_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Inspect a Run")

    def _run_label(rid: str) -> str:
        for r in runs:
            if r["run_id"] == rid:
                return f"{rid[:8]}… — {r['provider']} / {r['model']} ({r['task_count']} tasks)"
        return rid

    selected_run_id = st.selectbox(
        "Select run",
        [r["run_id"] for r in runs],
        format_func=_run_label,
        key="history_run_select",
    )

    if selected_run_id:
        summary = store.get_run_summary(selected_run_id)
        if summary:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Tasks", summary.get("task_count") or "—")
            c2.metric("Tokens", f"{(summary.get('total_tokens') or 0):,}")
            cost = summary.get("total_cost_usd") or 0
            c3.metric("Cost", f"${cost:.4f}")
            latency = summary.get("avg_latency_ms") or 0
            c4.metric("Avg Latency", f"{latency:.0f} ms")
            judge = summary.get("avg_llm_judge")
            c5.metric("Avg LLM Judge", f"{judge:.3f}" if judge else "—")

            started = summary.get("started_at", "")
            finished = summary.get("finished_at", "")
            if started and finished:
                st.caption(f"Started: {started}  ·  Finished: {finished}")
        else:
            st.warning("No result rows found for this run.")

# ---------------------------------------------------------------------------
# Tab 2 — Score Breakdown
# ---------------------------------------------------------------------------

with tab_scores:
    st.header("Score Breakdown by Provider & Task Type")

    if not agg_data:
        st.info("No scored results in the database yet.")
    else:
        df = pd.DataFrame(agg_data)

        if sel_providers:
            df = df[df["provider"].isin(sel_providers)]
        if sel_types:
            df = df[df["task_type"].isin(sel_types)]

        if df.empty:
            st.warning("No data matches the current sidebar filters.")
        else:
            METRIC_LABELS: dict[str, str] = {
                "avg_rouge_l": "ROUGE-L",
                "avg_bert_score": "BERTScore",
                "avg_exact_match": "Exact Match",
                "avg_llm_judge": "LLM Judge Score",
            }
            # Only show metrics that have at least one non-null value
            available_metrics = [
                m for m in METRIC_LABELS if df[m].notna().any()
            ]

            if not available_metrics:
                st.info("No score columns are populated yet (scores are null). Score results first.")
            else:
                metric = st.selectbox(
                    "Metric",
                    available_metrics,
                    format_func=lambda m: METRIC_LABELS[m],
                    key="scores_metric",
                )

                df["Provider / Model"] = df["provider"] + " / " + df["model"]

                fig = px.bar(
                    df,
                    x="task_type",
                    y=metric,
                    color="Provider / Model",
                    barmode="group",
                    text_auto=".3f",
                    labels={
                        "task_type": "Task Type",
                        metric: METRIC_LABELS[metric],
                    },
                    title=f"{METRIC_LABELS[metric]} by Task Type",
                )
                fig.update_layout(yaxis_range=[0, 1], yaxis_title=METRIC_LABELS[metric])
                st.plotly_chart(fig, use_container_width=True)

                display_cols = [
                    "provider", "model", "task_type", "n",
                    metric, "total_cost_usd", "avg_latency_ms",
                ]
                st.dataframe(
                    df[[c for c in display_cols if c in df.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

# ---------------------------------------------------------------------------
# Tab 3 — Heatmap (Provider × Difficulty)
# ---------------------------------------------------------------------------

with tab_heatmap:
    st.header("Performance Heatmap — Provider × Difficulty")
    st.caption(
        "Score = first available metric per row: ROUGE-L → BERTScore → Exact Match → LLM Judge."
    )

    heatmap_raw = store.agg_heatmap()

    if not heatmap_raw:
        st.info("No heatmap data available.")
    else:
        df_h = pd.DataFrame(heatmap_raw)

        if sel_providers:
            df_h = df_h[df_h["provider"].isin(sel_providers)]
        if sel_types:
            df_h = df_h[df_h["task_type"].isin(sel_types)]
        if sel_diffs:
            df_h = df_h[df_h["difficulty"].isin(sel_diffs)]

        if df_h.empty:
            st.warning("No data matches the current sidebar filters.")
        else:
            df_h["Provider / Model"] = df_h["provider"] + " / " + df_h["model"]

            for ttype in sorted(df_h["task_type"].unique()):
                st.subheader(f"Task Type: {ttype.capitalize()}")
                subset = df_h[df_h["task_type"] == ttype]
                pivot = subset.pivot_table(
                    index="Provider / Model",
                    columns="difficulty",
                    values="avg_score",
                )
                # Ensure consistent column order and fill missing difficulty columns
                for col in ["easy", "medium", "hard"]:
                    if col not in pivot.columns:
                        pivot[col] = float("nan")
                pivot = pivot[["easy", "medium", "hard"]]

                fig = px.imshow(
                    pivot,
                    text_auto=".3f",
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    labels={"color": "Avg Score"},
                    title=f"{ttype.capitalize()} — Provider × Difficulty",
                    zmin=0,
                    zmax=1,
                )
                fig.update_xaxes(title="Difficulty")
                fig.update_yaxes(title="Provider / Model")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 — Cost vs Quality scatter plot
# ---------------------------------------------------------------------------

with tab_cost:
    st.header("Cost vs Quality")
    st.caption("Each bubble is one benchmark run. Bubble size scales with task count.")

    cq_raw = store.agg_cost_vs_quality()

    if not cq_raw:
        st.info(
            "No cost data available.\n\n"
            "Ensure `estimated_cost_usd` is populated on `ScoredResult` objects before saving."
        )
    else:
        df_cq = pd.DataFrame(cq_raw)

        if sel_providers:
            df_cq = df_cq[df_cq["provider"].isin(sel_providers)]
        if sel_types:
            df_cq = df_cq[df_cq["task_type"].isin(sel_types)]

        if df_cq.empty:
            st.warning("No data matches the current sidebar filters.")
        else:
            df_cq["Provider / Model"] = df_cq["provider"] + " / " + df_cq["model"]
            df_cq["run_short"] = df_cq["run_id"].str[:8] + "…"

            fig = px.scatter(
                df_cq,
                x="total_cost_usd",
                y="avg_score",
                color="Provider / Model",
                size="n",
                hover_data={"run_short": True, "task_type": True, "n": True, "run_id": False},
                labels={
                    "total_cost_usd": "Total Cost (USD)",
                    "avg_score": "Average Score",
                    "run_short": "Run",
                },
                title="Cost vs Quality  (bubble size = task count)",
            )
            fig.update_layout(
                xaxis_title="Total Cost (USD)",
                yaxis_title="Average Score",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_cq.drop(columns=["run_short"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )

# ---------------------------------------------------------------------------
# Tab 5 — Response Viewer
# ---------------------------------------------------------------------------

with tab_viewer:
    st.header("Response Viewer")
    st.caption(
        "Select a task to compare raw model outputs side-by-side across providers."
    )

    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        vf_type = st.selectbox(
            "Task type", ["(all)"] + all_types, key="vf_type"
        )
    with vc2:
        vf_diff = st.selectbox(
            "Difficulty", ["(all)"] + all_diffs, key="vf_diff"
        )
    with vc3:
        vf_provider = st.selectbox(
            "Provider", ["(all)"] + all_providers, key="vf_provider"
        )

    viewer_rows = store.query(
        task_type=None if vf_type == "(all)" else vf_type,
        difficulty=None if vf_diff == "(all)" else vf_diff,
        provider=None if vf_provider == "(all)" else vf_provider,
        dry_run=False,
        limit=500,
    )

    if not viewer_rows:
        st.info("No results match these filters.")
    else:
        task_ids = sorted({r["task_id"] for r in viewer_rows})
        selected_task = st.selectbox("Task ID", task_ids, key="viewer_task")

        task_rows = [r for r in viewer_rows if r["task_id"] == selected_task]

        if task_rows:
            first = task_rows[0]
            st.markdown(
                f"**Type:** `{first['task_type']}` &nbsp;|&nbsp; "
                f"**Difficulty:** `{first['difficulty']}` &nbsp;|&nbsp; "
                f"**Domain:** `{first['domain']}`"
            )

            n_cols = min(len(task_rows), 4)  # cap at 4 columns for readability
            cols = st.columns(n_cols)

            SCORE_FIELDS = [
                ("rouge_l", "ROUGE-L"),
                ("bert_score", "BERTScore"),
                ("exact_match", "Exact Match"),
                ("token_f1", "Token F1"),
                ("llm_judge_score", "LLM Judge"),
            ]

            for col, row in zip(cols, task_rows):
                with col:
                    st.markdown(f"**{row['provider']} / {row['model']}**")
                    st.caption(
                        f"Latency: {row['latency_ms']:.0f} ms · "
                        f"Tokens: {row['total_tokens'] or 0:,}"
                    )

                    score_lines = [
                        f"**{label}:** {row[key]:.3f}"
                        for key, label in SCORE_FIELDS
                        if row.get(key) is not None
                    ]
                    if score_lines:
                        st.markdown("  \n".join(score_lines))
                    else:
                        st.caption("No scores yet")

                    st.text_area(
                        "Raw output",
                        value=row.get("raw_output") or "(empty)",
                        height=220,
                        key=f"raw_{row['id']}",
                        disabled=True,
                    )

                    parsed = row.get("parsed_output")
                    if parsed:
                        with st.expander("Parsed output"):
                            try:
                                st.json(json.loads(parsed))
                            except Exception:
                                st.text(parsed)

                    if row.get("parse_error"):
                        st.error(f"Parse error: {row['parse_error']}")
