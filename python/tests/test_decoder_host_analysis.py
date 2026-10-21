"""Saved-record checks for decoder host attribution."""

import sqlite3

import pytest
from scripts import decoder_host_analysis as analysis


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_trace_attributes_kernels_by_launch_even_when_gpu_finishes_outside_host_range(tmp_path):
    """Graph work belongs to its CUDA launch correlation, including asynchronous tails."""
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER);
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER);
        INSERT INTO NVTX_EVENTS VALUES(0,500,'issue40/controlled/bucket-8-pair-00',1);
        INSERT INTO StringIds VALUES(1,'cudaGraphLaunch_v10000');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(100,120,1,42,1);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(130,300,42),(250,700,42),(140,400,99);
        """)
    rows = analysis.trace_graphs(path)
    assert rows == [
        {
            "sample_id": "bucket-8-pair-00",
            "workload": "controlled",
            "graph_count": 1,
            "kernel_count": 2,
            "graph_span_median_ns": 570,
            "graph_busy_median_ns": 570,
            "launch_median_ns": 20,
        }
    ]
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=42")
    with pytest.raises(ValueError, match="kernels"):
        analysis.trace_graphs(path)


@pytest.mark.req("SP-LFFASTESTWHISPER-003")
def test_record_comparison_rejects_changed_or_missing_tokens_across_loads():
    """Each load, repeat, wall control and trace must retain the same complete token sequences."""
    one = {"run_id": "L0", "tokens": {"a": [1, 2], "b": [3]}}
    two = {"run_id": "L1", "tokens": {"a": [1, 2], "b": [3]}}
    assert analysis.compare_tokens([one, two]) == {"equal": True, "samples": 2, "runs": 2}
    two["tokens"]["a"] = [1, 4]
    with pytest.raises(ValueError, match="tokens"):
        analysis.compare_tokens([one, two])
    del two["tokens"]["a"]
    with pytest.raises(ValueError, match="samples"):
        analysis.compare_tokens([one, two])


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_origin_sensitivity_excludes_entire_groups_and_propagates_wait_bounds():
    """Clock sensitivity retains all primary samples and removes only predeclared wide anchors."""
    from scripts.decoder_host_profile import SECTIONS

    run = {
        "run_id": "01-L1",
        "condition": "L1",
        "anchors": [
            {"sample_id": "bucket-8", "lower_ns": 0, "upper_ns": 50_000},
            {"sample_id": "speech", "lower_ns": 0, "upper_ns": 150_000},
        ],
    }
    rows = []
    for name, workload in [("bucket-8-pair-00", "controlled"), ("speech", "utterance")]:
        row = {f"{section}_ns": 100 for section in SECTIONS}
        rows.append(
            dict(
                row,
                run_id="01-L1",
                condition="L1",
                sample_id=name,
                workload=workload,
                bucket=8,
                total_ns=700,
                gpu_pending_ns=60,
                after_graph_ns=40,
                gpu_pending_lower_ns=50,
                gpu_pending_upper_ns=70,
                graph_gpu_ns=300,
            )
        )
    result = analysis.clock_sensitivity([run], rows, max_width_ns=100_000)
    assert result["threshold_ns"] == 100_000
    assert result["excluded_steps"] == 1
    assert result["widths"]["L1"]["median_ns"] == 100_000
    assert result["widths"]["L1"]["p95_ns"] == 145_000
    assert result["widths"]["L1"]["max_ns"] == 150_000
    assert len(result["filtered_groups"]) == 1
    assert result["filtered_groups"][0]["workload"] == "controlled"
    group = result["per_origin"][0]
    assert group["after_graph_lower"]["median_ns"] == 30
    assert group["after_graph_upper"]["median_ns"] == 50
    assert len(rows) == 2


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_trace_tail_separates_dma_gaps_and_host_return_using_correlations(tmp_path):
    """Actual copy activity excludes host enqueue delays and preserves the complete tail."""
    path = tmp_path / "tail.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER);
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start INTEGER,end INTEGER,correlationId INTEGER,copyKind INTEGER,bytes INTEGER);
        INSERT INTO NVTX_EVENTS VALUES(0,1000,'issue40/controlled/bucket-8-pair-00',1);
        INSERT INTO StringIds VALUES(1,'cudaGraphLaunch_v10000'),(2,'cudaMemcpyAsync_v3020'),
            (3,'cudaStreamSynchronize_v3020');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (10,20,1,42,1),(30,400,1,43,2),(410,500,1,44,2),(510,600,1,45,3);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(40,200,42),(60,999,99);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
            (230,250,43,2,4),(450,470,44,2,8),(240,300,99,2,100);
        """)
    rows = analysis.trace_tails(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["tail_ns"] == 400
    assert row["before_copy_ns"] == 30
    assert row["copy_active_ns"] == 40
    assert row["between_copies_ns"] == 200
    assert row["host_return_ns"] == 130
    assert row["copy_bytes"] == 12
    assert row["tail_ns"] == sum(
        row[name]
        for name in ("before_copy_ns", "copy_active_ns", "between_copies_ns", "host_return_ns")
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE correlationId IN (43,44)")
    with pytest.raises(ValueError, match="copy"):
        analysis.trace_tails(path)
