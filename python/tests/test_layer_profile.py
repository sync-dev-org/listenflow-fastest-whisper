"""CPU checks for fused TensorRT layer timing reports."""

import pytest
from scripts import profile_layers


@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_layers_keep_fused_names_types_and_rank_mean_gpu_time():
    """The report ranks fused layers without charging one fusion to multiple categories."""
    samples = [
        ("PWN((Unnamed Layer* 8) [ElementWise], (Unnamed Layer* 9) [Unary])", 1.0),
        ("(Unnamed Layer* 4) [Matrix Multiply]", 4.0),
        ("(Unnamed Layer* 4) [Matrix Multiply]", 6.0),
        ("PWN((Unnamed Layer* 8) [ElementWise], (Unnamed Layer* 9) [Unary])", 3.0),
        ("Reformatting CopyNode for Input Tensor 0", 2.0),
        ("Reformatting CopyNode for Input Tensor 0", 2.0),
    ]
    report = profile_layers.summarize_layers(samples, iterations=2)
    assert [layer["kind"] for layer in report["layers"]] == ["matmul", "pointwise", "reformat"]
    assert report["total_mean_ms"] == 9
    assert report["layers"][0]["mean_ms"] == 5
    assert report["layers"][0]["share"] == pytest.approx(5 / 9)
    assert report["layers"][1]["operator_types"] == ["ElementWise", "Unary"]
    assert report["layers"][1]["name"] == samples[0][0]
    assert sum(report["kinds_ms"].values()) == 9


@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_layers_reject_incomplete_or_invalid_profiler_results():
    """Missing callbacks and negative GPU times cannot become a complete layer report."""
    for samples in ([], [("matmul", -1.0)], [("matmul", float("nan"))], [("matmul", 1.0)]):
        with pytest.raises(ValueError):
            profile_layers.summarize_layers(samples, iterations=2)


@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_bucket_selection_reads_the_product_boundaries():
    """Diagnostic bucket choices come directly from the decoder's padding implementation."""
    source = "def _pad_cache(self, inputs):\n    boundaries = (1, 7, 19, 41)\n"
    assert profile_layers.decoder_buckets(source=source) == (1, 7, 19, 41)
    assert profile_layers.decoder_buckets()[0] == 1
    assert profile_layers.decoder_buckets()[-1] == 448
    assert len(profile_layers.decoder_buckets()) == 16


@pytest.mark.req("SP-LFFASTESTWHISPER-011")
@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("__myl_Fc_myl0_15", "matmul"),
        ("_gemv_mha_v1_myl0_8", "attention"),
        ("__myl_AddCastMeanSubMulMeanAddSqrtDivMulMulAddCast_myl0_10", "normalization"),
        ("__myl_SlicReshConcReshConcTranReshMove_myl0_501", "concat"),
        ("__myl_TranMulReshMove_myl0_8", "shuffle"),
        ("__myl_SubExpDivMulLogSumMaxr_myl0_525", "softmax"),
    ],
)
def test_internal_fused_symbols_expose_their_operation_family(name, kind):
    """TensorRT's internal fused symbols retain an explicit inferred operation family."""
    assert profile_layers.layer_kind(name)[0] == kind


@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_enqueue_projection_uses_gpu_time_and_preserves_fused_layer_names(tmp_path):
    """Layer observations use GPU projections and omit the enclosing TensorRT range."""
    trace = tmp_path / "enqueue.csv"
    trace.write_text(
        '"Name","Projected Start (ns)","Projected Duration (ns)","Orig Duration (ns)",'
        '"PID","NumGPUOps","NumChild","RangeId"\n'
        '"TensorRT:ExecutionContext::enqueueV3",100,9000,80000,41,3,2,1\n'
        '"TensorRT:PWN(conv, relu)",200,3500,70000,41,2,0,2\n'
        '"other:unrelated",300,900000,900000,41,1,0,3\n'
        '"TensorRT:decoder_matmul",4000,1200,60000,41,1,0,4\n'
    )
    database = tmp_path / "enqueue.sqlite"
    with _graph_database(database) as connection:
        tid = (41 << 24) | 1
        connection.executemany(
            "INSERT INTO NVTX_EVENTS (rowid,start,end,eventType,globalTid,domainId,text,eventId) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (90, 0, None, 75, tid, 8, "TensorRT", 90),
                (1, 0, 10000, 59, tid, 8, "ExecutionContext::enqueueV3", 1),
                (2, 100, 200, 59, tid, 8, "PWN(conv, relu)", 2),
                (4, 300, 400, 59, tid, 8, "decoder_matmul", 4),
            ],
        )
    rows = profile_layers.read_enqueue_trace(trace, database)
    assert [row["layer"] for row in rows] == ["PWN(conv, relu)", "decoder_matmul"]
    assert [row["gpu_span_ns"] for row in rows] == [3500, 1200]
    assert rows[0]["gpu_start_ns"] == 200
    assert rows[0]["gpu_operation_count"] == 2
    assert rows[0]["pid"] == 41


def _graph_database(path):
    import sqlite3

    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER, globalTid INTEGER,
            domainId INTEGER, textId INTEGER, text TEXT, eventId INTEGER);
        CREATE TABLE CUDA_GRAPH_NODE_EVENTS (
            graphNodeId INTEGER, graphId INTEGER, globalTid INTEGER, start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            globalPid INTEGER, correlationId INTEGER, start INTEGER, end INTEGER,
            sourceGraphNodeId INTEGER, sourceGraphId INTEGER);
    """)
    return connection


@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_graph_replay_joins_capture_ranges_to_kernels_by_graph_and_process(tmp_path):
    """Replay kernels inherit the deepest captured TensorRT layer, once per launch."""
    database = tmp_path / "graph.sqlite"
    with _graph_database(database) as connection:
        pid, other_pid = 41 << 24, 42 << 24
        tid = pid | 1
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(1, "TensorRT"), (2, "ExecutionContext::enqueueV3"), (3, "PWN(conv, relu)")],
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0, None, 75, tid, 8, 1, None, 1),
                (100, 900, 59, tid, 8, 2, None, 2),
                (200, 700, 59, tid, 8, None, "outer_layer", 3),
                (300, 500, 60, tid, 8, 3, None, 4),
                (500, 600, 70, tid, 8, None, "decoder_matmul", 5),
                (100, 900, 59, tid, 99, None, "wrong_domain", 6),
                (0, None, 75, other_pid | 1, 8, 1, None, 7),
            ],
        )
        connection.executemany(
            "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?, ?, ?, ?, ?)",
            [
                (10, 2, tid, 350, 360),
                (11, 2, tid, 400, 410),
                (12, 2, tid, 550, 560),
                (10, 3, tid, 550, 560),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            [
                (pid, 80, 1000, 1100, 10, 2),
                (pid, 80, 1200, 1400, 11, 2),
                (pid, 80, 1500, 1600, 12, 2),
                (pid, 81, 2000, 2100, 10, 2),
                (pid, 82, 2200, 2250, 10, 3),
                (pid, 83, 2300, 2400, 0, 0),
                (other_pid, 80, 2500, 2600, 10, 2),
            ],
        )
    rows = profile_layers.read_graph_trace(database)
    assert [(row["launch_id"], row["layer"]) for row in rows] == [
        (80, "PWN(conv, relu)"),
        (80, "decoder_matmul"),
        (81, "PWN(conv, relu)"),
        (82, "decoder_matmul"),
    ]
    assert rows[0]["gpu_kernel_count"] == 2
    assert rows[0]["gpu_span_ns"] == 400
    assert rows[0]["gpu_kernel_sum_ns"] == 300
    assert rows[0]["pid"] == 41
    assert rows[0]["gpu_start_ns"] == 1000


@pytest.mark.req("SP-LFFASTESTWHISPER-012")
def test_enqueue_projection_rejects_cpu_only_or_incompatible_output(tmp_path):
    """Missing GPU projections fail visibly instead of reporting CPU range duration."""
    trace = tmp_path / "invalid.csv"
    trace.write_text("Name,Orig Duration (ns)\nTensorRT:conv,80000\n")
    with pytest.raises(ValueError, match="projection"):
        profile_layers.read_enqueue_trace(trace, tmp_path / "missing.sqlite")


@pytest.mark.req("SP-LFFASTESTWHISPER-012")
@pytest.mark.parametrize("domain", ["", "myelin-exec"])
def test_enqueue_selects_inner_operations_with_their_domain_and_tensor_rt_parent(tmp_path, domain):
    """Fused operations retain their domain and TensorRT parent while unrelated ranges are excluded."""
    database = tmp_path / "operations.sqlite"
    pid = 41 << 24
    tid = pid | 1
    with _graph_database(database) as connection:
        connection.executemany(
            "INSERT INTO NVTX_EVENTS (rowid,start,end,eventType,globalTid,domainId,text,eventId) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (90, 0, None, 75, tid, 1, "TensorRT", 90),
                (1, 100, 900, 59, tid, 1, "ExecutionContext::enqueueV3", 1),
                (2, 150, 850, 59, tid, 1, "ForeignNode[encoder]", 2),
                (3, 350, 650, 59, tid, 2, "operation_group", 3),
                (4, 400, 500, 59, tid, 2, "an opaque future operation name", 4),
                (5, 400, 500, 59, pid | 2, 2, "unrelated_thread", 5),
                (6, 50, 90, 59, tid, 2, "outside_enqueue", 6),
            ],
        )
        if domain:
            connection.execute(
                "INSERT INTO NVTX_EVENTS (start,eventType,globalTid,domainId,text,eventId) "
                "VALUES (0,75,?,2,?,91)",
                (tid, domain),
            )
    trace = tmp_path / "operations.csv"
    trace.write_text(
        "Name,Projected Start (ns),Projected Duration (ns),PID,NumGPUOps,NumChild,RangeId\n"
        "TensorRT:ExecutionContext::enqueueV3,1000,6000,41,3,1,1\n"
        "TensorRT:ForeignNode[encoder],1000,5000,41,3,0,2\n"
        f"{domain}:operation_group,1000,2000,41,2,1,3\n"
        f"{domain}:an opaque future operation name,1200,400,41,1,0,4\n"
        f"{domain}:unrelated_thread,1200,9000,41,1,0,5\n"
        f"{domain}:outside_enqueue,1200,9000,41,1,0,6\n"
    )
    rows = profile_layers.read_enqueue_trace(trace, database)
    assert len(rows) == 1
    assert rows[0]["layer"] == "an opaque future operation name"
    assert rows[0]["domain"] == domain
    assert rows[0]["parent_layer"] == "ForeignNode[encoder]"
    assert rows[0]["gpu_span_ns"] == 400
    summary = profile_layers.summarize_observations(rows)
    assert summary["layers"][0]["domain"] == domain
    assert summary["layers"][0]["parent_layer"] == "ForeignNode[encoder]"
    assert summary["layers"][0]["kind"] == "other"


@pytest.mark.req("SP-LFFASTESTWHISPER-012")
@pytest.mark.parametrize("domain", ["", "myelin-exec"])
def test_graph_selects_inner_operations_and_preserves_the_tensor_rt_parent(tmp_path, domain):
    """Graph kernels use the deepest scoped operation across domains and retain their fused parent."""
    database = tmp_path / "operations.sqlite"
    pid = 41 << 24
    tid = pid | 1
    with _graph_database(database) as connection:
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0, None, 75, tid, 1, None, "TensorRT", 1),
                (100, 900, 59, tid, 1, None, "ExecutionContext::enqueueV3", 2),
                (150, 850, 59, tid, 1, None, "ForeignNode[decoder]", 3),
                (350, 650, 59, tid, 2, None, "operation_group", 4),
                (400, 500, 59, tid, 2, None, "an opaque future operation name", 5),
                (400, 500, 59, pid | 2, 2, None, "unrelated_thread", 6),
                (50, 90, 59, tid, 2, None, "outside_enqueue", 7),
            ],
        )
        if domain:
            connection.execute(
                "INSERT INTO NVTX_EVENTS (start,eventType,globalTid,domainId,text,eventId) "
                "VALUES (0,75,?,2,?,8)",
                (tid, domain),
            )
        connection.executemany(
            "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?, ?, ?, ?, ?)",
            [(10, 2, tid, 420, 430), (11, 2, tid, 70, 80)],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            [
                (pid, 80, 1000, 1200, 10, 2),
                (pid, 80, 1300, 1600, 10, 2),
                (pid, 80, 1700, 1800, 11, 2),
            ],
        )
    rows = profile_layers.read_graph_trace(database)
    assert len(rows) == 1
    assert rows[0]["layer"] == "an opaque future operation name"
    assert rows[0]["domain"] == domain
    assert rows[0]["parent_layer"] == "ForeignNode[decoder]"
    assert rows[0]["gpu_kernel_count"] == 2
    assert rows[0]["gpu_span_ns"] == 600
    assert rows[0]["gpu_kernel_sum_ns"] == 500
