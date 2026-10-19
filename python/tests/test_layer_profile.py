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
    from listenflow.fastest_whisper._inference import _HostIO

    source = "def _pad_cache(self, inputs):\n    boundaries = (1, 7, 19, 41)\n"
    assert profile_layers.decoder_buckets(source=source) == (1, 7, 19, 41)
    assert profile_layers.decoder_buckets()[0] == 1
    assert profile_layers.decoder_buckets()[-1] == 448
    assert _HostIO._pad_cache.__name__ == "_pad_cache"


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
