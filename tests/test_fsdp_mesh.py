import pytest

from uv3.train.fsdp2 import resolve_mesh_shape


def test_node_local_shard_shape_scales_replicas():
    assert resolve_mesh_shape(8, num_shard=8) == (8,)
    assert resolve_mesh_shape(16, num_shard=8) == (2, 8)
    assert resolve_mesh_shape(256, num_shard=8) == (32, 8)


def test_legacy_replicate_shape_is_preserved():
    assert resolve_mesh_shape(256, num_replicate=32) == (32, 8)


def test_mesh_shape_rejects_invalid_or_conflicting_sizes():
    with pytest.raises(ValueError, match="divisible"):
        resolve_mesh_shape(10, num_shard=8)
    with pytest.raises(ValueError, match="conflicting"):
        resolve_mesh_shape(16, num_replicate=4, num_shard=8)
