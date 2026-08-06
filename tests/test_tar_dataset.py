import io

from PIL import Image

from uv3.data.tar_dataset import TarDescriptorDecoder, partition_shard_indices


def test_shard_partition_is_disjoint_and_complete_for_four_nodes():
    entry_count = 84_367
    total_workers = 4 * 8 * 4
    partitions = [
        partition_shard_indices(entry_count, 0, True, 7, worker, total_workers)
        for worker in range(total_workers)
    ]
    assigned = [index for partition in partitions for index in partition]
    assert len(assigned) == entry_count
    assert len(set(assigned)) == entry_count
    assert set(assigned) == set(range(entry_count))


def test_shard_partition_is_deterministic_per_epoch():
    first = partition_shard_indices(1_000, 0, True, 3, 11, 128)
    second = partition_shard_indices(1_000, 0, True, 3, 11, 128)
    changed = partition_shard_indices(1_000, 0, True, 4, 11, 128)
    assert first == second
    assert first != changed


def test_deferred_tar_descriptor_decodes_selected_shape(tmp_path):
    image = Image.new("RGB", (80, 40), (20, 40, 60))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    prefix = b"descriptor-prefix"
    path = tmp_path / "images.tar"
    path.write_bytes(prefix + encoded.getvalue())
    pixels = TarDescriptorDecoder()({
        "image_tar": str(path),
        "offset": len(prefix),
        "size": len(encoded.getvalue()),
        "image_height": 32,
        "image_width": 48,
    })
    assert pixels.shape == (3, 32, 48)
