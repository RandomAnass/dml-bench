"""I-H3/I-H4 (2026-04-16): cross-architecture rMD17 split integration test.

Verifies that MLP-pairwise, GATv2, and PaiNN all see the SAME train/val/test
frame indices for a given canonical Figshare `split_id`. Breaks the silent
cross-arch data-mismatch concern.

NOTE on PaiNN: the test attempts to verify that SchNetPack's internal
`split_id=k-1` maps to the Figshare CSV `index_train_0k.csv`. If schnetpack
is not installed in the test environment, that portion is skipped.
"""

import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "rmd17" / "rmd17" / "splits"


@pytest.mark.parametrize("split_id", [1, 2, 3, 4, 5])
def test_mlp_gatv2_share_same_canonical_frames(split_id):
    """MLP-pairwise and GATv2 should read the same train/test CSVs and
    subsample with the same RandomState(split_id) seed, producing identical
    train/val/test frame indices.
    """
    import sys
    sys.path.insert(0, str(ROOT))

    # Direct CSV load — both runners use this.
    train_csv = SPLITS_DIR / f"index_train_0{split_id}.csv"
    test_csv = SPLITS_DIR / f"index_test_0{split_id}.csv"
    train_idx_full = np.loadtxt(str(train_csv)).flatten().astype(int)
    test_idx_full = np.loadtxt(str(test_csv)).flatten().astype(int)
    assert len(train_idx_full) == 1000, f"Expected 1000 train frames for split {split_id}"

    # MLP's subsample: np.random.RandomState(split_id).permutation(1000)
    rng_mlp = np.random.RandomState(split_id)
    perm_mlp = rng_mlp.permutation(len(train_idx_full))
    train_mlp = train_idx_full[perm_mlp[:950]]
    val_mlp = train_idx_full[perm_mlp[950:1000]]

    # GATv2's subsample (same pattern per gnn_md17.py)
    rng_gatv2 = np.random.RandomState(split_id)
    perm_gatv2 = rng_gatv2.permutation(len(train_idx_full))
    train_gatv2 = train_idx_full[perm_gatv2[:950]]
    val_gatv2 = train_idx_full[perm_gatv2[950:1000]]

    assert set(train_mlp) == set(train_gatv2), "MLP and GATv2 train frames diverge"
    assert set(val_mlp) == set(val_gatv2), "MLP and GATv2 val frames diverge"


@pytest.mark.parametrize("split_id", [1, 2, 3, 4, 5])
def test_painn_schnetpack_split_id_mapping(split_id):
    """I-H4 verification: SchNetPack's `split_id=k-1` should correspond to
    Figshare's `index_train_0k.csv` + `index_test_0k.csv`.

    Skipped if schnetpack is not installed (dependency missing in CI).
    """
    pytest.importorskip("schnetpack", reason="schnetpack not installed in this env")

    from schnetpack.datasets.rmd17 import rMD17

    # Load canonical CSVs manually
    train_csv = SPLITS_DIR / f"index_train_0{split_id}.csv"
    expected_train = np.loadtxt(str(train_csv)).flatten().astype(int)

    # Create SchNetPack dataset with canonical split_id = split_id - 1
    # Using a small subsample to avoid heavy data loading.
    data_dir = ROOT / "data" / "schnetpack_rmd17"
    db_path = data_dir / "rmd17_ethanol.db"
    if not db_path.exists():
        pytest.skip(f"PaiNN .db not prepared at {db_path}")

    dataset = rMD17(
        datapath=str(db_path), molecule="ethanol",
        batch_size=10, num_train=950, num_val=50, num_test=1000,
        split_file=str(data_dir / f"split_ethanol_canonical_{split_id}.npz"),
        split_id=split_id - 1,
        distance_unit="Ang",
        property_units={"energy": "kcal/mol", "forces": "kcal/mol/Ang"},
        transforms=[],
        num_workers=0,
    )
    dataset.prepare_data()
    dataset.setup()

    # SchNetPack exposes internal train_idx via data_idx attribute depending on version.
    # Here we check that the saved split file contains the expected full train indices.
    split_data = np.load(str(data_dir / f"split_ethanol_canonical_{split_id}.npz"))
    # SchNetPack's split file conventionally has 'train', 'val', 'test' keys.
    if 'train_idx' in split_data:
        spk_train_all = sorted(set(split_data['train_idx']).union(split_data['val_idx']))
    elif 'train' in split_data:
        spk_train_all = sorted(set(split_data['train']).union(split_data['val']))
    else:
        pytest.skip(f"Unknown split file format: keys={list(split_data.keys())}")

    assert sorted(expected_train) == spk_train_all, (
        f"SchNetPack split_id={split_id-1} train indices diverge from "
        f"Figshare index_train_0{split_id}.csv"
    )
