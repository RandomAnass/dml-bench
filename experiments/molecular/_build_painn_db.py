#!/usr/bin/env python3
"""
Pre-build SchNetPack ASE .db files for rMD17 from the local .npz dataset.

SchNetPack's rMD17.prepare_data() normally downloads from Figshare. We have
the dataset locally at data/rmd17/rmd17/npz_data/ and just need to convert
each .npz to ASE .db format so SchNetPack will pick it up directly.

Run once:
  python experiments/molecular/_build_painn_db.py

Idempotent: skips molecules whose .db already exists with correct metadata.
"""
import sys
import os
import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MOLECULES = [
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
]

# Mirror the runner's name remap (run_painn.py:_SCHNETPACK_MOL_KEY).
# SchNetPack v2.2.0's rMD17 datamodule keys salicylic acid as
# "salicylic_acid" (see rmd17.py:57: salicylic_acid="rmd17_salicylic.npz").
# Our internal naming uses "salicylic" everywhere (filenames, splits,
# aggregator keys), but the .db metadata MUST be keyed as "salicylic_acid"
# so the runner's post-remap query lands. Other molecules pass through
# unchanged.
_DB_METADATA_MOL_KEY = {"salicylic": "salicylic_acid"}


def build_db_for_molecule(npz_path: Path, db_path: Path, molecule: str):
    """Convert one rMD17 .npz file to a SchNetPack ASE .db file."""
    if db_path.exists():
        log.info(f"  Already exists: {db_path}")
        return

    from schnetpack.data import ASEAtomsData, AtomsDataFormat
    from schnetpack.datasets.rmd17 import rMD17
    import schnetpack.properties as structure

    log.info(f"Loading {npz_path}")
    raw = np.load(str(npz_path))
    coords = raw["coords"]            # (N, n_atoms, 3)
    energies = raw["energies"]        # (N,)
    forces = raw["forces"]            # (N, n_atoms, 3)
    numbers = raw["nuclear_charges"]  # (n_atoms,)

    log.info(f"  {molecule}: {len(coords)} frames, {len(numbers)} atoms")

    log.info(f"Creating ASE dataset at {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ASEAtomsData.create(
        datapath=str(db_path),
        distance_unit="Ang",
        property_unit_dict={
            rMD17.energy: "kcal/mol",
            rMD17.forces: "kcal/mol/Ang",
        },
        atomrefs=rMD17.atomrefs,
    )
    # Write the SchnetPack-canonical metadata key (e.g. "salicylic_acid"),
    # not our internal alias ("salicylic"), so the runner's rMD17(...)
    # query matches. Verified 2026-05-02 against the live rebuild that
    # restored salicylic PaiNN function.
    metadata_molecule = _DB_METADATA_MOL_KEY.get(molecule, molecule)
    dataset.update_metadata(molecule=metadata_molecule)

    # P3 (2026-04-13): also record the 5 canonical Figshare splits in the
    # metadata so SchNetPack's split_id parameter works (mirrors what
    # SchNetPack's _download_data does after extracting the tarball).
    splits_root = ROOT / "data" / "rmd17" / "rmd17" / "splits"
    if splits_root.exists():
        train_splits = []
        test_splits = []
        for i in range(1, 6):
            train_csv = splits_root / f"index_train_0{i}.csv"
            test_csv = splits_root / f"index_test_0{i}.csv"
            if train_csv.exists() and test_csv.exists():
                train_splits.append(
                    np.loadtxt(str(train_csv)).flatten().astype(int).tolist()
                )
                test_splits.append(
                    np.loadtxt(str(test_csv)).flatten().astype(int).tolist()
                )
        if train_splits and test_splits:
            dataset.update_metadata(splits={"known": train_splits, "test": test_splits})
            log.info(f"  Added {len(train_splits)} canonical splits to metadata")

    log.info(f"  Writing {len(coords)} systems...")
    property_list = []
    for positions, e, f in zip(coords, energies, forces):
        ats = Atoms(positions=positions, numbers=numbers)
        properties = {
            rMD17.energy: np.array([e]),
            rMD17.forces: f,
            structure.Z: ats.numbers,
            structure.R: ats.positions,
            structure.cell: ats.cell,
            structure.pbc: ats.pbc,
        }
        property_list.append(properties)

    dataset.add_systems(property_list=property_list)
    log.info(f"  Done: {db_path}")


def main():
    npz_dir = ROOT / "data" / "rmd17" / "rmd17" / "npz_data"
    db_dir = ROOT / "data" / "schnetpack_rmd17"
    db_dir.mkdir(parents=True, exist_ok=True)

    if not npz_dir.exists():
        raise SystemExit(f"rMD17 .npz directory not found at {npz_dir}")

    for molecule in MOLECULES:
        npz_path = npz_dir / f"rmd17_{molecule}.npz"
        db_path = db_dir / f"rmd17_{molecule}.db"
        if not npz_path.exists():
            log.warning(f"SKIP: {npz_path} missing")
            continue
        build_db_for_molecule(npz_path, db_path, molecule)


if __name__ == "__main__":
    main()
