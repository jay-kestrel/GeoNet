# SPDX-License-Identifier: GPL-3.0-or-later
"""Full stage-5 oracle/probe runner, adapted from archived parallel_frozen.py.

Uses an isolated copy, never overwrites the hash-pinned oracle. Scientific
runtime must match the archived Python 3.7.7 environment for TIFF-byte parity.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from golden import CASES, assert_raster_matches_golden, sha256_file, verify_case

REPO = Path(__file__).resolve().parents[1]


def prepare_case(root, work, case):
    verify_case(root, case)
    source = work / "source"
    source.mkdir(parents=True)
    for path in REPO.glob("pygeonet_*.py"):
        shutil.copy2(str(path), str(source / path.name))
    for folder in ("GeoInputs", "GeoOutputs"):
        destination = work / "data" / folder / "GIS" / case
        shutil.copytree(str(root / "data" / folder / "GIS" / case), str(destination))
    return source


def stage_environment(mode, parallel):
    env = dict(os.environ, GEONET_KERNELS=mode, MPLBACKEND="Agg",
               PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    if parallel:
        env.pop("NUMBA_NUM_THREADS", None)
        env.pop("RAYON_NUM_THREADS", None)
    else:
        env.update(NUMBA_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    return env


def runtime_manifest(python, source, env):
    code = """
import json, os, sys, numba, numpy, skfmm, geonet_kernels
from geonet_kernels import geonet_kernels as binary
from osgeo import gdal
import pygeonet_fast_marching as stage
import pygeonet_kernels as dispatch
print(json.dumps(dict(python=sys.version, numpy=numpy.__version__,
    skfmm=skfmm.__version__, gdal=gdal.VersionInfo(),
    numba_threads=numba.get_num_threads(), logical_cpus=os.cpu_count(),
    native_module=binary.__file__,
    native_sha256=__import__('hashlib').sha256(open(binary.__file__, 'rb').read()).hexdigest(),
    dispatch_native=dispatch._native_module() is not None,
    stage_imported=stage.__name__)))
"""
    text = subprocess.check_output([python, "-c", code], cwd=str(source), env=env,
                                   universal_newlines=True)
    return json.loads(text.strip().splitlines()[-1])


def run_case(root, work, case, python, mode="rust", attempts=1, parallel=False):
    source = prepare_case(root, work, case)
    env = stage_environment(mode, parallel)
    subprocess.run([python, str(source / "pygeonet_configure.py"), "-dir",
                    str(work / "data"), "-p", case, "-n", case],
                   cwd=str(source), env=env, check=True, stdout=subprocess.PIPE)
    manifest = runtime_manifest(python, source, env)
    if manifest["numpy"] != "1.19.1" or manifest["skfmm"] != "2019.1.30":
        raise RuntimeError("Golden stage requires NumPy 1.19.1 / scikit-fmm 2019.1.30")
    if mode == "rust" and not manifest["dispatch_native"]:
        raise AssertionError("Rust was requested but native dispatch did not resolve")
    if parallel and (manifest["numba_threads"] != 8 or manifest["logical_cpus"] != 8):
        raise RuntimeError("Default-parallel acceptance requires an eight-CPU host")
    changed = [case + "_" + suffix + ".tif" for suffix in ("costfunction", "geodesicDistance")]
    output = work / "data" / "GeoOutputs" / "GIS" / case
    records = []
    report = dict(case=case, mode=mode, parallel=parallel, runtime=manifest,
                  input_sha256=CASES[case]["input"]["sha256"], records=records,
                  source_sha256={p.name: sha256_file(p) for p in source.glob("*.py")})
    try:
        for attempt in range(attempts):
            # A failed/no-op stage must not pass by leaving canonical files behind.
            for name in changed:
                (output / name).unlink()
            log = work / ("stage5-{}.log".format(attempt))
            with log.open("w") as stream:
                process = subprocess.run([python, str(source / "pygeonet_fast_marching.py")],
                                         cwd=str(source), env=env, stdout=stream,
                                         stderr=subprocess.STDOUT)
            record = dict(attempt=attempt, returncode=process.returncode, log=str(log))
            records.append(record)
            if process.returncode:
                raise AssertionError("Stage 5 failed: {} (see {})".format(process.returncode, log))
            record["sha256"] = {name: sha256_file(output / name) for name in changed}
            for name in changed:
                assert_raster_matches_golden(output / name, root, case, name)
            record["byte_identical"] = True
    finally:
        for name in changed:
            shutil.copy2(str(root / CASES[case]["outputs"][name]["path"]), str(output / name))
            assert_raster_matches_golden(output / name, root, case, name)
        report["canonical_restored"] = True
        (work / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    reports = []
    for case in CASES:
        reports.append(run_case(args.root.resolve(), args.work.resolve() / case,
                                case, args.python, attempts=5, parallel=True))
    (args.work / "probes.json").write_text(json.dumps(reports, indent=2))
    print("15/15 default-parallel stage-5 processes byte-identical; canonical files restored")


if __name__ == "__main__":
    main()
