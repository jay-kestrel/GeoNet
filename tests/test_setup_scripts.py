"""Tests for pygeonet_check_setup.py and pygeonet_envinfo.py.

pygeonet_prepare.py resolves the project cfg and creates the directory
structure at import time, so pygeonet_check_setup.py reproduces that
resolution without importing prepare. These tests pin the resolution
semantics (project_pointer.cfg first, then the GeoNet*.cfg glob) and the
readable failure messages for the crashes prepare itself produces (a
bare IndexError on a missing cfg, configparser errors on a malformed
one).

Both scripts are stdlib-only at module level, so no dependency stubbing
is needed (unlike test_fast_marching.py); fixtures are tempfile-based
cfg files and directory trees, and every test runs without gdal/GRASS.
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygeonet_check_setup as check_setup  # noqa: E402
import pygeonet_envinfo as envinfo  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_SETUP_SCRIPT = os.path.join(REPO_ROOT, 'pygeonet_check_setup.py')
ENVINFO_SCRIPT = os.path.join(REPO_ROOT, 'pygeonet_envinfo.py')

CFG_BODY = (
    '[Section]\n'
    'geofloodhomedir = %(home)s\n'
    'projectname = my_project\n'
    'dem_name = dem\n'
    'Input_dir = GeoInputs\n'
    'Output_dir = GeoOutputs\n'
)

MISSING_MODULE = 'no_such_geonet_module_xyz'


def write_cfg(path, home):
    with open(path, 'w') as cfg_file:
        cfg_file.write(CFG_BODY % {'home': home})


def make_project_setup(tmp_dir, with_dem=True, with_outputs=True):
    """A configured project: cfg next to the scripts + populated home dir.

    Layout mirrors pygeonet_prepare.py's documented file structure with
    projectname 'my_project'.
    """
    home = os.path.join(tmp_dir, 'home')
    dem_dir = os.path.join(home, 'GeoInputs', 'GIS', 'my_project')
    os.makedirs(dem_dir)
    if with_dem:
        with open(os.path.join(dem_dir, 'dem.tif'), 'wb'):
            pass
    if with_outputs:
        os.makedirs(os.path.join(home, 'GeoOutputs', 'GIS', 'my_project'))
        os.makedirs(os.path.join(home, 'GeoOutputs', 'Inundation',
                                 'my_project'))
    cfg_path = os.path.join(tmp_dir, 'GeoNet_my_project.cfg')
    write_cfg(cfg_path, home)
    return cfg_path, home


class TempDirTestCase(unittest.TestCase):
    """TestCase with a per-test temporary directory, cleaned up after."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='geonet-qol-')
        self.addCleanup(shutil.rmtree, self.tmp, True)


def run_main(script_main, *args, **kwargs):
    """Call a script main() with captured stdout; returns (code, output)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = script_main(*args, **kwargs)
    return code, buffer.getvalue()


def run_python(*args, cwd):
    """Run a python subprocess with the repo root importable."""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [REPO_ROOT] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return subprocess.run(
        (sys.executable,) + args, cwd=cwd, env=env,
        capture_output=True, text=True, timeout=120)


def levels_by_label(results):
    return {label: level for level, label, _detail in results}


class ResolveCfgPathTests(TempDirTestCase):
    def test_pointer_cfg_takes_precedence(self):
        """A project_pointer.cfg points at the cfg, wherever it lives."""
        pointed_cfg = os.path.join(self.tmp, 'elsewhere', 'GeoNet_proj.cfg')
        with open(os.path.join(self.tmp, 'project_pointer.cfg'), 'w') as fh:
            fh.write('[CFG Directory]\nproject_cfg_pointer = %s\n' % pointed_cfg)
        self.assertEqual(check_setup.resolve_cfg_path(self.tmp),
                         (pointed_cfg, None))

    def test_glob_finds_cfg_next_to_scripts(self):
        """Without a pointer, GeoNet*.cfg next to the scripts is used."""
        cfg_path = os.path.join(self.tmp, 'GeoNet_my_project.cfg')
        write_cfg(cfg_path, os.path.join(self.tmp, 'home'))
        self.assertEqual(check_setup.resolve_cfg_path(self.tmp),
                         (cfg_path, None))

    def test_missing_cfg_reports_readable_reason(self):
        """No cfg anywhere is a readable message, not prepare's IndexError."""
        path, reason = check_setup.resolve_cfg_path(self.tmp)
        self.assertIsNone(path)
        self.assertIn('pygeonet_configure', reason)


class ReadCfgTests(TempDirTestCase):
    def test_valid_cfg_round_trips_keys(self):
        """A configured project parses into the five keys prepare reads."""
        cfg_path, home = make_project_setup(self.tmp)
        cfg = check_setup.read_cfg(cfg_path)
        self.assertEqual(cfg['geofloodhomedir'], home)
        self.assertEqual(cfg['projectname'], 'my_project')
        self.assertEqual(cfg['dem_name'], 'dem')
        self.assertEqual(cfg['Input_dir'], 'GeoInputs')
        self.assertEqual(cfg['Output_dir'], 'GeoOutputs')

    def test_content_without_section_header_raises_readably(self):
        cfg_path = os.path.join(self.tmp, 'broken.cfg')
        with open(cfg_path, 'w') as fh:
            fh.write('not a cfg at all\n')
        with self.assertRaises(check_setup.SetupError) as ctx:
            check_setup.read_cfg(cfg_path)
        self.assertIn(cfg_path, str(ctx.exception))

    def test_cfg_without_section_raises_readably(self):
        cfg_path = os.path.join(self.tmp, 'other_section.cfg')
        with open(cfg_path, 'w') as fh:
            fh.write('[Other]\nfoo = bar\n')
        with self.assertRaises(check_setup.SetupError) as ctx:
            check_setup.read_cfg(cfg_path)
        self.assertIn('[Section]', str(ctx.exception))

    def test_cfg_missing_key_raises_readably(self):
        cfg_path, _home = make_project_setup(self.tmp)
        with open(cfg_path) as cfg_file:
            lines = cfg_file.read().splitlines()
        with open(cfg_path, 'w') as fh:
            fh.write('\n'.join(line for line in lines
                               if not line.startswith('Output_dir')) + '\n')
        with self.assertRaises(check_setup.SetupError) as ctx:
            check_setup.read_cfg(cfg_path)
        self.assertIn('Output_dir', str(ctx.exception))

    def test_pointer_to_missing_cfg_raises_readably(self):
        """A pointer at a nonexistent cfg is the readable form of
        prepare's NoSectionError from config2.read(missing_path)."""
        with self.assertRaises(check_setup.SetupError) as ctx:
            check_setup.read_cfg(os.path.join(self.tmp, 'absent.cfg'))
        self.assertIn('does not exist or is not readable', str(ctx.exception))


class RunChecksTests(TempDirTestCase):
    def test_complete_setup_has_no_warnings_or_failures(self):
        """A fully populated project checks clean."""
        _cfg_path, home = make_project_setup(self.tmp)
        cfg = check_setup.read_cfg(os.path.join(self.tmp,
                                                'GeoNet_my_project.cfg'))
        results = check_setup.run_checks(cfg)
        self.assertEqual(levels_by_label(results),
                         {label: check_setup.OK
                          for label in check_setup.expected_layout(cfg)})
        self.assertTrue(os.path.isdir(home))

    def test_missing_outputs_dirs_warn_but_do_not_fail(self):
        """Missing output dirs are warnings: prepare creates them itself."""
        make_project_setup(self.tmp, with_outputs=False)
        cfg = check_setup.read_cfg(os.path.join(self.tmp,
                                                'GeoNet_my_project.cfg'))
        levels = levels_by_label(check_setup.run_checks(cfg))
        self.assertEqual(levels['Outputs/GIS/my_project'], check_setup.WARN)
        self.assertEqual(levels['Outputs/Inundation/my_project'],
                         check_setup.WARN)
        self.assertNotIn(check_setup.FAIL, levels.values())
        code, output = run_main(check_setup.main, [],
                                script_dir=self.tmp)
        self.assertEqual(code, 0, output)

    def test_missing_dem_fails(self):
        """A missing input DEM is the failure the check exists for."""
        make_project_setup(self.tmp, with_dem=False)
        cfg = check_setup.read_cfg(os.path.join(self.tmp,
                                                'GeoNet_my_project.cfg'))
        levels = levels_by_label(check_setup.run_checks(cfg))
        self.assertEqual(levels['Input DEM (dem.tif)'], check_setup.FAIL)
        code, output = run_main(check_setup.main, [],
                                script_dir=self.tmp)
        self.assertEqual(code, 1)
        self.assertIn('Input DEM (dem.tif)', output)

    def test_missing_home_fails(self):
        make_project_setup(self.tmp)
        shutil.rmtree(os.path.join(self.tmp, 'home'))
        code, output = run_main(check_setup.main, [],
                                script_dir=self.tmp)
        self.assertEqual(code, 1)
        self.assertIn('FAIL GeoNet home directory', output)


class MainCliTests(TempDirTestCase):
    def test_main_reports_missing_cfg_readably(self):
        """A cfg-less directory exits 1 and names the fix."""
        code, output = run_main(check_setup.main, [],
                                script_dir=self.tmp)
        self.assertEqual(code, 1)
        self.assertIn('FAIL cfg resolution', output)
        self.assertIn('pygeonet_configure', output)

    def test_main_imports_flag_reports_ok_and_missing(self):
        """--imports reports each probe and fails when one is missing."""
        make_project_setup(self.tmp)
        code, output = run_main(check_setup.main, ['--imports'],
                                script_dir=self.tmp, probe_names=('json',))
        self.assertEqual(code, 0, output)
        self.assertIn('OK json:', output)
        code, output = run_main(check_setup.main, ['--imports'],
                                script_dir=self.tmp,
                                probe_names=(MISSING_MODULE,))
        self.assertEqual(code, 1)
        self.assertIn('FAIL %s:' % MISSING_MODULE, output)
        self.assertIn('ModuleNotFoundError', output)


class ImportCleanlinessTests(TempDirTestCase):
    def test_scripts_import_without_side_effects(self):
        """Importing both scripts in a cfg-less directory is a no-op.

        pygeonet_prepare.py cannot be imported in a cfg-less directory
        (it crashes at import time); these scripts must not inherit that
        behavior, directly or through a transitive import.
        """
        proc = run_python(
            '-c', 'import pygeonet_check_setup, pygeonet_envinfo; '
                  'print("import-ok")', cwd=self.tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('import-ok', proc.stdout)
        # no cfg resolution, no directory creation, nothing written
        self.assertEqual(os.listdir(self.tmp), [])

    def test_check_setup_cli_reports_missing_cfg(self):
        """The real CLI, run from anywhere, reports the missing cfg.

        The script is copied into the temp dir so cfg resolution is
        hermetic: other tests leave a real GeoNet_*.cfg in the repo root
        (gitignored), which would otherwise be picked up.
        """
        script_copy = os.path.join(self.tmp, 'pygeonet_check_setup.py')
        shutil.copy(CHECK_SETUP_SCRIPT, script_copy)
        proc = run_python(script_copy, cwd=self.tmp)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn('pygeonet_configure', proc.stdout)

    def test_check_setup_cli_passes_on_complete_setup(self):
        """The real CLI exits 0 on a fully populated project."""
        make_project_setup(self.tmp)
        script_copy = os.path.join(self.tmp, 'pygeonet_check_setup.py')
        shutil.copy(CHECK_SETUP_SCRIPT, script_copy)
        proc = run_python(script_copy, cwd=self.tmp)
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn('setup check passed', proc.stdout)

    def test_envinfo_cli_succeeds_without_cfg_or_git(self):
        """The environment report never fails for lack of setup."""
        script_copy = os.path.join(self.tmp, 'pygeonet_envinfo.py')
        shutil.copy(ENVINFO_SCRIPT, script_copy)
        proc = run_python(script_copy, cwd=self.tmp)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('Python:', proc.stdout)
        self.assertIn('Dependencies:', proc.stdout)


class EnvInfoTests(TempDirTestCase):
    def test_git_commit_outside_repo_is_none(self):
        self.assertIsNone(envinfo.git_commit(self.tmp))

    def test_git_commit_in_repo(self):
        commit = envinfo.git_commit(REPO_ROOT)
        self.assertTrue(commit)
        self.assertIsInstance(commit, str)

    def test_dependency_versions_degrade_gracefully(self):
        self.assertNotEqual(envinfo.dependency_version('json'),
                            'not installed')
        self.assertEqual(envinfo.dependency_version(MISSING_MODULE),
                         'not installed')

    def test_collect_env_info_shape(self):
        info = envinfo.collect_env_info(repo_dir=self.tmp,
                                        dependencies=('json',
                                                      MISSING_MODULE))
        self.assertTrue(info['python'])
        self.assertTrue(info['platform'])
        self.assertIsNone(info['git_commit'])
        self.assertEqual(info['dependencies'],
                         {'json': envinfo.dependency_version('json'),
                          MISSING_MODULE: 'not installed'})

    def test_format_env_info_lists_every_section(self):
        info = envinfo.collect_env_info(repo_dir=self.tmp,
                                        dependencies=('json',
                                                      MISSING_MODULE))
        formatted = envinfo.format_env_info(info)
        self.assertIn('GeoNet environment information', formatted)
        self.assertIn('Python: %s' % info['python'], formatted)
        self.assertIn('Git commit: unknown (not a git repository)', formatted)
        self.assertIn('  json:', formatted)
        self.assertIn('  %s: not installed' % MISSING_MODULE, formatted)

    def test_main_always_succeeds(self):
        code, output = run_main(envinfo.main, repo_dir=self.tmp)
        self.assertEqual(code, 0)
        self.assertIn('GeoNet environment information', output)


if __name__ == '__main__':
    unittest.main()
