#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import contextlib
import os
import logging
import platform
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(name)s [%(levelname)s] - %(message)s')
log = logging.getLogger("build")
log.setLevel(logging.DEBUG)


class BaseError(Exception):
    """Base class for errors originating from build.py"""


class BuildError(BaseError):
    """Error from running build steps"""

    def __init__(self, *messages):
        super().__init__("\n".join(messages))


class UsageError(BaseError):
    """Usage related error."""

    def __init__(self, message):
        super().__init__(message)


def _str_to_bool(s: str = None) -> bool:
    """Convert string to bool"""
    if s is None:
        return False
    match s.lower():
        case '1' | 'true' | 'on' | 'enable' | 'enabled' | 'yes':
            return True
        case _:
            return False


def _is_windows():
    return sys.platform.startswith('win')


def _is_linux():
    return sys.platform.startswith('linux')


def _resolve_executable_path(path: Path):
    executable_path = shutil.which(path)
    if executable_path is None:
        raise BuildError(f"Failed to resolve executable path for '{path}'.")
    return Path(executable_path).absolute()


def _get_config_build_dir(build_dir: Path, config: str):
    return (build_dir / config).absolute()


def _run(args, cwd=None, capture_stdout: bool = False, quiet: bool = False, library_path: Path = None,
         shell: bool = False, env: dict[str, str] = None, python_path: Path = None):

    if env is None:
        env = {}

    if isinstance(args, str):
        raise ValueError('args should be a sequence of strings, not a string')

    _env = os.environ.copy()

    if library_path:
        if _is_windows():
            if 'PATH' in _env:
                _env['PATH'] = str(library_path) + os.pathsep + _env['PATH']
            else:
                _env['PATH'] = str(library_path)
        else:
            if 'LD_LIBRARY_PATH' in _env:
                _env['LD_LIBRARY_PATH'] += os.pathsep + str(library_path)
            else:
                _env['LD_LIBRARY_PATH'] = str(library_path)

    if python_path:
        if 'PYTHONPATH' in _env:
            _env['PYTHONPATH'] += os.pathsep + str(python_path)
        else:
            _env['PYTHONPATH'] = str(python_path)

    _env.update(env)

    log.info("Running subprocess in '{}'\n  {}".format(
        cwd or os.getcwd(), ' '.join(shlex.quote(arg) for arg in args)))

    def output(is_stream_captured):
        return subprocess.PIPE if is_stream_captured else (subprocess.DEVNULL if quiet else None)

    status = subprocess.run(args, cwd=cwd, check=True, stdout=output(capture_stdout),
                            env=_env, shell=shell)

    log.debug(f'Subprocess completed. Return code: {status.returncode}')
    return status


def _update_submodules(source_dir: Path):
    _run(['git', 'submodule', 'sync', '--recursive'], cwd=source_dir)
    _run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=source_dir)


def _number_of_parallel_jobs(parallel: int):
    return max(1, os.cpu_count() - 2) if parallel == 0 else parallel


if _is_windows():
    def _extract_vs_env(vs_install_path: str = None, vs_version: str = None):
        if os.environ.get('VSINSTALLDIR') is not None:
            return {}
        if vs_install_path is None:
            vs_installer_path = f'{os.environ["ProgramFiles(x86)"]}\\Microsoft Visual Studio\\Installer'
            if vs_installer_path == '':
                raise BuildError('Could not find Visual Studio or Build Tools in the system')

            version_to_find = ['-version', '(17.0,17.15)'] \
                if vs_version is not None and '2022' in vs_version else ['-latest']
            vs_install_path = _run([f'{vs_installer_path}\\vswhere.exe',
                                    *version_to_find, '-property', 'installationPath'],
                                   capture_stdout=True).stdout.decode().strip()

        vs_dev_cmd = f'{vs_install_path.strip()}\\VC\\Auxiliary\\Build\\vcvars64.bat'
        stdout_text = _run([os.environ['ComSpec'], '/c', vs_dev_cmd, '>NUL', '&&', 'set'],
                           capture_stdout=True).stdout.decode().strip()
        vs_env = {}
        for stdout_line in stdout_text.split('\r\n'):
            equality_sign_loc = stdout_line.find('=')
            if equality_sign_loc != -1:
                key = stdout_line[:equality_sign_loc]
                value = stdout_line[equality_sign_loc + 1:]
                vs_env[key] = value

        return vs_env

@contextlib.contextmanager
def _build_environment(environ: dict[str, str] = None, vs_version: str = None):
    if environ is None:
        environ = {}

    existing_environ = dict(os.environ)
    environ.update(_extract_vs_env(os.getenv('VS_PATH'), vs_version) if _is_windows() else existing_environ)
    os.environ.clear()
    os.environ.update(environ)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(existing_environ)


def _generate_build_tree(cmake_path: Path, source_dir: Path, build_dir: Path, cmake_prefix_path: list[Path],
                         configs: set[str], cmake_extra_defines: list[str], args, cmake_extra_args: list[str]):

    log.info('Generating CMake build tree')

    cmake_args = [str(cmake_path)]

    if args.compile_no_warning_as_error:
        cmake_args += ['--compile-no-warning-as-error']

    if args.use_cache:
        cmake_args += ['-DBUILD_CACHE=ON']
        if _is_linux() and ('Ninja' in args.cmake_generator or 'Unix' in args.cmake_generator):
            cmake_args += ['-DCMAKE_CXX_COMPILER_LAUNCHER=ccache', '-DCMAKE_C_COMPILER_LAUNCHER=ccache']

    if _is_windows():
        cmake_args += ['-DUSE_DML=' + ('ON' if args.use_dml else 'OFF')]

    cmake_args += [
        '-DUSE_AMDGPU=' + ('ON' if args.use_amdgpu else 'OFF'),
        '-DUSE_MIGRAPHX=' + ('ON' if args.use_migraphx else 'OFF'),
        '-DCMAKE_PREFIX_PATH={}'.format(';'.join(path.as_posix() for path in cmake_prefix_path))
    ]

    if _is_windows():
        cmake_args += [
            '-DUSE_BINSKIM_COMPLIANT_COMPILE_FLAGS=' +
                ('ON' if args.use_binskim_compliant_compile_flags else 'OFF')
        ]

    cmake_args += [f'-D{d}' for d in cmake_extra_defines]
    cmake_args += cmake_extra_args

    for config in configs:
        config_build_dir = _get_config_build_dir(build_dir, config)
        _run([*cmake_args, f'-DCMAKE_BUILD_TYPE={config}',
              '-S', str(source_dir), '-B', str(config_build_dir)])


def _clean_targets(cmake_path: Path, build_dir: Path, configs: set[str]):
    for config in configs:
        log.info(f'Cleaning targets for {config} configuration')
        config_build_dir = _get_config_build_dir(build_dir, config)
        _run([cmake_path, '--build', config_build_dir, '--config', config, '--target', 'clean'])


def _build_targets(cmake_path: Path, build_dir: Path, configs: set[str], num_parallel_jobs: int,
                   target: list[str] = None):
    for config in configs:
        log.info(f'Building targets for {config} configuration')
        config_build_dir = _get_config_build_dir(build_dir, config)

        cmake_args = [str(cmake_path), '--build', str(config_build_dir), '--config', config]
        if target is not None and target:
            log.info(f'Building specified targets: {target}')
            cmake_args += ['-target', *target]

        if num_parallel_jobs > 1:
            cmake_args += [f'-j {num_parallel_jobs}']

        _run(cmake_args)


def _load_pyproject_toml(pyproject_path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise BuildError(
                "Neither 'tomllib' (Python 3.11+) nor 'tomli' package is available.\n"
                "Install 'tomli' with:  pip install tomli")

    with open(pyproject_path, 'rb') as f:
        return tomllib.load(f)


def _detect_ort_version(build_dir: Path, configs: set[str]) -> str | None:
    for config in configs:
        version_file = _get_config_build_dir(build_dir, config) / 'onnxruntime_version.txt'
        if version_file.is_file():
            version = version_file.read_text(encoding='utf-8').strip()
            if version:
                log.info(f"Detected ORT version from CMake build: {version}")
                return version
    return None


def _inject_ort_dependency(pyproject_path: Path, dep_spec: str):
    content = pyproject_path.read_text(encoding='utf-8')
    rp_match = re.search(r'^(requires-python\s*=\s*"[^"]*")\s*$', content, re.MULTILINE)
    if rp_match:
        insert_pos = rp_match.end()
        content = content[:insert_pos] + f'\ndependencies = [\n    "{dep_spec}",\n]' + content[insert_pos:]
    else:
        log.warning(f"Could not find requires-python in {pyproject_path}")
        return
    pyproject_path.write_text(content, encoding='utf-8')
    log.info(f"Injected dependency '{dep_spec}' into {pyproject_path}")


def _find_built_library(config_build_dir: Path, target_name: str) -> Path | None:
    if _is_windows():
        patterns = [f'{target_name}.dll']
    else:
        patterns = [f'lib{target_name}.so']

    for pattern in patterns:
        matches = sorted(config_build_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _get_python_tag() -> str:
    return f'cp{sys.version_info.major}{sys.version_info.minor}'


def _get_abi_tag() -> str:
    return f'cp{sys.version_info.major}{sys.version_info.minor}'


def _get_platform_tag() -> str:
    if _is_windows():
        return 'win_amd64'
    machine = platform.machine().lower()
    return f'manylinux_2_28_{machine}'


def _retag_wheel(dist_dir: Path):
    python_tag = _get_python_tag()
    abi_tag = _get_abi_tag()
    plat_tag = _get_platform_tag()
    for whl_file in dist_dir.glob('*.whl'):
        log.info(f'Retagging wheel {whl_file.name} -> '
                 f'{python_tag}-{abi_tag}-{plat_tag}')
        _run([
            sys.executable, '-m', 'wheel', 'tags',
            '--platform-tag', plat_tag,
            '--python-tag', python_tag,
            '--abi-tag', abi_tag,
            '--remove',
            str(whl_file),
        ])


def _generate_init_py(pkg_dir: Path, libraries: list[dict]):
    lib_filenames = [lib['filename'] for lib in libraries]
    ep_names = [lib['ep_name'] for lib in libraries if lib.get('ep_name')]

    init_content = textwrap.dedent("""\
        \"\"\"Auto-generated ONNX Runtime Execution Provider package.\"\"\"
        from __future__ import annotations

        import os
        import sys
        import pathlib

        __all__ = ['get_library_path', 'get_library_paths', 'get_ep_name', 'get_ep_names']

        _module_dir = pathlib.Path(__file__).parent

        # Add this directory to the DLL search path so that the EP library can
        # find its runtime dependencies (e.g. migraphx_c.dll, amdhip64_7.dll)
        # that are shipped alongside it in this package.
        if sys.platform == 'win32':
            _dir = str(_module_dir)
            os.add_dll_directory(_dir)
            # Also prepend to PATH for libraries loaded via LoadLibrary()
            # (e.g. by ONNX Runtime's register_execution_provider_library).
            os.environ['PATH'] = _dir + os.pathsep + os.environ.get('PATH', '')

        _LIBRARY_FILES = {lib_files!r}
        _EP_NAMES = {ep_names!r}


        def get_library_paths() -> list[str]:
            \"\"\"Return paths to all EP native libraries shipped with this package.\"\"\"
            paths = []
            for lib_file in _LIBRARY_FILES:
                p = _module_dir / lib_file
                if p.is_file():
                    paths.append(str(p))
            return paths


        def get_library_path() -> str:
            \"\"\"Return the path to the primary EP native library.\"\"\"
            paths = get_library_paths()
            if not paths:
                raise FileNotFoundError("No EP native library found in the package directory.")
            return paths[0]


        def get_ep_names() -> list[str]:
            \"\"\"Return the default EP registration name(s).\"\"\"
            return list(_EP_NAMES)


        def get_ep_name() -> str:
            \"\"\"Return the primary EP registration name.\"\"\"
            names = get_ep_names()
            if not names:
                raise ValueError("No EP names defined for this package.")
            return names[0]
    """).format(lib_files=lib_filenames, ep_names=ep_names)

    init_file = pkg_dir / '__init__.py'
    init_file.write_text(init_content, encoding='utf-8')
    log.info(f'Generated {init_file}')


def _build_python_wheel(source_dir: Path, build_dir: Path, configs: set[str], wheel_ep: str,
                        ort_version: str | None = None):
    ep_dir = source_dir / 'src' / wheel_ep
    pyproject_path = ep_dir / 'pyproject.toml'

    if not pyproject_path.is_file():
        raise BuildError(f"pyproject.toml not found at: {pyproject_path}")

    pyproject = _load_pyproject_toml(pyproject_path)
    project_meta = pyproject.get('project', {})
    ep_meta = pyproject.get('tool', {}).get('ep', {})

    pkg_name = project_meta.get('name', f'onnxruntime-ep-{wheel_ep}')
    module_name = ep_meta.get('module_name')
    if not module_name:
        raise BuildError(
            f"[tool.ep] 'module_name' is not set in {pyproject_path}.\n"
            "Each EP pyproject.toml must define [tool.ep] module_name.")

    components: list[dict] = list(ep_meta.get('components', []))

    if not components:
        raise BuildError(
            f"No components defined for EP '{wheel_ep}'.\n"
            "Define [[tool.ep.components]] entries in the EP pyproject.toml.")

    for config in configs:
        config_build_dir = _get_config_build_dir(build_dir, config)
        log.info(f"Building Python wheel for '{pkg_name}' ({config} configuration)")

        staging_dir = config_build_dir / 'wheel_staging' / pkg_name
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        pkg_dir = staging_dir / module_name
        pkg_dir.mkdir(parents=True)

        found_libraries: list[dict] = []
        copied_files: set[str] = set()
        for component in components:
            lib_target = component['library']
            src_lib = _find_built_library(config_build_dir, lib_target)
            if src_lib is None:
                log.warning(f"Built library not found for target '{lib_target}' "
                            f"in {config_build_dir} — skipping")
                continue
            dst_lib = pkg_dir / src_lib.name
            shutil.copy2(str(src_lib), str(dst_lib))
            copied_files.add(src_lib.name.lower())
            log.info(f"Copied library: {src_lib} -> {dst_lib}")
            found_libraries.append({
                'filename': src_lib.name,
                'ep_name': component.get('ep_name', ''),
            })

            if _is_windows():
                dep_patterns = ['*.dll', '*.pdb', '*.exe']
            else:
                dep_patterns = ['*.so', '*.so.*']

            for pattern in dep_patterns:
                for dep_file in src_lib.parent.glob(pattern):
                    if dep_file.name.lower() not in copied_files:
                        dst_dep = pkg_dir / dep_file.name
                        shutil.copy2(str(dep_file), str(dst_dep))
                        copied_files.add(dep_file.name.lower())
                        log.info(f"Copied dependency: {dep_file.name}")

        if not found_libraries:
            raise BuildError(
                f"No built libraries found for EP '{wheel_ep}' in {config_build_dir}.\n"
                "Ensure the native build completed successfully before building the wheel.")

        _generate_init_py(pkg_dir, found_libraries)

        staged_pyproject = staging_dir / 'pyproject.toml'
        shutil.copy2(str(pyproject_path), str(staged_pyproject))

        if ort_version:
            _inject_ort_dependency(staged_pyproject, ort_version)

        dist_dir = staging_dir / 'dist'
        log.info(f"Running 'python -m build --wheel' in {staging_dir}")
        _run([
            sys.executable, '-m', 'build',
            '--wheel',
            '--no-isolation',
            '--outdir', str(dist_dir),
        ], cwd=staging_dir, quiet=True)

        _retag_wheel(dist_dir)

        wheel_output_dir = config_build_dir / 'dist'
        wheel_output_dir.mkdir(exist_ok=True)
        for whl_file in dist_dir.glob('*.whl'):
            dst = wheel_output_dir / whl_file.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(whl_file), str(dst))
            log.info(f"Wheel ready: {dst}")

        # Clean up the staging directory and its parent if empty
        staging_parent = staging_dir.parent
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if staging_parent.exists() and not any(staging_parent.iterdir()):
            staging_parent.rmdir()
        log.info(f"Cleaned up staging directory")


def _deploy_python_wheel(build_dir: Path, configs: set[str],
                         repo_url: str | None = None,
                         token: str | None = None):
    for config in configs:
        config_build_dir = _get_config_build_dir(build_dir, config)
        dist_dir = config_build_dir / 'dist'

        wheel_files = sorted(dist_dir.glob('*.whl'))
        if not wheel_files:
            log.warning(f"No wheel files found in {dist_dir} — nothing to deploy")
            continue

        twine_args = [sys.executable, '-m', 'twine', 'upload']

        if repo_url:
            twine_args += ['--repository-url', repo_url]

        if token:
            twine_args += ['--username', '__token__', '--password', token]

        twine_args += [str(whl) for whl in wheel_files]

        log.info(f"Uploading {len(wheel_files)} wheel(s) from {dist_dir}")
        _run(twine_args)
        log.info(f"Deployment complete for {config} configuration")


def _build_nuget_package(cmake_path: Path, source_dir: Path, build_dir: Path, configs: set[str],
                         msbuild_extra_options: list[str]):
    pass


def main():
    log.debug('Command line arguments:\n {}'.format(' '.join(shlex.quote(arg) for arg in sys.argv[1:])))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build_dir",
        required=True,
        help="Path to the build directory."
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update makefiles.'
    )
    parser.add_argument(
        '--build',
        action='store_true',
        help='Run build for the selected configurations.'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Run clean for the selected configurations'
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Debug"],
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build."
    )
    parser.add_argument(
        '--parallel',
        nargs='?',
        const='0',
        default='0',
        type=int,
        help='Use parallel build. Optional value specifies max jobs (0=num CPUs).'
    )
    parser.add_argument(
        "--target",
        nargs='+',
        action='extend',
        default=[],
        help="Build one or more specific CMake target."
    )
    parser.add_argument(
        "--compile_no_warning_as_error",
        action="store_true",
        help="Prevent warnings from being treated as errors during compile."
    )
    parser.add_argument(
        "--enable_lto",
        action="store_true",
        help="Enable Link Time Optimization."
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use compiler artifacts caching (e.g. ccache)"
    )
    if _is_windows():
        parser.add_argument(
            "--use_binskim_compliant_compile_flags",
            action="store_true",
            help="Prepare for BinSkim Binary Analyzer"
        )
    parser.add_argument(
        "--cmake_extra_defines",
        nargs="+",
        action="extend",
        default=[],
        help="Etra CMake definitions (-D<key>=<value>). Provide as <key>=<value>."
    )
    parser.add_argument(
        '--cmake_path',
        default='cmake',
        help='Path to the CMake executable.'
    )
    parser.add_argument(
        '--cmake_generator',
        choices=[
            'Ninja',
            'Ninja Multi-Config',
            'NMake Makefiles',
            'NMake Makefiles JOM',
            'Unix Makefiles',
            'Visual Studio 17 2022',
            'Visual Studio 18 2026'
        ],
        default=None,
        help='Specify the generator for CMake.'
    )
    if _is_windows():
        parser.add_argument(
            '--vs_version',
            choices=[
              '17', '2022', '18', '2026'
            ],
            default=None,
            help='Visual Studio build tools to use with Ninja generator.'
    )
    # parser.add_argument(
    #     "--use_vcpkg",
    #     action="store_true",
    #     help="Use vcpkg for dependencies."
    # )
    # parser.add_argument(
    #     "--use_mimalloc",
    #     action="store_true",
    #     help="Use mimalloc memory allocator."
    # )
    parser.add_argument(
        '--build_wheel',
        action='store_true',
        help="Build Python wheel package for each enabled EP (--use_* flags)."
    )
    parser.add_argument(
        '--deploy_wheel',
        action='store_true',
        help='Deploy the built wheel(s) to a PyPI repository using twine.'
    )
    parser.add_argument(
        '--pypi_repo_url',
        default=None,
        help='PyPI repository URL for wheel deployment '
             '(default: https://upload.pypi.org/legacy/).'
    )
    parser.add_argument(
        '--pypi_token',
        default=None,
        help='PyPI API token for wheel deployment. '
             'Can also be set via the PYPI_TOKEN environment variable.'
    )
    parser.add_argument(
        '--build_nuget',
        action='store_true',
        help='Build C# library and NuGet package.'
    )
    parser.add_argument(
        "--msbuild_extra_options",
        nargs="+",
        action="extend",
        default=[],
        help="Extra MSBuild properties (/p:<key>=<value>). Provide as <key>=<value>."
    )
    parser.add_argument(
        "--use_amdgpu",
        action="store_true",
        help="Enable AMDGPU Execution Provider."
    )
    if _is_windows():
        parser.add_argument(
            "--use_dml",
            action="store_true",
            help="Enable DirectML Execution Provider."
        )
    parser.add_argument(
        "--use_migraphx",
        action="store_true",
        help="Enable MIGraphX Execution Provider."
    )
    parser.add_argument(
        "--migraphx_home",
        default=None,
        type=Path,
        help="Path to the MIGraphX installation directory."
    )
    parser.add_argument(
        "--onnxrt_home",
        default=None,
        type=Path,
        help="Path to the ONNXRuntime installation directory."
    )
    parser.add_argument(
        "--hip_path",
        default=os.getenv('HIP_PATH'),
        type=Path,
        help = "Path to the HIP SDK installation directory."
    )

    args = parser.parse_args()

    vs_version = None
    if _is_windows():
        if args.vs_version is None:
            if (args.cmake_generator is not None and
                    'Visual' in args.cmake_generator and
                    '2022' in args.cmake_generator):
                vs_version = '2022'
        else:
            vs_version = args.vs_version
            if '17' in vs_version:
                vs_version = '2022'

    _use_flag_to_ep_dir = {
        'use_amdgpu':  'amdgpu',
        'use_migraphx': 'migraphx'
    }
    if _is_windows():
        _use_flag_to_ep_dir['use_dml'] = 'directml'

    wheel_ep_dirs: list[str] = []
    if args.build_wheel:
        for flag, ep_dir in _use_flag_to_ep_dir.items():
            if getattr(args, flag, False):
                wheel_ep_dirs.append(ep_dir)

    if args.use_amdgpu:
        args.use_dml = args.use_migraphx = True

    # Default behavior when no action flags are specified
    if not args.update and not args.build and not args.clean:
        args.update = args.build = True

    print(args)

    if _str_to_bool(os.getenv("BUILD_WITH_CACHE")):
        args.use_cache = True

    if args.build_wheel and not wheel_ep_dirs:
        raise UsageError(
            "--build_wheel requires at least one --use_* EP flag.\n"
            "Example: --build_wheel --use_amdgpu")

    if args.deploy_wheel and not args.build_wheel:
        raise UsageError("--deploy_wheel requires --build_wheel to be specified as well.")

    cmake_extra_defines = list(args.cmake_extra_defines)
    msbuild_extra_options = list(args.msbuild_extra_options)
    configs = set(args.config)
    cmake_path = _resolve_executable_path(args.cmake_path)
    build_dir = Path(args.build_dir)
    script_dir = Path(__file__).resolve().parent
    source_dir = script_dir.parent.parent

    migraphx_home = os.getenv('MIGRAPHX_HOME')
    if migraphx_home is None:
        migraphx_home = os.getenv('MGX_HOME')
    if migraphx_home is None:
        migraphx_home = args.migraphx_home

    onnxrt_home = os.getenv('ONNXRT_HOME')
    if onnxrt_home is None:
        onnxrt_home = os.getenv('ORT_HOME')
    if onnxrt_home is None:
        onnxrt_home = os.getenv('ONNXRUNTIME_HOME')
    if onnxrt_home is None:
        onnxrt_home = args.onnxrt_home

    cmake_prefix_path = []
    if args.hip_path is not None:
        cmake_prefix_path += [args.hip_path]

    if onnxrt_home is not None:
        cmake_prefix_path += [
            Path(onnxrt_home) if not isinstance(onnxrt_home, Path) else onnxrt_home]

    if args.use_migraphx and migraphx_home is not None:
        cmake_prefix_path += [
            Path(migraphx_home) if not isinstance(migraphx_home, Path) else migraphx_home]

    with _build_environment(vs_version=vs_version):
        log.info("Build started")
        if args.update:
            cmake_extra_args = []
            target_arch = platform.machine().lower()
            if platform.architecture()[0] == '32bit' or 'arm' in target_arch or 'aarch' in target_arch:
                raise BuildError('ARM or 32bit architectures are not supported')

            if args.cmake_generator is not None:
                cmake_extra_args += ['-G', args.cmake_generator]

            _generate_build_tree(cmake_path, source_dir, build_dir, cmake_prefix_path, configs,
                                 cmake_extra_defines, args, cmake_extra_args)

        if args.clean:
            _clean_targets(cmake_path, build_dir, configs)

        if args.build:
            if args.parallel < 0:
                raise BuildError(f'Invalid parallel jobs count: {args.parallel}')
            _build_targets(cmake_path, build_dir, configs, _number_of_parallel_jobs(args.parallel),
                           args.target)

            if args.build_wheel:
                requirements_file = source_dir / 'requirements.txt'
                if requirements_file.is_file():
                    log.info("Installing wheel-build requirements from requirements.txt")
                    _run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                         quiet=True)

                # Auto-detect the ORT version from the build dependency
                ort_version = _detect_ort_version(build_dir, configs)
                if ort_version:
                    parts = ort_version.split('.')
                    ort_dep_spec = f'onnxruntime~={parts[0]}.{parts[1]}.0' if len(parts) >= 2 else f'onnxruntime>={ort_version}'
                    log.info(f"Installing {ort_dep_spec} from PyPI")
                    try:
                        _run([sys.executable, '-m', 'pip', 'install', ort_dep_spec])
                    except Exception:
                        log.warning(f"No compatible onnxruntime package available on PyPI "
                                    f"for version {ort_version} ({ort_dep_spec}). "
                                    f"Wheels will be built without onnxruntime dependency.")
                        ort_dep_spec = None
                else:
                    log.warning("Could not detect ONNXRuntime version — wheels will "
                                "not declare an onnxruntime dependency")
                    ort_dep_spec = None

                for ep_dir in wheel_ep_dirs:
                    _build_python_wheel(source_dir, build_dir, configs, ep_dir,
                                        ort_version=ort_dep_spec)

            if args.deploy_wheel:
                pypi_token = args.pypi_token or os.getenv('PYPI_TOKEN')
                _deploy_python_wheel(build_dir, configs,
                                     repo_url=args.pypi_repo_url,
                                     token=pypi_token)

            if args.build_nuget:
                _build_nuget_package(cmake_path, source_dir, build_dir, configs,
                                     msbuild_extra_options)

        log.info('Build complete')


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except BaseError as e:
        sys.exit(1)
