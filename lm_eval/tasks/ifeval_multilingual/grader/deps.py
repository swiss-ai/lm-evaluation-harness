"""
Grader dependencies, installed on first use so the task needs nothing beyond the evaluation environment.

A few native instructions are scored with language analyzers:
    stanza       de diminutives, el formal plural, pl non-virile past (and grammar-form entries)
    camel_tools  ar broken plurals, idafa chains (MLE), verbal sentences (BERT)
    opencc       zh traditional characters
    pymorphy3    ru / uk verb aspect and case (not in v15)

Each is imported from the environment if present, otherwise installed with `pip install --target` into a cache
(packages only, never their dependencies: a package already in the environment is not installed again, and torch /
transformers / numpy always come from the environment). Models and data go into the same cache. A file lock
serializes installs of jobs sharing the cache. Failures raise: an instruction is never skipped silently.

Cache: $IFEVAL_MULTILINGUAL_CACHE, else ~/.cache/ifeval_multilingual. Needs pip and internet on first use.
Stanza models: $STANZA_RESOURCES_DIR if set; CAMeL data: $CAMELTOOLS_DATA if set.
"""

import contextlib
import fcntl
import functools
import importlib
import importlib.metadata
import os
import subprocess
import sys

CACHE = os.path.expanduser(os.environ.get("IFEVAL_MULTILINGUAL_CACHE") or "~/.cache/ifeval_multilingual")
LIB = os.path.join(CACHE, f"lib-py{sys.version_info.major}{sys.version_info.minor}")

# distributions per analyzer, pinned as when the dataset was made (ritranslation-hooks scripts/ifeval/setup_taggers.sh);
# each installed without its dependencies, only if the environment lacks it. torch and transformers are not listed:
# they must come from the environment (a second torch would clash with the container's).
PACKAGES = {
    "stanza": ["stanza==1.10.1", "emoji", "protobuf>=3.15.0", "networkx", "tqdm", "requests"],
    # camel-kenlm (C++, no aarch64 wheel) left out: the disambiguators do not use it
    "camel_tools": ["camel-tools==1.6.0", "emoji", "networkx", "tqdm", "requests", "cachetools>=6.0.0", "dill>=0.4.0",
                    "docopt", "editdistance", "future", "muddler", "pandas", "pyrsistent", "regex", "scikit-learn",
                    "scipy", "six", "tabulate"],
    "opencc": ["opencc-python-reimplemented==0.1.7"],
    "pymorphy3": ["pymorphy3==2.0.6", "dawg2-python", "pymorphy3-dicts-ru==2.4.417150.4580142",
                  "pymorphy3-dicts-uk==2.4.1.1.1663094765"],
}
# modules each analyzer imports from the environment
NEEDS = {"stanza": ["torch"], "camel_tools": ["torch", "transformers"], "opencc": [], "pymorphy3": []}
# CAMeL data packages per disambiguator
CAMEL_DATA = {"calima-msa-r13": ["morphology-db-msa-r13", "disambig-mle-calima-msa-r13"],
              "bert-unfactored-msa": ["morphology-db-msa-r13", "disambig-bert-unfactored-msa"]}


def stanza_dir():
    return os.environ.get("STANZA_RESOURCES_DIR") or os.path.join(CACHE, "stanza_resources")


def camel_dir():
    return os.environ.get("CAMELTOOLS_DATA") or os.path.join(CACHE, "camel_tools_data")


@contextlib.contextmanager
def lock():
    """Exclusive lock on the cache, across processes and jobs."""
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, ".lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _name(requirement):
    for sep in "=<>!~[":
        requirement = requirement.split(sep)[0]
    return requirement.strip()


def _installed(requirement):
    try:
        importlib.metadata.version(_name(requirement))
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _use_lib():
    """The cache's packages, after the environment's: an environment package always wins."""
    if LIB not in sys.path:
        sys.path.append(LIB)
        importlib.invalidate_caches()


def _install(analyzer):
    missing = [r for r in PACKAGES[analyzer] if not _installed(r)]
    if not missing:
        return
    os.makedirs(LIB, exist_ok=True)
    command = [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", LIB, *missing]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ifeval_multilingual: installing {missing} for {analyzer} failed:\n{result.stderr[-3000:]}")
    importlib.invalidate_caches()


def _camel_data(model):
    folder = camel_dir()
    os.makedirs(folder, exist_ok=True)      # camel_data does not create its folder
    for name in CAMEL_DATA.get(model, []):
        mark = os.path.join(folder, f".installed-{name}")
        if os.path.exists(mark):
            continue
        env = {**os.environ, "CAMELTOOLS_DATA": folder, "PYTHONPATH": os.pathsep.join([*sys.path])}
        result = subprocess.run([sys.executable, "-m", "camel_tools.cli.camel_data", "-i", name], input="y\n",
                                capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"ifeval_multilingual: CAMeL data {name} failed:\n{result.stdout[-1500:]}{result.stderr[-1500:]}")
        open(mark, "w").close()


@functools.lru_cache(maxsize=None)
def ensure(analyzer, model=None):
    """Makes `analyzer` importable (and, for camel_tools, the data of `model` installed); raises if it cannot."""
    for module in NEEDS[analyzer]:
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError(f"ifeval_multilingual: {analyzer} needs {module} in the evaluation environment") from error
    _use_lib()
    if analyzer == "camel_tools":
        os.environ.setdefault("CAMELTOOLS_DATA", camel_dir())     # read by camel_tools when it is imported
    with lock():
        _install(analyzer)
        if analyzer == "camel_tools" and model:
            _camel_data(model)
    try:
        importlib.import_module(analyzer)
    except ImportError as error:
        raise RuntimeError(f"ifeval_multilingual: {analyzer} installed in {LIB} but cannot be imported: {error}") from error
