"""Identity-split checks for the zero-outcome v008 version-forward."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
V001_ROOT = ATTEMPT_ROOT.parent / "v001"
if str(ATTEMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT_ROOT))

import analysis
import latency
import launcher
import power_analysis
import preseal
import terminal_workflow


ACTIVE_MODULES = (
    analysis,
    latency,
    launcher,
    power_analysis,
    preseal,
    terminal_workflow,
)
FROZEN_SCIENCE_OBJECTS = (
    "DGP_MATRIX.json",
    "candidate_grid.json",
    "cohort_seed_ledger.json",
    "outcome_mapping.json",
    "power_baseline.json",
    "power_rule.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _function(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


class _CanonicalAttemptNames(ast.NodeTransformer):
    """Erase emitted attempt metadata while retaining all scientific logic."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in {"ATTEMPT", "ACTIVE_ATTEMPT"}:
            return ast.copy_location(ast.Name(id="ACTIVE_ATTEMPT", ctx=node.ctx), node)
        return node

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        node = self.generic_visit(node)
        retained = [
            (key, value)
            for key, value in zip(node.keys, node.values, strict=True)
            if not (isinstance(key, ast.Constant) and key.value == "attempt")
        ]
        node.keys = [key for key, _ in retained]
        node.values = [value for _, value in retained]
        return node


def _canonical_function(path: Path, name: str) -> str:
    node = _CanonicalAttemptNames().visit(_function(path, name))
    ast.fix_missing_locations(node)
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize("module", ACTIVE_MODULES, ids=lambda module: module.__name__)
def test_every_active_module_derives_and_guards_v008(module: object) -> None:
    assert module.ATTEMPT_ROOT == ATTEMPT_ROOT
    assert module.ACTIVE_ATTEMPT == ATTEMPT_ROOT.name == "v008"


def test_inherited_science_identity_remains_v001_only_where_consumed() -> None:
    assert power_analysis.SCIENCE_ATTEMPT == "v001"
    assert preseal.SCIENCE_ATTEMPT == "v001"
    assert not hasattr(analysis, "ATTEMPT")
    assert not hasattr(terminal_workflow, "ATTEMPT")


def test_preoutcome_and_preselection_authorization_are_inherited() -> None:
    assert preseal.PRE_DATA_SEAL_PATH == (
        ATTEMPT_ROOT / "audit/pre_data_inheritance_seal.json"
    )
    assert preseal.PRE_SELECTION_SEAL_PATH == (
        ATTEMPT_ROOT.parent / "v006/audit/pre_selection_seal.json"
    )
    assert preseal.PRE_CONFIRMATION_SEAL_PATH == (
        ATTEMPT_ROOT / "audit/pre_confirmation_package_seal.json"
    )


@pytest.mark.parametrize("relative", FROZEN_SCIENCE_OBJECTS)
def test_frozen_science_objects_are_byte_exact_v001_carry_forwards(
    relative: str,
) -> None:
    assert _sha256(ATTEMPT_ROOT / relative) == _sha256(V001_ROOT / relative)


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    (
        ("analysis", "adaptive_compute_account"),
        ("analysis", "strongest_transition_independent_allocation"),
        ("analysis", "stagewise_rank_and_calibration"),
        ("analysis", "feature_and_gain_shift"),
        ("analysis", "analyze_regime"),
        ("analysis", "bootstrap_all"),
        ("analysis", "summarize_bootstrap"),
        ("analysis", "terminal_mapping"),
        ("analysis", "run_sealed_analysis"),
        ("latency", "benchmark_path"),
        ("latency", "path_equivalence"),
        ("latency", "expected_analytic_mixture_latency"),
        ("power_analysis", "one_sided_mean_power"),
        ("power_analysis", "upper_95_sd"),
        ("power_analysis", "_grid_for_claims"),
        ("power_analysis", "inherited_family8_yardstick"),
        ("power_analysis", "normalize_claim_summary"),
        ("power_analysis", "compute_binding_power"),
    ),
)
def test_numerical_function_ast_is_exact_after_attempt_metadata_normalization(
    module_name: str, function_name: str
) -> None:
    filename = f"{module_name}.py"
    assert _canonical_function(ATTEMPT_ROOT / filename, function_name) == (
        _canonical_function(V001_ROOT / filename, function_name)
    )
