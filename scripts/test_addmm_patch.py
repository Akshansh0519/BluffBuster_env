"""
scripts/test_addmm_patch.py — Local (CPU) verification of the addmm_ dtype patch.

Tests:
  1. Baseline: unpatched addmm_ raises on dtype mismatch (Half+Float).
  2. Patch installs without error.
  3. Patched addmm_ no longer raises on dtype mismatch.
  4. Patched addmm_ produces numerically correct result (beta*self + alpha*mat1@mat2).
  5. Same-dtype addmm_ is numerically unchanged by the patch.
  6. Re-install guard: second call is a no-op and does not double-wrap.
  7. Patch survives keyword-only beta/alpha arguments.
  8. mat1 dtype mismatch (not just mat2) is also auto-cast.
"""

import sys
import traceback
import torch


# ── replicate the exact function we add to train_grpo.py ──────────────────

def _install_addmm_dtype_patch() -> bool:
    if getattr(torch.Tensor.addmm_, "_bluffbuster_patched", False):
        print("[addmm-patch] already installed, skipping")
        return True

    _orig = torch.Tensor.addmm_

    def _safe_addmm_(self, mat1, mat2, *, beta=1.0, alpha=1.0):
        if mat1.dtype != self.dtype:
            mat1 = mat1.to(self.dtype)
        if mat2.dtype != self.dtype:
            mat2 = mat2.to(self.dtype)
        return _orig(self, mat1, mat2, beta=beta, alpha=alpha)

    _safe_addmm_._bluffbuster_patched = True
    torch.Tensor.addmm_ = _safe_addmm_
    return True


def _uninstall_addmm_patch(orig):
    torch.Tensor.addmm_ = orig


PASS = 0
FAIL = 0


def _check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


# ── Test 1: baseline error ─────────────────────────────────────────────────
print("\n[1] Baseline: unpatched addmm_ raises RuntimeError on Half+Float")
orig = torch.Tensor.addmm_
acc = torch.zeros(4, 6, dtype=torch.float16)
m1  = torch.randn(4, 8, dtype=torch.float16)
m2  = torch.randn(8, 6, dtype=torch.float32)
try:
    acc.addmm_(m1, m2)
    _check("raises on Half+Float", False, "no error raised")
except RuntimeError as e:
    _check("raises on Half+Float", "dtype" in str(e).lower() or "must match" in str(e).lower(),
           str(e))

# ── Test 2: patch installs ─────────────────────────────────────────────────
print("\n[2] Patch installs without error")
try:
    ok = _install_addmm_dtype_patch()
    _check("install returns True", ok is True)
    _check("_bluffbuster_patched flag set",
           getattr(torch.Tensor.addmm_, "_bluffbuster_patched", False))
except Exception as e:
    _check("no exception during install", False, str(e))

# ── Test 3: no error after patch (Half acc + Float mat2) ──────────────────
print("\n[3] Patched addmm_ handles Half+Float without error")
acc = torch.zeros(4, 6, dtype=torch.float16)
m1  = torch.randn(4, 8, dtype=torch.float16)
m2  = torch.randn(8, 6, dtype=torch.float32)
try:
    acc.addmm_(m1, m2, beta=1.0, alpha=1.0)
    _check("no RuntimeError", True)
except RuntimeError as e:
    _check("no RuntimeError", False, str(e))

# ── Test 4: numerically correct result ────────────────────────────────────
print("\n[4] Numerically correct: result == beta*self + alpha*mat1@mat2")
beta, alpha = 0.5, 2.0
acc = torch.ones(4, 6, dtype=torch.float16) * 3.0
m1  = torch.randn(4, 8, dtype=torch.float16)
m2  = torch.randn(8, 6, dtype=torch.float32)
expected = (beta * acc + alpha * m1.float() @ m2.float()).half()
acc.addmm_(m1, m2, beta=beta, alpha=alpha)
close = torch.allclose(acc.float(), expected.float(), atol=1e-2)
_check("result matches beta*self + alpha*mat1@mat2", close,
       f"max_diff={( acc.float() - expected.float()).abs().max().item():.4f}")

# ── Test 5: same-dtype unchanged ──────────────────────────────────────────
print("\n[5] Same-dtype addmm_ numerically unchanged by patch")
acc = torch.zeros(4, 6, dtype=torch.float32)
m1  = torch.randn(4, 8, dtype=torch.float32)
m2  = torch.randn(8, 6, dtype=torch.float32)
expected = m1 @ m2
acc.addmm_(m1, m2)
close = torch.allclose(acc, expected, atol=1e-5)
_check("float32+float32 result unchanged", close,
       f"max_diff={( acc - expected).abs().max().item():.6f}")

# ── Test 6: re-install guard ──────────────────────────────────────────────
print("\n[6] Second install call is a no-op (guard)")
fn_before = torch.Tensor.addmm_
_install_addmm_dtype_patch()
fn_after = torch.Tensor.addmm_
_check("second install does not re-wrap", fn_before is fn_after)

# ── Test 7: keyword beta/alpha ────────────────────────────────────────────
print("\n[7] Keyword-only beta/alpha arguments work")
acc = torch.zeros(3, 5, dtype=torch.float16)
m1  = torch.randn(3, 7, dtype=torch.float16)
m2  = torch.randn(7, 5, dtype=torch.float32)
try:
    acc.addmm_(m1, m2, beta=0.0, alpha=1.0)
    _check("keyword beta/alpha", True)
except Exception as e:
    _check("keyword beta/alpha", False, str(e))

# ── Test 8: mat1 dtype mismatch also cast ─────────────────────────────────
print("\n[8] mat1 dtype mismatch (Float mat1 + Half acc) also auto-cast")
acc = torch.zeros(4, 6, dtype=torch.float16)
m1  = torch.randn(4, 8, dtype=torch.float32)  # mat1 is Float
m2  = torch.randn(8, 6, dtype=torch.float16)
try:
    acc.addmm_(m1, m2)
    _check("no RuntimeError for Float mat1 + Half acc", True)
except RuntimeError as e:
    _check("no RuntimeError for Float mat1 + Half acc", False, str(e))

# ── Cleanup ───────────────────────────────────────────────────────────────
_uninstall_addmm_patch(orig)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    sys.exit(1)
