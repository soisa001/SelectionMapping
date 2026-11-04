import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.susie_rscript import run_susie


_FAKE_RSCRIPT = r'''import sys
from pathlib import Path

import numpy as np

if sys.argv[1] != "--vanilla":
    raise SystemExit("missing --vanilla")

n, p, L = (int(value) for value in sys.argv[3:6])
x_path, y_path, pip_path, cs_path = map(Path, sys.argv[6:10])
X = np.fromfile(x_path, dtype="<f8").reshape(n, p)
y = np.fromfile(y_path, dtype="<f8")
if L != 2 or y.shape != (n,):
    raise SystemExit("invalid scalar or vector arguments")

((X.mean(axis=0) + y.mean()) / 10).astype("<f8").tofile(pip_path)
cs_path.write_text("L1\t0,2\n", encoding="utf-8")
if len(sys.argv) == 11:
    Path(sys.argv[10]).write_text(
        'key\tvalue\n'
        'susie_converged\tTRUE\n'
        'susie_niter\t12\n'
        'susie_version\tsynthetic\n'
        'susie_prior_variances\t0.1,0.0\n'
        'susie_n_active_effects\t1\n'
        'susie_elbo_final\t-12.5\n'
        'susie_warnings\tSynthetic warning retained\n', encoding="utf-8")
'''


class RunSusieTests(unittest.TestCase):
    def test_binary_round_trip_and_credible_sets(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([0.25, 0.75])

        with tempfile.TemporaryDirectory() as tmp:
            fake_rscript = Path(tmp) / "fake_rscript.py"
            fake_rscript.write_text(_FAKE_RSCRIPT, encoding="utf-8")
            pip, credible_sets = run_susie(
                X,
                y,
                L=2,
                rscript=[sys.executable, str(fake_rscript)],
                ensure_package=False,
            )

        np.testing.assert_allclose(pip, [0.3, 0.4, 0.5])
        self.assertEqual(credible_sets, {"L1": [0, 2]})

    def test_rejects_mismatched_inputs_before_starting_r(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            run_susie(
                np.ones((2, 3)),
                np.ones(3),
                rscript=[sys.executable],
                ensure_package=False,
            )

    def test_optional_diagnostics_and_probability_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_rscript = Path(tmp) / "fake_rscript.py"
            fake_rscript.write_text(_FAKE_RSCRIPT, encoding="utf-8")
            diagnostics = {}
            run_susie(np.zeros((2, 3)), np.zeros(2), L=2,
                      rscript=[sys.executable, str(fake_rscript)],
                      ensure_package=False, diagnostics=diagnostics)
            self.assertIs(diagnostics["susie_converged"], True)
            self.assertEqual(diagnostics["susie_niter"], 12)
            self.assertEqual(diagnostics["susie_n_active_effects"], 1)
            self.assertEqual(diagnostics["susie_elbo_final"], -12.5)
            self.assertIn("warning retained", diagnostics["susie_warnings"])
            fake_rscript.write_text(_FAKE_RSCRIPT.replace(
                'susie_converged\\tTRUE', 'susie_converged\\tFALSE'), encoding="utf-8")
            run_susie(np.zeros((2, 3)), np.zeros(2), L=2,
                      rscript=[sys.executable, str(fake_rscript)],
                      ensure_package=False, diagnostics=diagnostics)
            self.assertIs(diagnostics["susie_converged"], False)
            with self.assertRaisesRegex(RuntimeError, "invalid PIP"):
                run_susie(np.full((2, 3), 20), np.zeros(2), L=2,
                          rscript=[sys.executable, str(fake_rscript)], ensure_package=False)


if __name__ == "__main__":
    unittest.main()
