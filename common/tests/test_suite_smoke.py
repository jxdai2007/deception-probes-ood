import os
def test_all_modules_import():
    import jrp_common.metrics, jrp_common.probes, jrp_common.activations, jrp_common.capability  # noqa

def test_slurm_templates_exist():
    base = os.path.join(os.path.dirname(__file__), "..", "env", "slurm")
    assert os.path.exists(os.path.join(base, "eval_job.sbatch"))
    assert os.path.exists(os.path.join(base, "activation_dump.sbatch"))
