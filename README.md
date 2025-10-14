# SWARM ROBOTS FOR DYNAMIC FIREFIGHTING (ADAPTIVE COVERAGE CONTROL)
Milestone 2 – Problem Formulation as Code (Python)

## Files
- `models.py` — workspace, robot & simulation dataclasses; single-integrator dynamics; dynamic fire-front sampler.
- `objectives.py` — coverage cost, travel distance, energy-balance variance.
- `constraints.py` — soft penalties for speed, spacing, connectivity, energy, and workspace/obstacles.
- `fitness.py` — `compute_fitness(...)` returns weighted objective + penalties; plug into any optimizer.
- `example_run.py` — small demo that prints a cost breakdown and saves results.

## Install
```bash
python -m venv env
# Windows PowerShell:
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
