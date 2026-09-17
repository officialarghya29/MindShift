# Notebooks

Exploration notebooks go here. The full pipeline is runnable headlessly —
see the README quickstart:

- corpus stats: `python -m cerebro.data.generator`
- training + evaluation: `python -m evaluation.run_full fit && python -m evaluation.run_full eval`
- graphs: `python -m evaluation.make_graphs`
