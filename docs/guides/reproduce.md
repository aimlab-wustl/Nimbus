# Reproducing the paper

The repository reproduces every result panel and table of the paper at three levels:

1. **Draw the figures** from the plotted values in `source_data/` (minutes, no runs needed).
2. **Rerun the analysis** that writes `source_data/` from the raw run outputs, which are archived at Zenodo.
3. **Rerun the experiments** themselves, as described in each experiment folder.

## 1. Draw the figures

```bash
pip install -e ".[plot]"
python figures/make_all.py        # every panel and table -> figures/out/ (PDF and PNG)
python figures/fig5d.py           # one panel
```

Each panel script reads one or more CSVs in `source_data/` and draws the panel with its labels and legend.

| Paper item | Experiment folder | Panel scripts (`figures/`) | Source data (`source_data/`) |
|---|---|---|---|
| Fig. 3c-f | [graph_matching][gm] | `fig3c.py`, `fig3d.py`, `fig3e.py`, `fig3f.py` | `fig3/` |
| Fig. 4a-e | [graph_matching][gm] | `fig4a.py` - `fig4e.py` | `fig4/` |
| Fig. 4g | [graph_matching][gm] | `fig4g.py` | `fig4/fig4g.csv` |
| Fig. 5c-i | [surface_code][sc] | `fig5c.py` - `fig5i.py` | `fig5/` |
| Fig. 6a-b | [event_driven][ed] | `fig6a.py`, `fig6b.py` | `fig6/` |
| Fig. S1a-c | [graph_matching][gm] | `figS1a.py`, `figS1b.py`, `figS1c.py` | `figS1/` |
| Fig. S2a-b | [graph_matching][gm] | `figS2a.py`, `figS2b.py` | `figS2/` |
| Fig. S3a-e | [event_driven][ed] | `figS3a.py` - `figS3e.py` | `figS3/` |
| Fig. S4a-b | [graph_matching][gm] | `figS4a.py`, `figS4b.py` | `figS4/` |
| Fig. S7a-e | [surface_code][sc] | `figS7a.py` - `figS7e.py` | `figS7/` |
| Fig. S8a-c | [surface_code][sc] | `figS8a.py`, `figS8b.py`, `figS8c.py` | `figS8/` |
| Fig. S9a-d | [event_driven][ed] | `figS9a.py` - `figS9d.py` | `figS9/` |
| Table S2 | [graph_matching][gm] | `tableS2.py` | `tables/tableS2.csv` |
| Table S3 | [graph_matching][gm] | `tableS3.py` | `tables/tableS3.csv` |
| Table S4 | [graph_matching][gm] | `tableS4.py` | `tables/tableS4.csv` |
| Table S5 | [surface_code][sc] | `tableS5.py` | `tables/tableS5.csv` |

## 2. Rerun the analysis

The raw run outputs and board result files are archived at Zenodo,
DOI [10.5281/zenodo.24182700](https://doi.org/10.5281/zenodo.24182700): one zip per experiment folder, the matching
instances, the FPGA bitstreams and a snapshot of this code. The README of the record maps each folder to its analysis
command. Rerunning the analysis reproduces the means exactly and the bootstrap intervals to within resampling noise.

## 3. Rerun the experiments

Each experiment folder lists, per panel, the run, its task table and seeds, the analysis script and the CSV it writes,
and the compute used:

- [`experiments/graph_matching/`][gm]: Fig. 3, 4, S1, S2, S4, Tables S2-S4
- [`experiments/surface_code/`][sc]: Fig. 5, S7, S8, Table S5
- [`experiments/event_driven/`][ed]: Fig. 6, S3, S9

Runs write one compressed shard per task and skip finished shards, so an interrupted job resumes when the same command
is run again.

[gm]: https://github.com/aimlab-wustl/Nimbus/tree/main/experiments/graph_matching
[sc]: https://github.com/aimlab-wustl/Nimbus/tree/main/experiments/surface_code
[ed]: https://github.com/aimlab-wustl/Nimbus/tree/main/experiments/event_driven
