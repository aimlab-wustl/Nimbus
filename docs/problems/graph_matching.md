---
extra_css:
  - assets/demo/matching.css
extra_js:
  - assets/demo/matching.js
---

# Graph matching

Graph matching assigns the L keypoints of one image to the L keypoints of another. NIMBUS solves the third-order
form: the objective rewards pairs of keypoint triangles with similar shapes, so each term links three correspondences.

| | |
|---|---|
| Spins | L x L assignment matrix, one spin per candidate correspondence |
| Constraint | permutation: exactly one selected spin in every row and every column |
| Objective | third-order polynomial over the assignment spins |
| Moves | exchange the targets of two rows, which keeps the assignment a permutation |
| Problem kind | `matching` ([Settings](../guides/settings.md#problem)) |

## Live demo

<div id="nimbus-matching-demo"></div>

The machine assigns every point of image 1 to one point of image 2 and maximizes the similarity of matched triangles.
It changes the assignment only by exchanging the targets of two points, a move that keeps every point matched exactly
once, so every state it visits is a valid matching. Solid lines are correct matches. Drag any point to change the
problem; the machine continues from its current matching. The demo runs the matching model of the package, with the
same moves, acceptance rule, temperature calibration and annealing schedule, in your browser.

## Move libraries

A move library is the set of row exchanges the machine can propose.

| Library | Exchanges |
|---|---|
| path | the backbone of the L - 1 exchanges of consecutive rows |
| complete | the backbone with all shortcuts, all L(L - 1)/2 exchanges |
| informed shortcuts | the backbone with K exchanges chosen from the objective tensor alone (`nimbus.problems.matching.informed_pairs`) |
| random shortcuts | the backbone with K exchanges chosen uniformly at random (`random_pairs`) |

## Run it

```bash
nimbus solve examples/matching_small.yaml --output local_runs/matching            # complete library, L = 8
nimbus solve examples/matching_small.yaml --set transitions.library=path           # path library
python examples/make_matching_instance.py --n 12 --seed 1 --out local_runs/matching_n12.npz   # a new instance
```

[`examples/paper_protocol.py`](https://github.com/aimlab-wustl/Nimbus/blob/main/examples/paper_protocol.py) runs the
protocol of the paper through the Python API: compile, temperature calibration, Fowler-Nordheim schedule, replicas,
success fraction and iterations to solution per budget, and the shortcut libraries.

## In the paper

The paper compares the machine with penalty-based neuromorphic Ising machines (NeuroSA and N-HOIM) and with three
hypergraph-matching solvers (BCAGM3, Adapt-BCAGM3 and RRWHM), on synthetic instances and on WILLOW-ObjectClass image
pairs, and studies how the move library sets the annealing time. The experiments, task tables, analysis scripts and
source data of Fig. 3, Fig. 4, Fig. S1, S2, S4 and Tables S2-S4 are in
[`experiments/graph_matching/`](https://github.com/aimlab-wustl/Nimbus/tree/main/experiments/graph_matching).
