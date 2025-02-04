# PArtial Wasserstein on the Line (PAWL)

This repository contains code for the PAWL method presented in the paper 
"One for all and all for one: Efficient computation of partial Wasserstein distances on the line".

If you wish to cite this paper, use the following Bibtex entry:

```bibtex
@inproceedings{
  chapel2025one,
  title={One for all and all for one: Efficient computation of partial Wasserstein distances on the line},
  author={Chapel, Laetitia and Tavenard, Romain},
  booktitle={ICLR},
  year={2025},
  url={https://openreview.net/forum?id=kzEPsHbJDv}
}
```

The main file is `partial.py`, which contains the following two functions:
* `partial_ot_1d(x, y, max_iter, p=1)` where `x` and `y` are numpy arrays of 
  shape (n, ) and (m, ) and `max_iter` is the number of pairs one wants to 
  include in the solution. Note however that all partial solutions of size
  lower than `max_iter` can be retrieved from the output of this function
  (see docs of the function for more details)
* `partial_ot_1d_elbow(x, y, return_all_solutions=False, p=1)` where `x` and `y` are numpy arrays of 
  shape (n, ) and (m, ). In this alternative implementation, the number of 
  pairs to be included in the solution is inferred using the elbow method on
  the series of costs of partial solutions (see docs of the function for 
  more details).

If your goal is to reproduce the results contained in the paper, you can:
* reproduce Fig. 4 (timings) by running `timings_partial.py`
* reproduce Fig. 5 (Gradient Flows) by running `Figure5 - partialGF bimodal distributions.ipynb`
* reproduce Table 1 and Fig. 6 (point cloud registration) by
    1. running `run_baselines_icp.py`
    2. gathering results using `visus_shape.py` and `tab_results_icp.py`
* reproduce Fig. 7 (Unbalanced Domain Adaptation) by running `Figure 7.ipynb`

Note that code for the baselines is located in subfolder `baselines/` and obtained from the following github repositories:
* https://github.com/thibsej/fast_uot for Fast-UOT
* https://github.com/yikun-baio/sliced_opt for SOPT and SPOT baselines and, more generally, code related to the point cloud registration experiment

Finally, `visu_elbow.py` gives a simple visualization of the effect of selecting the mass to be transferred using the elbow method.
