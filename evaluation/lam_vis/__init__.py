"""Visualisations of a frozen CoMo action model (IDM + decoder, models/como.py) on Zelda: continuous action z only, no k-means.

  - actions.py   : the frozen model as two functions, infer z from a frame pair and apply z to a start frame
  - data.py      : the frame pairs (judge-labelled test pairs, held-out test windows, a sample of train pairs)
  - cycle.py     : action transfer and the cycle / leakage tests of Garrido et al. 2026 (arXiv 2601.05230, §6)
  - embedding.py : PCA of z and the cosine similarity of z between label groups
  - figures.py   : the PNGs
  - page.py      : the data of the interactive 3D PCA page (page.html)
"""
