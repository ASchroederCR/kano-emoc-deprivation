# Kano EmOC Access Deprivation

Analysis of Emergency Obstetric Care (EmOC) access deprivation in the Kano Functional Urban Area, Nigeria, combining IDEAMAPS E2SFCA model outputs with the GRID3 NGA Health Facilities dataset and the Olubodun/Macharia et al. (2023) verified comprehensive-EmOC masterlist.

- **[Interactive map](index.html)** — togglable layers for the deprivation grid, community-validation focus cells, WorldPop women 15-49 population density, GRID3 hospitals, and Macharia-verified facilities. Built by [build_interactive_map.py](build_interactive_map.py).
- **[kano_emoc_deprivation.qmd](kano_emoc_deprivation.qmd)** — the Quarto source report (R).
- **[kano_emoc_deprivation.html](kano_emoc_deprivation.html)** — the rendered report.

## Data sources

- [IDEAMAPS EmOC model outputs](https://github.com/urbanbigdatacentre/ideamaps-models)
- [GRID3 NGA Health Facilities v2.0](https://data.grid3.org/datasets/a0ed9627a8b240ff8b315a84575754a4_0)
- [Olubodun, Macharia et al. (2023)](https://doi.org/10.6084/m9.figshare.22689667.v2)
- [WorldPop age and sex structures, Nigeria 2020 (constrained)](https://hub.worldpop.org/geodata/summary?id=50256) — women 15-49 population layer, built by [fetch_worldpop_layer.py](fetch_worldpop_layer.py)
