# Heavy-Tailed Burr XII Distribution Uncovers Seasonal Extreme Rainfall Risks across MATOPIBA, Brazil

Official repository for data processing, Extreme Value Theory (EVT) modeling, parametric bootstrap uncertainty quantification, and spatial risk mapping as presented in the study published in *Earth Systems and Environment*.

## Overview

Extreme rainfall events pose severe hydro-meteorological risks to the MATOPIBA agricultural frontier (comprising parts of Maranhão, Tocantins, Piauí, and Bahia in Northeastern Brazil). This repository provides a fully reproducible physical-statistical modeling framework that:
1. Integrates multi-decadal daily precipitation datasets (CPC/NOAA 1979–2020) and IBGE microregional shapefiles.
2. Applies a Regional Pooled Analysis utilizing both the classical 2-parameter Generalized Pareto Distribution (GPD) and the heavy-tailed 3-parameter Burr XII Distribution.
3. Incorporates temporal de-clustering (3-day runs method) to ensure statistical independence of extreme excesses above the localized 99th percentile (P99).
4. Quantifies sampling uncertainty via Parametric Bootstrap resampling (N = 500), producing 95% confidence intervals for 100-year return levels (RL100).
5. Evaluates model fit using formal Goodness-of-Fit (GOF) metrics (AIC, BIC, Kolmogorov-Smirnov test).
6. Downscales regional EVT parameters to local grid scales, producing a 16-subplot spatial matrix (10, 20, 30, and 40-year return levels) and pinpointing top vulnerable microregions.

## Repository Structure

```text
matopiba-extreme-rainfall/
├── 01_spatial_descriptive_stats.py       # Table 1 & Spatial Hurst/Skewness maps
├── 02_seasonal_pooled_bootstrap.py       # Table 2, GPD vs Burr XII, Bootstrap 95% CI (Fig 4 & 5)
├── 03_declustering_gof_spatial_matrix.py # Declustering, GOF (AIC/BIC/KS) & 16-Subplot Matrix
├── data/
│   ├── MATOPIBA_dado_diario_1979_2020.nc # Daily precipitation NetCDF (1979-2020)
│   └── shapefile/
│       ├── Matopiba_Microrregioes.shp    # IBGE Microregions Shapefile (.shp, .shx, .dbf, .prj)
│       ├── Matopiba_Microrregioes.shx
│       ├── Matopiba_Microrregioes.dbf
│       └── Matopiba_Microrregioes.prj
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md                             # Repository documentation
```

## Data Availability & Zenodo Archive

The full daily precipitation NetCDF dataset (`MATOPIBA_dado_diario_1979_2020.nc`) and the IBGE microregional shapefiles are permanently archived on Zenodo:
- **Zenodo Repository DOI:** 10.5281/zenodo.XXXXXXX

To run the scripts, download the data package from Zenodo and extract the files into the `data/` folder following the directory structure above.

## Setup & Workflow

### 1. Installation
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

### 2. Sequential Execution

- **Step 1: Descriptive Statistics & Spatial Persistence**
  ```bash
  python 01_spatial_descriptive_stats.py
  ```
  Generates domain-wide statistics (Table 1) and spatial maps for Skewness and Hurst exponent (`figure_spatial_analysis_final.png`).

- **Step 2: Regional Pooled Analysis & Bootstrap Uncertainty (95% CI)**
  ```bash
  python 02_seasonal_pooled_bootstrap.py
  ```
  Fits GPD and Burr XII models via MLE, performs N=500 parametric bootstrap, and outputs Figure 4, Figure 5 (`figure5_return_levels_bootstrap_95ci.png`), and Table 2 CSV.

- **Step 3: De-clustering, GOF & Spatial Matrix**
  ```bash
  python 03_declustering_gof_spatial_matrix.py
  ```
  Applies 3-day runs declustering, evaluates AIC/BIC/KS tests, downscales parameters, and generates the 16-subplot spatial risk matrix (`figure_matrix_16_subplots_critical_BurrXII_Declustered.png`).

## Summary of Results (RL100)

- **Summer (DJF):** Best model = **Burr XII** | RL100 = 190.78 mm/day (Burr XII; 95% CI: 187.75–206.58) vs 212.99 mm/day (GPD)
- **Autumn (MAM):** Best model = **GPD**      | RL100 = 152.03 mm/day (GPD; 95% CI: 142.33–161.55) vs 168.46 mm/day (Burr XII)
- **Winter (JJA):** Best model = **Burr XII** | RL100 = 120.26 mm/day (Burr XII; 95% CI: 114.09–149.11) vs 158.54 mm/day (GPD)
- **Spring (SON):** Best model = **GPD**      | RL100 = 174.29 mm/day (GPD; 95% CI: 158.87–194.29) vs 171.85 mm/day (Burr XII)

## Citation
If you use this repository or dataset, please cite:
Earth Systems and Environment (2026). DOI: 10.1007/s41748-026-xxxxx-x

## License
MIT License
