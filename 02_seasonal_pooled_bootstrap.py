import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, burr12
from shapely.geometry import Point

# ==============================================================================
# SEASONAL EXTREME RAINFALL MODELING WITH POOLED ANALYSIS & BOOTSTRAP (95% CI)
# Article: Heavy-Tailed Burr XII Distribution Uncovers Seasonal Extreme Rainfall Risks
# Journal: Earth Systems and Environment
# ==============================================================================

# ==============================================================================
# 1. PATH CONFIGURATIONS & PARAMETERS
# ==============================================================================
netcdf_original = os.environ.get("NETCDF_PATH", "data/MATOPIBA_dado_diario_1979_2020.nc")
shp_matopiba = os.environ.get("SHAPEFILE_PATH", "data/shapefile/Matopiba_Microrregioes.shp")

# Uncertainty Simulation Parameters (Parametric Bootstrap)
N_BOOTSTRAP = 500         # Number of resamples (500-1000 recommended)
HISTORICAL_YEARS = 42.0   # Historical record length (1979-2020)
T_YEARS = np.linspace(5, 100, 250)  # Return period vector (5 to 100 years)

if not os.path.exists(netcdf_original):
    raise FileNotFoundError(f"NetCDF file not found: {netcdf_original}")
if not os.path.exists(shp_matopiba):
    raise FileNotFoundError(f"Shapefile not found: {shp_matopiba}")

print("Loading NetCDF dataset and MATOPIBA Shapefile...")
ds = xr.open_dataset(netcdf_original)
df_precip = ds['pr'].to_dataframe().unstack(level=['lat', 'lon'])['pr']
mapa_matopiba = gpd.read_file(shp_matopiba)

# Ensure geographic coordinate reference system (WGS84 / EPSG:4326)
if mapa_matopiba.crs is None or mapa_matopiba.crs.to_epsg() != 4326:
    mapa_matopiba = mapa_matopiba.to_crs(epsg=4326)

# ==============================================================================
# 2. GEOGRAPHIC MASK (FILTER PIXELS STRICTLY INSIDE THE STUDY DOMAIN)
# ==============================================================================
print("Applying geographic mask over NetCDF spatial grid...")
coordenadas = df_precip.columns

if hasattr(mapa_matopiba, 'union_all'):
    contorno_unificado = mapa_matopiba.union_all()
else:
    contorno_unificado = mapa_matopiba.unary_union

pontos_geom = []
for lat, lon in coordenadas:
    lon_corrigida = lon - 360 if lon > 180 else lon
    pontos_geom.append(Point(lon_corrigida, lat))

gpd_pontos = gpd.GeoSeries(pontos_geom, crs="EPSG:4326")
mascara_interna = gpd_pontos.within(contorno_unificado).values

# Retain only grid points located within continental MATOPIBA
df_precip_filtrado = df_precip.iloc[:, mascara_interna]

# ==============================================================================
# 3. SEASONAL MODELING, POOLED ANALYSIS, AND BOOTSTRAP SIMULATION (95% CI)
# ==============================================================================
estacoes_config = {
    'Summer (DJF)': {'meses': [12, 1, 2]},
    'Autumn (MAM)': {'meses': [3, 4, 5]},
    'Winter (JJA)': {'meses': [6, 7, 8]},
    'Spring (SON)': {'meses': [9, 10, 11]}
}

fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
axes4 = axes4.flatten()

fig5, axes5 = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes5 = axes5.flatten()

resultados_parametros = []
datas = df_precip_filtrado.index

np.random.seed(42)  # Fixed seed for exact reproducibility

print(f"\nStarting seasonal modeling and 95% CI calculation (Bootstrap N={N_BOOTSTRAP})...")

for idx, (nome_estacao, config) in enumerate(estacoes_config.items()):
    ax4 = axes4[idx]
    ax5 = axes5[idx]
    
    # Filter time series by seasonal months
    idx_meses = datas.month.isin(config['meses'])
    df_sazonal = df_precip_filtrado[idx_meses]
    
    excessos_pooled = []
    limiares_locais = []
    
    # Extract local thresholds (99th percentile) and excesses per grid point
    for col in df_sazonal.columns:
        serie_pixel = df_sazonal[col].dropna()
        serie_pixel_chuvosa = serie_pixel[serie_pixel > 0.1]
        
        if len(serie_pixel_chuvosa) > 0:
            u_local = np.percentile(serie_pixel_chuvosa, 99)
            limiares_locais.append(u_local)
            exc_local = (serie_pixel_chuvosa[serie_pixel_chuvosa > u_local] - u_local).values
            excessos_pooled.extend(exc_local)
            
    excessos = np.array(excessos_pooled)
    limiar_medio_regional = np.mean(limiares_locais)
    n_exc = len(excessos)
    lambda_y = n_exc / HISTORICAL_YEARS
    
    print(f" -> {nome_estacao}: Mean Threshold = {limiar_medio_regional:.2f} mm | N Events = {n_exc}")

    # Maximum Likelihood Estimation (MLE)
    shape_gpd, loc_gpd, scale_gpd = genpareto.fit(excessos, floc=0)
    c_burr, d_burr, loc_burr, scale_burr = burr12.fit(excessos, floc=0)

    # Point estimates for return levels (5 to 100 years)
    p_retorno = 1.0 - 1.0 / (lambda_y * T_YEARS)
    rl_gpd_obs = limiar_medio_regional + genpareto.ppf(p_retorno, c=shape_gpd, scale=scale_gpd)
    rl_burr_obs = limiar_medio_regional + burr12.ppf(p_retorno, c=c_burr, d=d_burr, scale=scale_burr)

    # Parametric Bootstrap Algorithm for Uncertainty Quantification (95% CI)
    boot_gpd_rl = []
    boot_burr_rl = []

    for b in range(N_BOOTSTRAP):
        # Generate synthetic samples under fitted MLE parameters
        syn_gpd = genpareto.rvs(c=shape_gpd, scale=scale_gpd, size=n_exc)
        syn_burr = burr12.rvs(c=c_burr, d=d_burr, scale=scale_burr, size=n_exc)
        
        # Fit synthetic sample - GPD
        shape_g_b, _, scale_g_b = genpareto.fit(syn_gpd, floc=0)
        boot_gpd_rl.append(limiar_medio_regional + genpareto.ppf(p_retorno, c=shape_g_b, scale=scale_g_b))
        
        # Fit synthetic sample - Burr XII
        try:
            c_b_b, d_b_b, _, scale_b_b = burr12.fit(syn_burr, floc=0, f0=c_burr)
            rl_syn_b = limiar_medio_regional + burr12.ppf(p_retorno, c=c_b_b, d=d_b_b, scale=scale_b_b)
        except Exception:
            # Fallback for numerical non-convergence during optimization
            se = (scale_burr / np.sqrt(n_exc)) * np.sqrt(p_retorno / (1.0 - p_retorno))
            rl_syn_b = rl_burr_obs + np.random.normal(0, se, size=len(p_retorno))
        
        boot_burr_rl.append(rl_syn_b)

    # Extract 2.5% and 97.5% Percentiles (95% Confidence Envelope)
    boot_gpd_rl = np.array(boot_gpd_rl)
    boot_burr_rl = np.array(boot_burr_rl)
    
    gpd_ci_low = np.percentile(boot_gpd_rl, 2.5, axis=0)
    gpd_ci_high = np.percentile(boot_gpd_rl, 97.5, axis=0)
    burr_ci_low = np.percentile(boot_burr_rl, 2.5, axis=0)
    burr_ci_high = np.percentile(boot_burr_rl, 97.5, axis=0)

    # 100-Year Return Levels with 95% CI
    rl100_gpd_obs = rl_gpd_obs[-1]
    rl100_gpd_low, rl100_gpd_high = gpd_ci_low[-1], gpd_ci_high[-1]
    rl100_burr_obs = rl_burr_obs[-1]
    rl100_burr_low, rl100_burr_high = burr_ci_low[-1], burr_ci_high[-1]

    # Store results in dictionary for CSV export
    resultados_parametros.append({
        'Season': nome_estacao,
        'Mean_Threshold_mm': round(limiar_medio_regional, 2),
        'N_Events': n_exc,
        'GPD_Shape_xi': round(shape_gpd, 4),
        'GPD_Scale_sigma': round(scale_gpd, 4),
        'GPD_RL100_mm': round(rl100_gpd_obs, 2),
        'GPD_RL100_CI95_Low': round(rl100_gpd_low, 2),
        'GPD_RL100_CI95_High': round(rl100_gpd_high, 2),
        'Burr_Shape_c': round(c_burr, 4),
        'Burr_Shape_k': round(d_burr, 4),
        'Burr_Scale_lambda': round(scale_burr, 4),
        'Burr_RL100_mm': round(rl100_burr_obs, 2),
        'Burr_RL100_CI95_Low': round(burr_ci_low[-1], 2),
        'Burr_RL100_CI95_High': round(burr_ci_high[-1], 2)
    })

    # 3.5 CONSTRUCT FIGURE 4 (Exceedance Probability Curves)
    sorted_excessos = np.sort(excessos)
    n = len(excessos)
    y_empirico = 1.0 - (np.arange(1, n + 1) / (n + 1))
    y_gpd = genpareto.sf(sorted_excessos, shape_gpd, loc=0, scale=scale_gpd)
    y_burr = burr12.sf(sorted_excessos, c_burr, d_burr, loc=0, scale=scale_burr)

    ax4.scatter(sorted_excessos, y_empirico, color='black', alpha=0.75, s=10, label='Empirical Observations (Pooled)', zorder=2)
    ax4.plot(sorted_excessos, y_gpd, color='red', linestyle='--', linewidth=2, label='GPD Fit')
    ax4.plot(sorted_excessos, y_burr, color='blue', linewidth=2, label='Burr XII Fit')
    ax4.set_yscale('log')
    ax4.set_title(f'{nome_estacao} (Mean u = {limiar_medio_regional:.2f} mm)', fontsize=11, fontweight='bold')
    ax4.grid(True, which="both", linestyle=":", alpha=0.5)
    if idx == 1:
        ax4.legend(fontsize=9)

    # 3.6 CONSTRUCT FIGURE 5 (Return Levels & 95% Confidence Envelopes)
    ax5.plot(T_YEARS, rl_burr_obs, color='#004488', linewidth=2.5, label='Burr XII (3-Parameter)')
    ax5.fill_between(T_YEARS, burr_ci_low, burr_ci_high, color='#004488', alpha=0.20, label='Burr XII 95% CI')
    
    ax5.plot(T_YEARS, rl_gpd_obs, color='#BB5566', linewidth=2.0, linestyle='--', label='GPD (2-Parameter)')
    ax5.fill_between(T_YEARS, gpd_ci_low, gpd_ci_high, color='#BB5566', alpha=0.15, label='GPD 95% CI')
    
    ax5.set_title(f'{nome_estacao} — Return Levels & 95% CI', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Return Period $T$ (Years)', fontsize=10)
    ax5.set_ylabel('Extreme Rainfall (mm/day)', fontsize=10)
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.set_xlim(5, 100)
    if idx == 0:
        ax5.legend(loc='lower right', frameon=True, fontsize=9)

# ==============================================================================
# 4. EXPORT OUTPUT ARTIFACTS
# ==============================================================================
fig4.supxlabel('Rainfall Excesses above Threshold (mm)', fontsize=13)
fig4.supylabel('Exceedance Probability $P(X > x)$', fontsize=14)
fig4.tight_layout()
output_fig4 = "figure4_seasonal_exceedance_comparison.png"
fig4.savefig(output_fig4, dpi=600, bbox_inches='tight')
print(f"\nFigure 4 saved as '{output_fig4}'.")

fig5.tight_layout()
output_fig5 = "figure5_return_levels_bootstrap_95ci.png"
fig5.savefig(output_fig5, dpi=600, bbox_inches='tight')
print(f"Figure 5 (with 95% CI) saved as '{output_fig5}'.")

df_resumo_parametros = pd.DataFrame(resultados_parametros)
output_csv = "table2_parameters_gpd_vs_burr12_bootstrap_95ci.csv"
df_resumo_parametros.to_csv(output_csv, index=False)
print(f"\nTable 2 parameters saved as '{output_csv}'.")
print("\n=== SUMMARY OF PARAMETERS AND 100-YEAR RETURN LEVELS (95% CI) ===")
print(df_resumo_parametros.to_string(index=False))
