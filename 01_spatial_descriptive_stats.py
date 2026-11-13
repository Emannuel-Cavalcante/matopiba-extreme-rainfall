import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Point

# ==============================================================================
# 1. PATH CONFIGURATIONS
# ==============================================================================
csv_data_path = os.environ.get("CSV_DATA_PATH", "data/tabela_suplementar_revisor2_brutos.csv")
shp_matopiba_path = os.environ.get("SHAPEFILE_PATH", "data/shapefile/Matopiba_Microrregioes.shp")
netcdf_original_path = os.environ.get("NETCDF_PATH", "data/MATOPIBA_dado_diario_1979_2020.nc")

if not os.path.exists(csv_data_path):
    raise FileNotFoundError(f"Raw CSV file not found: {csv_data_path}")
if not os.path.exists(shp_matopiba_path):
    raise FileNotFoundError(f"Shapefile not found: {shp_matopiba_path}")

print("Loading MATOPIBA Shapefile and raw statistical data...")
mapa_matopiba = gpd.read_file(shp_matopiba_path)

if mapa_matopiba.crs is None or mapa_matopiba.crs.to_epsg() != 4326:
    mapa_matopiba = mapa_matopiba.to_crs(epsg=4326)

if hasattr(mapa_matopiba, 'union_all'):
    contorno_unificado = mapa_matopiba.union_all()
else:
    contorno_unificado = mapa_matopiba.unary_union

# ==============================================================================
# 2. GEOGRAPHIC FILTERING OF CSV DATA (STRICTLY WITHIN SHAPEFILE)
# ==============================================================================
df_stats_bruto = pd.read_csv(csv_data_path)

# Ensure longitude is negative for WGS84 Western Hemisphere coordinates (-180 to 0)
lons_csv_corr = [-abs(x) if x > 0 else x for x in df_stats_bruto['Longitude_Graus_W']]
lats_csv = df_stats_bruto['Latitude'].values

# Create geometric points to filter raw supplementary data
pontos_csv = [Point(lon, lat) for lon, lat in zip(lons_csv_corr, lats_csv)]
gpd_pontos_csv = gpd.GeoSeries(pontos_csv, crs="EPSG:4326")
mascara_csv_interna = gpd_pontos_csv.within(contorno_unificado).values

# Strict subset inside continental MATOPIBA
df_stats = df_stats_bruto[mascara_csv_interna].copy()

# ==============================================================================
# 3. NETCDF FILTERING AND SPATIAL CROSS-CORRELATION COMPUTATION
# ==============================================================================
print("Loading NetCDF to extract spatial cross-correlation within boundary...")
ds = xr.open_dataset(netcdf_original_path)
df_precip = ds['pr'].to_dataframe().unstack(level=['lat', 'lon'])['pr']

coordenadas_nc = df_precip.columns
pontos_nc = []
for col in coordenadas_nc:
    lat_nc, lon_nc = col[0], col[1]
    lon_nc_corr = lon_nc - 360 if lon_nc > 180 else (lon_nc if lon_nc < 0 else -lon_nc)
    pontos_nc.append(Point(lon_nc_corr, lat_nc))

gpd_pontos_nc = gpd.GeoSeries(pontos_nc, crs="EPSG:4326")
mascara_nc_interna = gpd_pontos_nc.within(contorno_unificado).values

df_precip_filtrado = df_precip.iloc[:, mascara_nc_interna].dropna(axis=1, how='all')

print("Calculating spatial cross-correlation matrix...")
matriz_corr = df_precip_filtrado.corr()
valores_corr = matriz_corr.values[~np.eye(matriz_corr.shape[0], dtype=bool)]
valores_corr = valores_corr[~np.isnan(valores_corr)]

# ==============================================================================
# 4. PRINT AND SAVE SYNTHETIC SUMMARY TABLE (TABLE 1)
# ==============================================================================
tabela_resumo = {
    "Statistical Metric": [
        "Time Series Length (N, days)", "Missing Data (%)", 
        "Dry Days Ratio (%)", "Mean Rainfall (mm/day)",
        "Skewness Index (S)", "Kurtosis Index (K)", 
        "Lag-1 Autocorrelation", "Spatial Cross-Correlation (r)", "Hurst Parameter (H)"
    ],
    "Domain Minimum": [
        int(df_stats['N_Dias_Total'].min()), f"{df_stats['Falhas_Pct'].min():.2f}%",
        f"{df_stats['Dias_Sem_Chuva_Pct'].min():.2f}%", f"{df_stats['Media_mm'].min():.4f}",
        f"{df_stats['Assimetria_Skew'].min():.4f}", f"{df_stats['Curtose_Kurt'].min():.4f}",
        f"{df_stats['Autocorr_Lag1'].min():.4f}", f"{valores_corr.min():.4f}", f"{df_stats['Hurst_H'].min():.4f}"
    ],
    "Domain Maximum": [
        int(df_stats['N_Dias_Total'].max()), f"{df_stats['Falhas_Pct'].max():.2f}%",
        f"{df_stats['Dias_Sem_Chuva_Pct'].max():.2f}%", f"{df_stats['Media_mm'].max():.4f}",
        f"{df_stats['Assimetria_Skew'].max():.4f}", f"{df_stats['Curtose_Kurt'].max():.4f}",
        f"{df_stats['Autocorr_Lag1'].max():.4f}", f"{valores_corr.max():.4f}", f"{df_stats['Hurst_H'].max():.4f}"
    ],
    "Regional Domain Mean": [
        int(df_stats['N_Dias_Total'].mean()), f"{df_stats['Falhas_Pct'].mean():.2f}%",
        f"{df_stats['Dias_Sem_Chuva_Pct'].mean():.2f}%", f"{df_stats['Media_mm'].mean():.4f}",
        f"{df_stats['Assimetria_Skew'].mean():.4f}", f"{df_stats['Curtose_Kurt'].mean():.4f}",
        f"{df_stats['Autocorr_Lag1'].mean():.4f}", f"{valores_corr.mean():.4f}", f"{df_stats['Hurst_H'].mean():.4f}"
    ]
}

df_print = pd.DataFrame(tabela_resumo)
print("\n>>> TABLE 1 SUMMARY (STRICTLY WITHIN MATOPIBA BOUNDARY) <<<")
print(df_print.to_string(index=False))
df_print.to_csv("Table1_Statistical_Summary_Strict.csv", index=False)

# ==============================================================================
# 5. SPATIAL MAP GENERATION (SKEWNESS & HURST PERSISTENCE)
# ==============================================================================
print("\n=== GENERATING MASKED SPATIAL MAPS (ANTI-EDGE CLIPPING STRATEGY) ===")

lat_bruta = lats_csv
lon_bruta = np.array(lons_csv_corr)
skew_bruto = df_stats_bruto['Assimetria_Skew'].values
hurst_bruto = df_stats_bruto['Hurst_H'].values

xmin, ymin, xmax, ymax = mapa_matopiba.total_bounds
lon_grid = np.linspace(xmin - 0.05, xmax + 0.05, 400)
lat_grid = np.linspace(ymin - 0.05, ymax + 0.05, 400)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

grid_skew = griddata((lon_bruta, lat_bruta), skew_bruto, (lon_mesh, lat_mesh), method='linear')
grid_hurst = griddata((lon_bruta, lat_bruta), hurst_bruto, (lon_mesh, lat_mesh), method='linear')

pontos_grade = [Point(x, y) for x, y in zip(lon_mesh.ravel(), lat_mesh.ravel())]
gpd_pontos = gpd.GeoSeries(pontos_grade, crs="EPSG:4326")
dentro_do_mapa = gpd_pontos.within(contorno_unificado).values.reshape(lon_mesh.shape)

grid_skew[~dentro_do_mapa] = np.nan
grid_hurst[~dentro_do_mapa] = np.nan

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.subplots_adjust(wspace=-0.08) 

tam_titulo = 15
tam_eixo = 14
tam_ticks = 18
tam_cbar = 14

# Map A: Skewness
contour1 = ax1.contourf(lon_mesh, lat_mesh, grid_skew, cmap='YlOrRd', levels=20)
mapa_matopiba.plot(ax=ax1, color='none', edgecolor='black', linewidth=0.5, alpha=0.8, zorder=2)
cbar1 = fig.colorbar(contour1, ax=ax1, orientation='vertical', shrink=0.78, pad=0.02)
cbar1.set_label('Skewness Index ($S$)', fontsize=tam_cbar, fontweight='bold')
cbar1.ax.tick_params(labelsize=tam_ticks)

ax1.set_title('A) Spatial Distribution of Skewness', fontsize=tam_titulo, fontweight='bold')
ax1.set_xlim(xmin - 0.05, xmax + 0.05)
ax1.set_ylim(ymin - 0.05, ymax + 0.05)
ax1.set_xlabel('Longitude (°W)', fontsize=tam_eixo)
ax1.set_ylabel('Latitude (°S)', fontsize=tam_eixo)
ax1.tick_params(axis='both', labelsize=tam_ticks)
ax1.grid(True, linestyle=':', alpha=0.4)

# Map B: Hurst Parameter
contour2 = ax2.contourf(lon_mesh, lat_mesh, grid_hurst, cmap='viridis', levels=20)
mapa_matopiba.plot(ax=ax2, color='none', edgecolor='black', linewidth=0.5, alpha=0.8, zorder=2)
cbar2 = fig.colorbar(contour2, ax=ax2, orientation='vertical', shrink=0.78, pad=0.02)
cbar2.set_label('Hurst Coefficient ($H$)', fontsize=tam_cbar, fontweight='bold')
cbar2.ax.tick_params(labelsize=tam_ticks)

ax2.set_title('B) Long-Term Persistence (Hurst Parameter)', fontsize=tam_titulo, fontweight='bold')
ax2.set_xlim(xmin - 0.05, xmax + 0.05)
ax2.set_xlabel('Longitude (°W)', fontsize=tam_eixo)
ax2.tick_params(axis='both', labelsize=tam_ticks)
ax2.grid(True, linestyle=':', alpha=0.4)

output_img = "figure_spatial_analysis_final.png"
plt.savefig(output_img, dpi=600, bbox_inches='tight')
print(f"Spatial figure saved successfully as '{output_img}'.")
