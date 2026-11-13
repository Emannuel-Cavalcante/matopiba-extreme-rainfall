import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import genpareto, burr12, kstest
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# SEASONAL CONFIGURATION (NO HARDCODED PARAMETERS - FITTED DYNAMICALLY VIA MLE)
# ==============================================================================
ESTACOES_CONFIG = {
    'Summer (DJF)': {'meses': [12, 1, 2], 'modelo': 'burr12'},
    'Autumn (MAM)': {'meses': [3, 4, 5], 'modelo': 'gpd'},
    'Winter (JJA)': {'meses': [6, 7, 8], 'modelo': 'burr12'},
    'Spring (SON)': {'meses': [9, 10, 11], 'modelo': 'gpd'}
}

PERIODOS_RETORNO = [10, 20, 30, 40]


def declustear_por_runs(serie_sazonal, limiar, run_length=3):
    """
    Performs temporal de-clustering using the runs declustering method.
    Returns raw excesses and independent peak excesses.
    """
    serie_limpa = serie_sazonal.dropna()
    acima = serie_limpa > limiar
    if not acima.any():
        return np.array([]), np.array([])

    excessos_brutos = serie_limpa[acima].values - limiar

    indices = np.where(acima)[0]
    if len(indices) == 0:
        return excessos_brutos, np.array([])

    picos_excessos = []
    cluster_atual = [indices[0]]

    for idx in indices[1:]:
        if idx - cluster_atual[-1] <= run_length:
            cluster_atual.append(idx)
        else:
            val_max = serie_limpa.iloc[cluster_atual].max()
            picos_excessos.append(val_max - limiar)
            cluster_atual = [idx]

    if cluster_atual:
        val_max = serie_limpa.iloc[cluster_atual].max()
        picos_excessos.append(val_max - limiar)

    return excessos_brutos, np.array(picos_excessos)


def extrair_excessos_ponto(args):
    """
    Extracts local P99 threshold and declustered excesses for a single grid point (lat, lon).
    """
    col, serie_ponto, estacoes_config = args
    lat, lon = col[0], col[1]

    stats_amostra = {est: {'raw': 0, 'declustered': 0} for est in estacoes_config}
    excessos_ponto = {est: [] for est in estacoes_config}
    limiares_ponto = {est: np.nan for est in estacoes_config}
    dias_chuvosos_ponto = {est: 0 for est in estacoes_config}

    if serie_ponto.isna().all() or np.all(serie_ponto == 0):
        return lat, lon, stats_amostra, excessos_ponto, limiares_ponto, dias_chuvosos_ponto

    datas_ponto = serie_ponto.index

    for nome_estacao, config in estacoes_config.items():
        idx_meses = datas_ponto.month.isin(config['meses'])
        dados_sazonais = serie_ponto[idx_meses].dropna()

        dados_chuva = dados_sazonais[dados_sazonais > 0.1].values

        if len(dados_chuva) < 50:
            continue

        limiar_local = np.percentile(dados_chuva, 99)
        raw_exc, excessos_locais = declustear_por_runs(dados_sazonais, limiar_local, run_length=3)

        stats_amostra[nome_estacao]['raw'] = len(raw_exc)
        stats_amostra[nome_estacao]['declustered'] = len(excessos_locais)
        excessos_ponto[nome_estacao] = excessos_locais
        limiares_ponto[nome_estacao] = limiar_local
        dias_chuvosos_ponto[nome_estacao] = len(dados_chuva)

    return lat, lon, stats_amostra, excessos_ponto, limiares_ponto, dias_chuvosos_ponto


def calcular_gof_dinamico(excessos_array, gpd_params, burr_params):
    """Calculates AIC, BIC, and Kolmogorov-Smirnov test dynamically for both GPD and Burr XII."""
    n = len(excessos_array)
    if n == 0:
        return {}

    # GPD GOF
    log_lik_gpd = np.sum(genpareto.logpdf(excessos_array, gpd_params['c'], loc=0, scale=gpd_params['scale']))
    aic_gpd = 2 * 2 - 2 * log_lik_gpd
    bic_gpd = 2 * np.log(n) - 2 * log_lik_gpd
    ks_gpd = kstest(excessos_array, 'genpareto', args=(gpd_params['c'], 0, gpd_params['scale']))

    # Burr XII GOF
    log_lik_burr = np.sum(burr12.logpdf(excessos_array, burr_params['c'], burr_params['d'], loc=0, scale=burr_params['scale']))
    aic_burr = 2 * 3 - 2 * log_lik_burr
    bic_burr = np.log(n) * 3 - 2 * log_lik_burr
    ks_burr = kstest(excessos_array, 'burr12', args=(burr_params['c'], burr_params['d'], 0, burr_params['scale']))

    return {
        'N': n,
        'GPD': {'AIC': aic_gpd, 'BIC': bic_gpd, 'KS_pvalue': ks_gpd.pvalue},
        'BurrXII': {'AIC': aic_burr, 'BIC': bic_burr, 'KS_pvalue': ks_burr.pvalue}
    }


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    netcdf_original = os.environ.get("NETCDF_PATH", "data/MATOPIBA_dado_diario_1979_2020.nc")
    shp_matopiba = os.environ.get("SHAPEFILE_PATH", "data/shapefile/Matopiba_Microrregioes.shp")

    if not os.path.exists(netcdf_original):
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_original}")
    if not os.path.exists(shp_matopiba):
        raise FileNotFoundError(f"Shapefile not found: {shp_matopiba}")

    print("Loading MATOPIBA precipitation NetCDF and Shapefile...")
    ds = xr.open_dataset(netcdf_original)
    df_precip = ds['pr'].to_dataframe().unstack(level=['lat', 'lon'])['pr']
    mapa_matopiba = gpd.read_file(shp_matopiba)

    if mapa_matopiba.crs is None or mapa_matopiba.crs.to_epsg() != 4326:
        mapa_matopiba = mapa_matopiba.to_crs(epsg=4326)

    coordenadas = df_precip.columns

    totais_amostra = {est: {'raw': 0, 'declustered': 0} for est in ESTACOES_CONFIG}
    excessos_pooled_all = {est: [] for est in ESTACOES_CONFIG}

    print("Step 1: Extracting local P99 thresholds and declustered excesses across all grid points...")
    tarefas_extracao = [(col, df_precip[col], ESTACOES_CONFIG) for col in coordenadas]

    with ProcessPoolExecutor() as executor:
        resultados_extracao = list(executor.map(extrair_excessos_ponto, tarefas_extracao))

    # Store point metadata for second pass
    dados_pontos = []
    for lat, lon, stats_ponto, exc_ponto, limiares_ponto, dias_chuvosos_ponto in resultados_extracao:
        lon_corr = lon - 360 if lon > 180 else lon
        dados_pontos.append({
            'lat': lat,
            'lon': lon_corr,
            'stats': stats_ponto,
            'excessos': exc_ponto,
            'limiares': limiares_ponto,
            'dias_chuvosos': dias_chuvosos_ponto
        })

        for est in ESTACOES_CONFIG:
            totais_amostra[est]['raw'] += stats_ponto[est]['raw']
            totais_amostra[est]['declustered'] += stats_ponto[est]['declustered']
            excessos_pooled_all[est].extend(exc_ponto[est])

    # Dynamic MLE fitting on pooled declustered excesses for each season
    print("\nStep 2: Dynamic MLE parameter estimation on regional pooled declustered excesses...")
    fitted_params = {}

    for est in ESTACOES_CONFIG:
        exc_arr = np.array(excessos_pooled_all[est])
        exc_arr = exc_arr[~np.isnan(exc_arr)]

        # Fit GPD dynamically via MLE
        shape_gpd, loc_gpd, scale_gpd = genpareto.fit(exc_arr, floc=0)
        # Fit Burr XII dynamically via MLE
        c_burr, d_burr, loc_burr, scale_burr = burr12.fit(exc_arr, floc=0)

        fitted_params[est] = {
            'gpd': {'c': shape_gpd, 'loc': loc_gpd, 'scale': scale_gpd},
            'burr12': {'c': c_burr, 'd': d_burr, 'loc': loc_burr, 'scale': scale_burr}
        }

        print(f" -> {est} Fitted Parameters:")
        print(f"    GPD:     Shape (xi) = {shape_gpd:.4f}, Scale (sigma) = {scale_gpd:.4f}")
        print(f"    Burr XII: Shape (c) = {c_burr:.4f}, Shape (k) = {d_burr:.4f}, Scale (lambda) = {scale_burr:.4f}")

    print("\n==========================================================================")
    print("      DE-CLUSTERING & DYNAMIC GOF REPORT: RAW VS. INDEPENDENT EXCEEDANCES")
    print("==========================================================================")
    for est, val in totais_amostra.items():
        reducao = ((val['raw'] - val['declustered']) / val['raw']) * 100 if val['raw'] > 0 else 0
        exc_arr = np.array(excessos_pooled_all[est])
        exc_arr = exc_arr[~np.isnan(exc_arr)]

        gof_res = calcular_gof_dinamico(exc_arr, fitted_params[est]['gpd'], fitted_params[est]['burr12'])

        selected_model = ESTACOES_CONFIG[est]['modelo']
        gof_sel = gof_res['BurrXII'] if selected_model == 'burr12' else gof_res['GPD']

        print(
            f"  {est:<15} | Raw (N_raw): {val['raw']:6d} | Independent (N_decl): {val['declustered']:6d} | Reduction: {reducao:5.1f}%")
        print(
            f"                  | Selected Model ({selected_model.upper()}) GOF -> AIC: {gof_sel['AIC']:.1f} | BIC: {gof_sel['BIC']:.1f} | K-S p-value: {gof_sel['KS_pvalue']:.4e}")
    print("==========================================================================\n")

    # Step 3: Compute spatial return levels per point using dynamically fitted parameters
    print("Step 3: Calculating spatial return levels using dynamically fitted parameters...")
    lats_validas = []
    lons_validas = []
    niveis_espaciais = {est: {t: [] for t in PERIODOS_RETORNO} for est in ESTACOES_CONFIG}

    for item in dados_pontos:
        lats_validas.append(item['lat'])
        lons_validas.append(item['lon'])

        for est, config in ESTACOES_CONFIG.items():
            limiar_local = item['limiares'][est]
            excessos_locais = item['excessos'][est]
            n_dias_chuva = item['dias_chuvosos'][est]

            if np.isnan(limiar_local) or len(excessos_locais) == 0 or n_dias_chuva == 0:
                for t in PERIODOS_RETORNO:
                    niveis_espaciais[est][t].append(np.nan)
                continue

            n_dias_chuvosos_ano = n_dias_chuva / 42.0
            lambda_u = len(excessos_locais) / n_dias_chuva

            model_name = config['modelo']
            params = fitted_params[est][model_name]

            for t in PERIODOS_RETORNO:
                p_excedencia = 1.0 / (t * n_dias_chuvosos_ano)
                p_condicional = p_excedencia / lambda_u
                p_condicional = np.clip(p_condicional, 1e-6, 0.9999)

                try:
                    if model_name == 'gpd':
                        excesso_estimado = genpareto.ppf(1.0 - p_condicional, params['c'], loc=params['loc'], scale=params['scale'])
                    else:
                        excesso_estimado = burr12.ppf(1.0 - p_condicional, params['c'], params['d'], loc=params['loc'], scale=params['scale'])

                    chuva_total_estimada = limiar_local + excesso_estimado
                except Exception:
                    chuva_total_estimada = np.nan

                niveis_espaciais[est][t].append(chuva_total_estimada)

    lats_validas = np.array(lats_validas)
    lons_validas = np.array(lons_validas)

    # Step 4: CONSTRUCT 16-SUBPLOT SPATIAL MATRIX
    print("Step 4: Generating 16-subplot spatial matrix with critical microregion identification...")
    xmin, ymin, xmax, ymax = mapa_matopiba.total_bounds
    lon_grid = np.linspace(xmin - 0.05, xmax + 0.05, 300)
    lat_grid = np.linspace(ymin - 0.05, ymax + 0.05, 300)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    if hasattr(mapa_matopiba, 'union_all'):
        contorno_unificado = mapa_matopiba.union_all()
    else:
        contorno_unificado = mapa_matopiba.unary_union

    pontos_grade = [Point(x, y) for x, y in zip(lon_mesh.ravel(), lat_mesh.ravel())]
    gpd_pontos = gpd.GeoSeries(pontos_grade, crs="EPSG:4326")
    dentro_do_mapa = gpd_pontos.within(contorno_unificado).values.reshape(lon_mesh.shape)

    pontos_originais_geom = [Point(lon, lat) for lon, lat in zip(lons_validas, lats_validas)]
    gpd_pontos_originais = gpd.GeoSeries(pontos_originais_geom, crs="EPSG:4326")
    mascara_pontos_internos = gpd_pontos_originais.within(contorno_unificado).values

    valores_globais_internos = []
    for nome_estacao in ESTACOES_CONFIG:
        for t in PERIODOS_RETORNO:
            vals = np.array(niveis_espaciais[nome_estacao][t])[mascara_pontos_internos]
            valores_globais_internos.extend(vals[np.isfinite(vals)])

    vmin_dinamico = np.floor(np.min(valores_globais_internos))
    vmax_dinamico = np.ceil(np.max(valores_globais_internos))
    niveis_cores = np.linspace(vmin_dinamico, vmax_dinamico, 16)

    fig, axes = plt.subplots(4, 4, figsize=(18, 18), sharex=True, sharey=True)

    tam_titulo_subplot = 11
    tam_eixos_globais = 15
    tam_ticks = 12
    tam_cbar = 16

    print("\n==========================================================================")
    print("      VALIDATION REPORT: TOP 5 MICROREGIONS PER SUBPLOT")
    print("==========================================================================")

    for row_idx, (nome_estacao, config) in enumerate(ESTACOES_CONFIG.items()):
        nome_modelo = "Burr XII" if config['modelo'] == 'burr12' else "GPD"

        for col_idx, t in enumerate(PERIODOS_RETORNO):
            ax = axes[row_idx, col_idx]
            valores_ponto = np.array(niveis_espaciais[nome_estacao][t])

            valid_mask = np.isfinite(valores_ponto)
            grid_chuva = griddata(
                (lons_validas[valid_mask], lats_validas[valid_mask]),
                valores_ponto[valid_mask],
                (lon_mesh, lat_mesh),
                method='linear'
            )
            grid_chuva[~dentro_do_mapa] = np.nan

            contour = ax.contourf(lon_mesh, lat_mesh, grid_chuva, cmap='winter_r', levels=niveis_cores, extend='both')
            mapa_matopiba.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, alpha=0.8)

            mascara_critica = valid_mask & mascara_pontos_internos
            lons_filtradas = lons_validas[mascara_critica]
            lats_filtradas = lats_validas[mascara_critica]
            valores_filtrados = valores_ponto[mascara_critica]

            if len(valores_filtrados) >= 5:
                gdf_pts = gpd.GeoDataFrame(
                    {'val': valores_filtrados, 'lon': lons_filtradas, 'lat': lats_filtradas},
                    geometry=[Point(x, y) for x, y in zip(lons_filtradas, lats_filtradas)],
                    crs="EPSG:4326"
                )

                try:
                    gdf_joined = gpd.sjoin(gdf_pts, mapa_matopiba, how='inner', predicate='within')
                except TypeError:
                    gdf_joined = gpd.sjoin(gdf_pts, mapa_matopiba, how='inner', op='within')

                cols_possiveis = [c for c in mapa_matopiba.columns if
                                  any(k in c.upper() for k in ['MICRO', 'NOME', 'NM_'])]
                col_nome_micro = cols_possiveis[0] if len(cols_possiveis) > 0 else (
                    'index_right' if 'index_right' in gdf_joined.columns else gdf_joined.columns[0])

                idx_max_por_micro = gdf_joined.groupby(col_nome_micro)['val'].idxmax()
                top_pontos_por_micro = gdf_joined.loc[idx_max_por_micro]

                top5_microrregioes = top_pontos_por_micro.sort_values(by='val', ascending=False).head(5)

                print(f"\n--- Subplot: {nome_estacao} ({nome_modelo}) | T = {t} Years ---")
                for rank, (_, row_micro) in enumerate(top5_microrregioes.iterrows(), 1):
                    nome_m = str(
                        row_micro[col_nome_micro]) if col_nome_micro in row_micro else f"Microregion {row_micro.name}"
                    print(
                        f"  Star #{rank}: {nome_m:<30} | Peak: {row_micro['val']:6.2f} mm/day | Lat: {row_micro['lat']:6.2f}°S, Lon: {row_micro['lon']:6.2f}°W")
                print(
                    f"  [OK] Total unique microregions in selection: {len(top5_microrregioes[col_nome_micro].unique())} of 5")

                ax.scatter(top5_microrregioes['lon'], top5_microrregioes['lat'],
                           color='yellow', marker='*', s=150, edgecolor='black', linewidth=0.5,
                           label='Top 5 Microregions', zorder=5)

            ax.set_title(f'{nome_estacao} ({nome_modelo}) | T = {t} Years', fontsize=tam_titulo_subplot,
                         fontweight='bold')
            ax.set_xlim(xmin - 0.05, xmax + 0.05)
            ax.set_ylim(ymin - 0.05, ymax + 0.05)
            ax.tick_params(axis='both', labelsize=tam_ticks)
            ax.grid(True, linestyle=':', alpha=0.3)

    fig.supxlabel('Longitude (°W)', fontsize=tam_eixos_globais, fontweight='bold', y=0.015)
    fig.supylabel('Latitude (°S)', fontsize=tam_eixos_globais, fontweight='bold', x=0.015)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.05, left=0.05, right=0.90, hspace=0.07, wspace=0.001)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Estimated Extreme Rainfall Depth (mm/day)', fontsize=tam_cbar, fontweight='bold')
    cbar.ax.tick_params(labelsize=tam_ticks)

    output_img = "figure_matrix_16_subplots_critical_BurrXII_Declustered.png"
    plt.savefig(output_img, dpi=600, bbox_inches='tight')
    print(f"\n==========================================================================")
    print(f">>> SUCCESS! File saved as '{output_img}'")
    print(f"==========================================================================")
