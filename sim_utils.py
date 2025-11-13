
import re
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


# ----------------------
# Paso 2 — Limpieza
# ----------------------
def reconstruct_matches_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DF a nivel partido con columnas:
      ['date','home','away','gh','ga','result','season','round']
    Acepta:
      (A) vista equipo (team/opponent + venue + gf/ga)  -> fusiona home/away
      (B) nivel partido (team/opponent + gf/ga)         -> reetiqueta a home/away
      (C) nivel partido (home/away + gh/ga)             -> normaliza
    """
    import numpy as np
    out_cols = ["date","home","away","gh","ga","result","season","round"]

    x = df.copy()
    # ---- parse fecha y round ----
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
    if "round" in x.columns:
        x["round"] = (x["round"].astype(str).str.extract(r"(\d+)").squeeze())
        x["round"] = pd.to_numeric(x["round"], errors="coerce").astype("Int64")
    if "season" in x.columns:
        x["season"] = x["season"].astype(str).str.strip()

    # ---- casos ya a nivel partido (C) home/away + gh/ga ----
    if {"home","away","gh","ga"}.issubset(x.columns):
        y = x[["date","home","away","gh","ga","season","round"]].copy()
        y["result"] = np.where(y["gh"]>y["ga"], "H", np.where(y["gh"]<y["ga"], "A","D"))
        y = y.dropna(subset=["home","away","gh","ga"]).sort_values(["date","home","away"]).reset_index(drop=True)
        return y[out_cols]

    # ---- casos nivel partido (B) con team/opponent + gf/ga (sin venue) ----
    if {"team","opponent","gf","ga"}.issubset(x.columns) and "venue" not in x.columns:
        y = x.rename(columns={"team":"home","opponent":"away","gf":"gh","ga":"ga"})[
            ["date","home","away","gh","ga","season","round"]
        ].copy()
        y["result"] = np.where(y["gh"]>y["ga"], "H", np.where(y["gh"]<y["ga"], "A","D"))
        y = y.dropna(subset=["home","away","gh","ga"]).sort_values(["date","home","away"]).reset_index(drop=True)
        return y[out_cols]

    # ---- caso (A): vista equipo (team/opponent + venue + gf/ga) -> fusionar ----
    # normalizo venue
    if "venue" in x.columns:
        x["venue"] = x["venue"].astype(str).str.lower().str.strip()

    # clave de partido (fecha + set ordenado de equipos)
    def mk(row):
        a, b = sorted([str(row.get("team","")).strip(), str(row.get("opponent","")).strip()])
        d = row.get("date")
        return f"{d}_{a}__{b}"
    x["match_key"] = x.apply(mk, axis=1)

    home_rows = x[x.get("venue","").eq("home")].copy()
    away_rows = x[x.get("venue","").eq("away")].copy()

    home_keep = home_rows[["match_key","date","season","round","team","opponent","gf","ga"]].rename(columns={
        "team":"home","opponent":"away","gf":"gh","ga":"ga_h_view"
    })
    away_keep = away_rows[["match_key","team","opponent","gf","ga"]].rename(columns={
        "team":"away_check","opponent":"home_check","gf":"ga_from_away_view","ga":"gh_from_away_view"
    })

    merged = pd.merge(home_keep, away_keep, on="match_key", how="outer")

    gh = merged["gh"].fillna(merged["gh_from_away_view"])
    ga = merged["ga_h_view"].fillna(merged["ga_from_away_view"])
    home = merged["home"].fillna(merged["home_check"])
    away = merged["away"].fillna(merged["away_check"])

    result = np.where(gh > ga, "H", np.where(gh < ga, "A", "D"))

    clean = pd.DataFrame({
        "date": merged["date"],
        "home": home,
        "away": away,
        "gh": pd.to_numeric(gh, errors="coerce").astype("Int64"),
        "ga": pd.to_numeric(ga, errors="coerce").astype("Int64"),
        "result": result,
        "season": merged.get("season", pd.Series(index=merged.index, dtype="object")).astype(str).str.strip(),
        "round": merged.get("round",  pd.Series(index=merged.index, dtype="Int64"))
    })
    clean = (clean.dropna(subset=["home","away","gh","ga"])
                  .drop_duplicates(subset=["date","home","away","gh","ga"])
                  .sort_values(["date","home","away"]).reset_index(drop=True))
    return clean[out_cols]

# ----------------------
# Paso 3 — Partición
# ----------------------

def _season_sort_key(s: str):
    import re
    if s is None or (isinstance(s,float) and pd.isna(s)):
        return (-1, -1)
    s = str(s).strip()
    m = re.search(r"(\d{4})", s)
    if m:
        y1 = int(m.group(1))
    else:
        m2 = re.search(r"(\d{2})[^\d]?(\d{2})", s)
        y1 = 2000 + int(m2.group(1)) if m2 else -1
    return (y1, len(s))

def split_est_val(df_clean: pd.DataFrame):
    seasons = sorted(df_clean["season"].dropna().astype(str).unique(), key=_season_sort_key)
    val_season = seasons[-1] if seasons else None
    df_est = df_clean[df_clean["season"].astype(str) != val_season].copy()
    df_val = df_clean[df_clean["season"].astype(str) == val_season].copy()
    return df_est, df_val, val_season

# ----------------------
# Paso 4–5 — Parámetros
# ----------------------
def empirical_joint_and_cdf(sub: pd.DataFrame) -> pd.DataFrame:
    """Frecuencias y CDF de (gh,ga) en orden lexicográfico."""
    tbl = (sub.groupby(["gh","ga"], as_index=False).size().rename(columns={"size":"count"})
           .sort_values(["gh","ga"]).reset_index(drop=True))
    total = tbl["count"].sum()
    tbl["prob"] = tbl["count"] / total
    tbl["cdf"] = tbl["prob"].cumsum()
    return tbl

def empirical_global(df_est: pd.DataFrame) -> pd.DataFrame:
    return empirical_joint_and_cdf(df_est)

def empirical_by_team(df_est: pd.DataFrame, min_n_home:int=35, min_n_away:int=35) -> pd.DataFrame:
    out = []
    teams = pd.unique(pd.concat([df_est["home"], df_est["away"]]))
    for t in teams:
        sub_h = df_est[df_est["home"] == t]
        if len(sub_h) >= min_n_home:
            tbl_h = empirical_joint_and_cdf(sub_h).copy()
            tbl_h.insert(0,"team",t); tbl_h.insert(1,"condition","home")
            out.append(tbl_h)
        sub_a = df_est[df_est["away"] == t]
        if len(sub_a) >= min_n_away:
            tbl_a = empirical_joint_and_cdf(sub_a).copy()
            tbl_a.insert(0,"team",t); tbl_a.insert(1,"condition","away")
            out.append(tbl_a)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def poisson_lambdas(df_est: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(pd.unique(pd.concat([df_est["home"], df_est["away"]]).dropna().astype(str)))
    rows = []
    for team in teams:
        sub_home = df_est[df_est["home"] == team]
        sub_away = df_est[df_est["away"] == team]
        lam_home = sub_home["gh"].mean() if len(sub_home) else np.nan
        lam_away = sub_away["ga"].mean() if len(sub_away) else np.nan
        rows.append({
            "team": team,
            "lambda_home_for": lam_home, "n_home": len(sub_home),
            "lambda_away_for": lam_away, "n_away": len(sub_away),
        })
    return pd.DataFrame(rows)

# ----------------------
# Paso 6 — Configuración RNG
# ----------------------
def set_seed(seed: int = 42):
    """Fija semilla para reproducibilidad."""
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

# ----------------------
# Paso 7 — Motores de simulación
# ----------------------
def _sample_from_empirical_cdf(cdf_tbl: pd.DataFrame, u: float):
    """
    Dada una tabla con columnas (gh, ga, cdf) y un U~U(0,1), devuelve (gh, ga)
    siguiendo Transformada Inversa (discreta).
    Asume cdf ordenada ascendente y en [0,1].
    """
    # búsqueda lineal simple (suficiente para tablas pequeñas de goles)
    idx = np.searchsorted(cdf_tbl["cdf"].values, u, side="right")
    if idx >= len(cdf_tbl):
        idx = len(cdf_tbl) - 1
    row = cdf_tbl.iloc[idx]
    return int(row["gh"]), int(row["ga"])

def _poisson_sample_lam(lam: float):
    """Genera un conteo Poisson(lam) por inversa discreta (Knuth)."""
    # Método de Knuth (generación de Poisson)
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= np.random.rand()
    return k - 1

def simulate_season_empirical(df_fixture: pd.DataFrame,
                              emp_global: pd.DataFrame,
                              emp_by_team: pd.DataFrame | None,
                              R: int = 1000,
                              seed: int = 123) -> pd.DataFrame:
    """
    Simula una temporada completa usando muestreo empírico de (gh,ga).
    df_fixture: partidos de la temporada objetivo (columns: home, away)
    emp_global: tabla (gh,ga,prob,cdf) global
    emp_by_team: tabla por equipo/condición con (team, condition, gh, ga, prob, cdf) o None
    Devuelve una tabla larga con puntos por equipo y réplica: columns = [replica, team, points]
    """
    set_seed(seed)

    # Indexar CDFs para acceso rápido
    emp_global = emp_global[["gh","ga","cdf"]].sort_values(["gh","ga"]).reset_index(drop=True)
    team_cdfs = {}
    if emp_by_team is not None and len(emp_by_team):
        for (t, cond), sub in emp_by_team.groupby(["team","condition"]):
            team_cdfs[(t,cond)] = sub[["gh","ga","cdf"]].sort_values(["gh","ga"]).reset_index(drop=True)

    teams = sorted(pd.unique(pd.concat([df_fixture["home"], df_fixture["away"]]).dropna().astype(str)))
    out = []

    for r in range(1, R+1):
        # puntos acumulados de esta réplica
        pts = {t: 0 for t in teams}
        for _, row in df_fixture.iterrows():
            h, a = str(row["home"]), str(row["away"])
            # Seleccionar CDF por equipo si existe suficiente N, si no global
            cdf_h = team_cdfs.get((h,"home"), emp_global)
            u = np.random.rand()
            gh, ga = _sample_from_empirical_cdf(cdf_h, u)
            # NOTA: Para mantener simetría simple, usamos la misma CDF; en variantes más ricas
            # puede usarse cdf específica del visitante. Nos ceñimos al apunte (discreta conjunta).
            # Asignar puntos
            if gh > ga:
                pts[h] += 3
            elif gh < ga:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1
        for t in teams:
            out.append({"replica": r, "team": t, "points": pts[t]})
    return pd.DataFrame(out)

def simulate_season_poisson(df_fixture: pd.DataFrame,
                            lambdas: pd.DataFrame,
                            R: int = 1000,
                            seed: int = 123) -> pd.DataFrame:
    """
    Simula una temporada completa usando Poisson por condición:
      gh ~ Poisson(lambda_home_for(home))
      ga ~ Poisson(lambda_away_for(away))
    Devuelve tabla larga [replica, team, points].
    """
    set_seed(seed)
    lam_h = {row["team"]: float(row["lambda_home_for"]) for _, row in lambdas.iterrows()}
    lam_a = {row["team"]: float(row["lambda_away_for"]) for _, row in lambdas.iterrows()}
    teams = sorted(pd.unique(pd.concat([df_fixture["home"], df_fixture["away"]]).dropna().astype(str)))
    out = []

    for r in range(1, R+1):
        pts = {t: 0 for t in teams}
        for _, row in df_fixture.iterrows():
            h, a = str(row["home"]), str(row["away"])
            lam_gh = lam_h.get(h, np.nan)
            lam_ga = lam_a.get(a, np.nan)
            # Si falta lambda para algún equipo, usar promedio global simple
            if np.isnan(lam_gh): lam_gh = np.nanmean(list(lam_h.values()))
            if np.isnan(lam_ga): lam_ga = np.nanmean(list(lam_a.values()))
            gh = _poisson_sample_lam(max(lam_gh, 0.0001))
            ga = _poisson_sample_lam(max(lam_ga, 0.0001))
            if gh > ga:
                pts[h] += 3
            elif gh < ga:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1
        for t in teams:
            out.append({"replica": r, "team": t, "points": pts[t]})
    return pd.DataFrame(out)

def summarize_points(points_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por equipo: media, desvío, p50, p90, p95 de puntos simulados."""
    def pct(s, q): return float(np.percentile(s, q))
    g = points_df.groupby("team")["points"]
    out = pd.DataFrame({
        "mean": g.mean(),
        "std":  g.std(ddof=1),
        "p50":  g.apply(lambda s: pct(s, 10)),
        "p90":  g.apply(lambda s: pct(s, 50)),
        "p95":  g.apply(lambda s: pct(s, 90)),
        "n_rep": g.size() / g.count().groupby(level=0).first()  # aproximación n_rep
    })
    out = out.reset_index().sort_values("mean", ascending=False)
    return out




