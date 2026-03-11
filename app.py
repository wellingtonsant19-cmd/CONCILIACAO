import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import time

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Motor de Conciliação",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }

.header-box {
    background: linear-gradient(135deg, #0f2044 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 28px;
}
.stat-box {
    background: #0a1628;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stat-label { font-size: 10px; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.stat-value { font-size: 22px; font-weight: 800; color: #e2e8f0; font-family: 'IBM Plex Mono', monospace; }
.stat-value.green  { color: #4ade80; }
.stat-value.yellow { color: #f59e0b; }
.stat-value.red    { color: #f87171; }

.alert-retencao {
    background: #ef444410; border: 1px solid #ef444430;
    border-radius: 8px; padding: 14px 18px;
    color: #fca5a5; font-size: 13px; margin-top: 12px;
}
.alert-multi {
    background: #f59e0b10; border: 1px solid #f59e0b30;
    border-radius: 8px; padding: 12px 16px;
    color: #fcd34d; font-size: 13px; margin-bottom: 16px;
}
.alert-escopo {
    background: #8b5cf610; border: 1px solid #8b5cf630;
    border-radius: 8px; padding: 12px 16px;
    color: #c4b5fd; font-size: 13px; margin-bottom: 16px;
}
.badge {
    display: inline-block; border-radius: 4px;
    padding: 2px 8px; font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
}
.badge-green  { background: #22c55e20; color: #4ade80;  border: 1px solid #22c55e40; }
.badge-yellow { background: #f59e0b20; color: #fbbf24;  border: 1px solid #f59e0b40; }
.badge-red    { background: #ef444420; color: #f87171;  border: 1px solid #ef444440; }
.badge-blue   { background: #3b82f620; color: #60a5fa;  border: 1px solid #3b82f640; }
.badge-purple { background: #8b5cf620; color: #c4b5fd;  border: 1px solid #8b5cf640; }
.badge-gray   { background: #33415520; color: #94a3b8;  border: 1px solid #33415540; }

div[data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURAÇÕES DO ALGORITMO
# ============================================================

TOLERANCIA     = 0.01
MAX_NOTAS      = 8
TOP_RESULTADOS = 10

# ============================================================
# LEITURA E NORMALIZAÇÃO DA PLANILHA
# ============================================================

COLUNAS = {
    "nome":         ["NOME CLIENTE", "NOME"],
    "cnpj":         ["CNPJ"],
    "cidade":       ["CIDADE"],
    "uf":           ["UF"],
    "nf":           ["NF", "NR NFEM"],
    "num_doc":      ["DOC", "NUM DOC MATERA"],
    "rbase":        ["RBASE RAIZ", "RBASE"],
    "valor_titulo": ["BRUTO", "VLR TITULO", "VALOR TITULO"],
    "saldo":        ["SALDO", "VLR SALDO", "VALOR SALDO"],
    "ir":           ["IR", "VLR RETIDO"],
    "iss":          ["ISS", "ISS RETIDO"],
    "venc":         ["VENCIMENTO", "VENC", "DT VENCIMENTO"],
    "atraso":       ["ATRASO"],
}

def _get_col(df, candidates):
    cols_upper = {c.strip().upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in cols_upper:
            return cols_upper[cand.upper()]
    return None

@st.cache_data(show_spinner=False)
def carregar_planilha(file_bytes, file_name):
    import io
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    renomear = {}
    for destino, candidates in COLUNAS.items():
        col = _get_col(df, candidates)
        if col:
            renomear[col] = destino
    df.rename(columns=renomear, inplace=True)
    for col in ["valor_titulo", "saldo", "ir", "iss", "atraso"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2)
    if "cnpj" in df.columns:
        df["cnpj_limpo"] = df["cnpj"].astype(str).str.replace(r"\D", "", regex=True)
    df.index = range(len(df))
    return df

# ============================================================
# ALGORITMO — CAMADA 1: PRÉ-PROCESSAMENTO
# ============================================================

def preprocessar(valores, alvo):
    limite = round(alvo + TOLERANCIA, 2)
    filtrado = {k: v for k, v in valores.items() if 0 < round(v, 2) <= limite}
    return dict(sorted(filtrado.items(), key=lambda x: x[1], reverse=True))

# ============================================================
# ALGORITMO — CAMADA 2: GREEDY DESCENDING
# ============================================================

def greedy(valores, alvo, max_notas):
    alvo_r = round(alvo, 2)
    soma, combo = 0.0, []
    for chave, val in valores.items():
        if len(combo) >= max_notas: break
        if round(soma + val, 2) <= round(alvo_r + TOLERANCIA, 2):
            soma = round(soma + val, 2)
            combo.append(chave)
        if abs(soma - alvo_r) <= TOLERANCIA:
            return (round(abs(soma - alvo_r), 2), tuple(combo))
    return None

# ============================================================
# ALGORITMO — CAMADA 3: MEET-IN-THE-MIDDLE (adaptativo)
# ============================================================

def _somas_grupo(items, max_n, cap=None):
    """
    DP de somas com cap opcional de combinações por soma.
    cap=None  → exato (comportamento original, para n pequeno)
    cap=int   → bounded (limita memória, para n grande)
    """
    dp = {0: [()]}
    for chave, val in items:
        new_e = {}
        for sa, combos in dp.items():
            nv = sa + val
            novos = [c + (chave,) for c in combos if len(c) < max_n]
            if novos:
                if nv not in new_e: new_e[nv] = []
                new_e[nv].extend(novos)
        for nv, novos in new_e.items():
            if nv not in dp:
                dp[nv] = novos[:cap] if cap else novos
            else:
                merged = dp[nv] + novos
                if cap and len(merged) > cap:
                    merged.sort(key=len)
                    merged = merged[:cap]
                dp[nv] = merged
    return dp

def meet_in_the_middle(valores, alvo, max_notas):
    n = len(valores)
    # Cap adaptativo: exato para ≤70 notas, bounded para mais
    if   n <= 70:  cap = None   # exato
    elif n <= 100: cap = 3      # ~1-2s
    else:          cap = 2      # fallback rápido

    alvo_cents = round(round(alvo, 2) * 100)
    tol_cents  = round(TOLERANCIA * 100)
    items = [(k, round(round(v, 2) * 100)) for k, v in valores.items() if round(v, 2) > 0]
    if not items: return []

    mid   = len(items) // 2
    left, right = items[:mid], items[mid:]
    max_l = max(1, max_notas * len(left)  // len(items))
    max_r = max(1, max_notas * len(right) // len(items))

    somas_l = _somas_grupo(left,  max_l, cap)
    somas_r = _somas_grupo(right, max_r, cap)

    resultados, vistas = [], set()
    for soma_l, combos_l in somas_l.items():
        needed = alvo_cents - soma_l
        for delta in range(-tol_cents, tol_cents + 1):
            soma_r = needed + delta
            if soma_r not in somas_r: continue
            for cl in combos_l:
                for cr in somas_r[soma_r]:
                    merged = frozenset(cl + cr)
                    if len(merged) > max_notas or merged in vistas: continue
                    vistas.add(merged)
                    total = soma_l + soma_r
                    dif   = round(abs(alvo_cents - total) / 100, 2)
                    resultados.append((dif, tuple(sorted(merged))))

    resultados.sort(key=lambda x: (x[0], len(x[1])))
    return resultados[:TOP_RESULTADOS * 3]

def fix1_busca(valores, alvo, max_notas):
    """
    Ancora cada nota grande (> alvo/max_notas) e roda MITM no restante.
    Cobre combinações que o MITM top-100 poderia perder em bases grandes.
    """
    resultados = []
    thr = alvo / max(max_notas, 1)
    sd  = sorted(valores.items(), key=lambda x: x[1], reverse=True)
    # Âncoras: notas entre alvo/max_notas e alvo, ordenadas pelo mais próximo do meio
    ancoras = sorted([(k, v) for k, v in sd if thr < v < alvo - TOLERANCIA],
                     key=lambda x: abs(x[1] - alvo / 2))
    vistas = set()
    for k1, v1 in ancoras:
        resto = round(alvo - v1, 2)
        sub = dict(sorted(
            {k: v for k, v in valores.items() if k != k1 and 0 < v <= resto + TOLERANCIA}.items(),
            key=lambda x: x[1], reverse=True
        )[:50])
        if not sub: continue
        for _, combo in meet_in_the_middle(sub, resto, max_notas - 1)[:5]:
            full = tuple(sorted((k1,) + combo))
            key  = frozenset(full)
            if key not in vistas:
                vistas.add(key)
                soma = round(v1 + sum(valores.get(k, 0) for k in combo), 2)
                resultados.append((round(abs(soma - alvo), 2), full))
        # Parar cedo se achou combinação exata
        if resultados and resultados[0][0] <= TOLERANCIA:
            break
    return resultados

# ============================================================
# ALGORITMO — CAMADA 4: SOLVER MILP (fallback)
# ============================================================

def solver_milp(valores, alvo, max_notas):
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        keys = list(valores.keys())
        vals = np.array([round(valores[k], 2) for k in keys])
        n    = len(keys)
        c    = np.ones(n)
        tol  = TOLERANCIA
        constraints  = LinearConstraint(vals.reshape(1, -1), lb=alvo - tol, ub=alvo + tol)
        constraints2 = LinearConstraint(np.ones((1, n)), lb=0, ub=max_notas)
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        res = milp(c, constraints=[constraints, constraints2],
                   integrality=np.ones(n), bounds=bounds)
        if res.success and res.x is not None:
            xi    = np.round(res.x).astype(int)
            combo = tuple(keys[i] for i in range(n) if xi[i] == 1)
            soma  = round(sum(valores[k] for k in combo), 2)
            dif   = round(abs(soma - round(alvo, 2)), 2)
            return [(dif, combo)]
    except Exception:
        pass
    return []

# ============================================================
# ALGORITMO — CAMADA 5: RANKING INTELIGENTE
# ============================================================

def _score(combo, valores, df_cli, alvo):
    alvo_r  = round(alvo, 2)
    soma    = round(sum(round(valores.get(k, 0), 2) for k in combo), 2)
    dif     = round(abs(soma - alvo_r), 2)
    n_notas = len(combo)
    rbases, atrasos = set(), []
    for chave in combo:
        idx_str, _ = chave.split("|")
        try:
            row = df_cli.loc[int(idx_str)]
            if "rbase" in df_cli.columns:
                rbases.add(str(row.get("rbase", "")))
            try:
                atrasos.append(float(row.get("atraso", 0) or 0))
            except Exception:
                atrasos.append(0)
        except Exception:
            pass
    media_atraso = float(np.mean(atrasos)) if atrasos else 0
    return (dif, n_notas, len(rbases), -media_atraso)

def _combo_sem_nf_duplicada(combo):
    """Rejeita combinações onde o mesmo índice de nota aparece mais de uma vez.
    Isso acontece quando o algoritmo seleciona duas variantes da mesma NF
    (ex: saldo + saldo_ir da NF 405599 ao mesmo tempo)."""
    indices = [chave.split("|")[0] for chave in combo]
    return len(indices) == len(set(indices))

def ranquear(brutos, valores, df_cli, alvo):
    # Deduplica por conjunto de ÍNDICES de nota (não por chave completa).
    # Isso evita que a mesma combinação de NFs apareça várias vezes só porque
    # uma delas usa "saldo_ir" num combo e "saldo_ir_iss" noutro.
    # De cada grupo de combos com os mesmos índices, fica só o de menor diferença.
    vistas_idx, com_score = set(), []
    for dif, combo in brutos:
        if not _combo_sem_nf_duplicada(combo): continue
        idx_set = frozenset(chave.split("|")[0] for chave in combo)
        if idx_set in vistas_idx: continue
        vistas_idx.add(idx_set)
        score = _score(combo, valores, df_cli, alvo)
        com_score.append((score, combo))
    com_score.sort(key=lambda x: x[0])
    return [(round(s[0], 2), c) for s, c in com_score[:TOP_RESULTADOS]]

# ============================================================
# ORQUESTRADOR
# ============================================================

def gerar_opcoes_nota(row):
    s   = round(float(row.get("saldo", 0)), 2)
    ir  = round(float(row.get("ir",    0)), 2)
    iss = round(float(row.get("iss",   0)), 2)
    ops = [("saldo", s)]
    if ir  != 0: ops.append(("saldo_ir",      round(s + ir,       2)))
    if iss != 0: ops.append(("saldo_iss",     round(s + iss,      2)))
    if ir != 0 or iss != 0:
        ops.append(("saldo_ir_iss", round(s + ir + iss, 2)))
    return ops

def buscar_combinacoes(valores_raw, alvo, df_cli, max_notas=MAX_NOTAS):
    valores = preprocessar(valores_raw, alvo)
    if not valores: return [], {}

    n_filtradas = len(valores)
    brutos = []

    # C1: Greedy (instantâneo)
    g = greedy(valores, alvo, max_notas)
    if g: brutos.append(g)

    # C2: MITM (exato ≤70 notas, bounded acima)
    mitm = meet_in_the_middle(valores, alvo, max_notas)
    brutos.extend(mitm)

    # C3: Fix-1 (ancora nota grande + MITM no restante) — para bases maiores
    if n_filtradas > 50 and not (brutos and brutos[0][0] <= TOLERANCIA):
        fix1 = fix1_busca(valores, alvo, max_notas)
        brutos.extend(fix1)

    # C4: MILP fallback (quando nada encontrou)
    if not brutos:
        brutos.extend(solver_milp(valores, alvo, max_notas))

    if not brutos: return [], {"pre": n_filtradas, "camada": "Nenhuma"}

    camada = ("Greedy"              if g and not mitm
              else "MITM + Fix-1"   if mitm and n_filtradas > 50
              else "Meet-in-the-Middle" if mitm
              else "MILP")
    resultado = ranquear(brutos, valores, df_cli, alvo)
    return resultado, {"pre": n_filtradas, "camada": camada}

def _montar_df_valores(df_sub):
    """Monta dicionário de valores para um subconjunto de notas."""
    valores = {}
    for idx, row in df_sub.iterrows():
        for nome, valor in gerar_opcoes_nota(row):
            valores[f"{idx}|{nome}"] = valor
    return valores

def conciliar(df_cli, valor_alvo, escopo="direto"):
    """
    escopo:
      "direto"  — df_cli já é o conjunto certo (CNPJ ou NOME)
      "cidade"  — aplica busca em cascata: CNPJ individual → grupo RBASE → cidade
    """
    t0 = time.time()

    if escopo == "cidade":
        resultados_raw, meta, origem = _conciliar_cascata_cidade(df_cli, valor_alvo)
    else:
        valores = _montar_df_valores(df_cli)
        resultados_raw, meta = buscar_combinacoes(valores, valor_alvo, df_cli)
        origem = "cnpj"

    elapsed = round((time.time() - t0) * 1000)
    meta["tempo_ms"] = elapsed
    meta["origem"]   = origem

    resultados = []
    for dif, combo in resultados_raw:
        rbases, linhas = [], []
        total = 0
        for chave in combo:
            idx_str, tipo = chave.split("|")
            idx = int(idx_str)
            try:
                row = df_cli.loc[idx]
            except Exception:
                continue
            s   = round(float(row.get("saldo", 0)), 2)
            ir  = round(float(row.get("ir",    0)), 2)
            iss = round(float(row.get("iss",   0)), 2)

            if   tipo == "saldo":        val = s
            elif tipo == "saldo_ir":     val = round(s + ir,       2)
            elif tipo == "saldo_iss":    val = round(s + iss,      2)
            else:                        val = round(s + ir + iss, 2)

            retencao = round(s - val, 2)
            total    = round(total + val, 2)
            rb = str(row.get("rbase", ""))
            if rb and rb not in rbases: rbases.append(rb)

            linhas.append({
                "NF":            row.get("nf",    idx),
                "Nome":          str(row.get("nome",  "")),
                "CNPJ":          str(row.get("cnpj",  "")),
                "Venc.":         row.get("venc",  ""),
                "Atraso":        row.get("atraso", ""),
                "Saldo":         s,
                "IR":            ir,
                "ISS":           iss,
                "Vlr Utilizado": val,
                "Tipo":          tipo,
                "Retencao":      retencao,
            })

        resultados.append({
            "dif":   round(dif, 2),
            "total": round(total, 2),
            "rbase": " / ".join(rbases) or "N/A",
            "notas": pd.DataFrame(linhas),
        })

    return resultados, meta

# ============================================================
# BUSCA EM CASCATA PARA CIDADE
# ============================================================

def _conciliar_cascata_cidade(df_cidade, valor_alvo):
    """
    Tenta fechar o valor em 3 camadas, parando na primeira que encontrar resultado:

    Camada A — CNPJ individual
        Para cada CNPJ distinto da cidade, tenta fechar o valor
        usando somente as notas daquele CNPJ.
        Lógica: o pagamento mais comum é de um único pagador.

    Camada B — Grupo RBASE
        Agrupa CNPJs que compartilham a mesma RBASE e tenta fechar
        o valor dentro de cada grupo.
        Lógica: entidades do mesmo grupo econômico pagam juntas.

    Camada C — Cidade ampla (fallback)
        Abre para todos os CNPJs da cidade.
        Sinaliza que a composição mistura pagadores distintos.
    """

    # ── Camada A: CNPJ individual ─────────────────────────────
    col_cnpj = "cnpj_limpo" if "cnpj_limpo" in df_cidade.columns else "cnpj"
    for cnpj_val in df_cidade[col_cnpj].unique():
        df_sub = df_cidade[df_cidade[col_cnpj] == cnpj_val]
        valores = _montar_df_valores(df_sub)
        res, meta = buscar_combinacoes(valores, valor_alvo, df_sub)
        if res:
            nome_cnpj = df_sub["nome"].iloc[0] if "nome" in df_sub.columns else cnpj_val
            meta["escopo_descricao"] = f"CNPJ individual: {nome_cnpj}"
            return res, meta, "cnpj_individual"

    # ── Camada B: Grupo RBASE ─────────────────────────────────
    if "rbase" in df_cidade.columns:
        for rbase_val in df_cidade["rbase"].unique():
            df_sub = df_cidade[df_cidade["rbase"] == rbase_val]
            if df_sub["cnpj_limpo" if "cnpj_limpo" in df_cidade.columns else "cnpj"].nunique() < 2:
                continue   # grupo com só 1 CNPJ já foi testado na camada A
            valores = _montar_df_valores(df_sub)
            res, meta = buscar_combinacoes(valores, valor_alvo, df_sub)
            if res:
                meta["escopo_descricao"] = f"Grupo RBASE {rbase_val} ({len(df_sub)} notas de {df_sub['cnpj'].nunique() if 'cnpj' in df_sub.columns else '?'} CNPJs)"
                return res, meta, "grupo_rbase"

    # ── Camada C: Cidade ampla ────────────────────────────────
    valores = _montar_df_valores(df_cidade)
    res, meta = buscar_combinacoes(valores, valor_alvo, df_cidade)
    meta["escopo_descricao"] = f"Cidade completa ({df_cidade[col_cnpj].nunique()} CNPJs distintos)"
    return res, meta, "cidade_ampla"

# ============================================================
# HELPERS DE EXIBIÇÃO
# ============================================================

TIPO_BADGE = {
    "saldo":        ("🟢 Saldo",           "green"),
    "saldo_ir":     ("🟡 Saldo − IR",      "yellow"),
    "saldo_iss":    ("🟡 Saldo − ISS",     "yellow"),
    "saldo_ir_iss": ("🔴 Saldo − IR/ISS",  "red"),
}

ORIGEM_CONFIG = {
    "cnpj_individual": ("🎯 CNPJ individual",    "blue",   "Composição de notas de um único pagador."),
    "grupo_rbase":     ("🔗 Grupo RBASE",         "purple", "Composição mistura CNPJs do mesmo grupo econômico (mesma RBASE)."),
    "cidade_ampla":    ("⚠️  Cidade completa",    "yellow", "Atenção: composição mistura CNPJs de pagadores distintos. Confirme se houve pagamento conjunto."),
    "cnpj":            ("🎯 Busca direta",         "blue",   ""),
}

def brl(v):
    try:
        return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"

def fmt_tipo(tipo):
    label, _ = TIPO_BADGE.get(tipo, (tipo, "gray"))
    return label

def exibir_combo(combo, i, n_total):
    is_best      = (i == 0)
    tem_retencao = combo["notas"]["Retencao"].abs().gt(TOLERANCIA).any()
    df           = combo["notas"].copy()

    tags = []
    if is_best:      tags.append('<span class="badge badge-blue">★ MELHOR OPÇÃO</span>')
    if tem_retencao: tags.append('<span class="badge badge-red">⚠ RETENÇÃO DETECTADA</span>')

    st.markdown(f"""
    <div style="border:{'1.5px solid #3b82f6' if is_best else '1px solid #1e293b'};
         border-radius:12px; overflow:hidden; margin-bottom:20px;
         {'box-shadow:0 0 0 1px #3b82f620, 0 4px 24px #3b82f610;' if is_best else ''}">
        <div style="background:{'#1e3a5f' if is_best else '#0a0f1a'};
             padding:14px 20px; border-bottom:1px solid #1e293b;
             display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <span style="background:{'#3b82f6' if is_best else '#334155'};
                  color:#fff; border-radius:6px; padding:3px 12px;
                  font-size:12px; font-weight:800; font-family:'IBM Plex Mono',monospace;">
                #{i+1}
            </span>
            {''.join(tags)}
            <span style="margin-left:auto; font-size:12px; color:#64748b;">
                {len(df)} nota(s) &nbsp;|&nbsp; RBASE: {combo['rbase']}
                &nbsp;|&nbsp; Total:
                <b style="color:{'#4ade80' if combo['dif']==0 else '#f59e0b'};">{brl(combo['total'])}</b>
                &nbsp;|&nbsp; Δ:
                <b style="color:{'#4ade80' if combo['dif']==0 else '#f87171'};">{brl(combo['dif'])}</b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_show = df.copy()
    # Mostra CNPJ só se a combinação tiver mais de 1 CNPJ distinto
    cnpjs_distintos = df_show["CNPJ"].nunique() if "CNPJ" in df_show.columns else 1
    colunas_exibir = ["NF", "Nome"]
    if cnpjs_distintos > 1:
        colunas_exibir.append("CNPJ")
    colunas_exibir += ["Venc.", "Atraso", "Saldo", "IR", "ISS", "Vlr Utilizado", "Tipo"]
    df_show = df_show[[c for c in colunas_exibir if c in df_show.columns]]

    for col in ["Saldo", "IR", "ISS", "Vlr Utilizado", "Retencao"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(brl)
    df_show["Tipo"] = df_show["Tipo"].apply(fmt_tipo)
    if "Venc." in df_show.columns:
        df_show["Venc."] = df_show["Venc."].apply(
            lambda v: v.strftime("%d/%m/%Y") if hasattr(v, "strftime") else str(v) if v else "—"
        )

    st.dataframe(
        df_show, use_container_width=True, hide_index=True,
        column_config={"Atraso": st.column_config.NumberColumn("Atraso", format="%d dias")}
    )

    if tem_retencao:
        retidos = combo["notas"][combo["notas"]["Retencao"].abs() > TOLERANCIA]
        itens = []
        for _, r in retidos.iterrows():
            alerta = "IR+ISS" if r["IR"] != 0 and r["ISS"] != 0 else "IR" if r["IR"] != 0 else "ISS"
            itens.append(f"NF {r['NF']} — retido <b>{brl(r['Retencao'])}</b> ({alerta})")
        st.markdown(f"""
        <div class="alert-retencao">
            <b>⚠ Possível retenção indevida</b> — cliente deduziu IR/ISS do pagamento.<br>
            Verifique isenção (ex: Simples Nacional) e solicite reembolso se aplicável.<br><br>
            {"<br>".join(itens)}
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# CALCULADORA DE COMBINAÇÕES LIVRES (aba 2)
# ============================================================

def _parse_num(tok):
    """Converte um token de texto para float, suportando formato BR e EN."""
    import re
    tok = tok.strip().replace("\xa0", "").replace(" ", "")
    tok = re.sub(r"[R$]", "", tok)
    if tok in ("-", "–", "—", ""):
        return 0.0
    if re.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$", tok):
        tok = tok.replace(".", "").replace(",", ".")
    else:
        tok = tok.replace(",", ".")
    try:
        return float(tok)
    except ValueError:
        return None

def _parse_valores_livres(texto, modo="saldo"):
    """
    Aceita dois formatos:
    1. Valor simples por linha: 1.234,56
    2. Tabular com tabs: Saldo \t IR \t ISS  (formato planilha)

    modo:
      "saldo"   → usa só a 1ª coluna
      "liquido" → soma todas as colunas da linha (saldo + IR + ISS)
    """
    import re
    rows = []
    for line in texto.splitlines():
        # Linha com tabs → formato tabular
        if "\t" in line:
            cols = re.split(r"\t", line)
        else:
            # Linha simples: trata ; e | como separadores
            cols = re.split(r"[;|]", line)

        parsed = []
        for c in cols:
            v = _parse_num(c)
            if v is not None:
                parsed.append(v)

        if not parsed:
            continue

        if modo == "liquido":
            # Soma todos os valores da linha (IR e ISS já são negativos)
            val = round(sum(parsed), 2)
        else:
            # Só o primeiro valor positivo da linha
            val = next((v for v in parsed if v > 0), None)
            if val is None:
                continue
            val = round(val, 2)

        if val > 0:
            rows.append(val)

    return rows

def _buscar_combinacoes_livres(valores_lista, alvo, max_n, top):
    """
    Usa o mesmo motor MITM para uma lista simples de floats.
    Retorna lista de (soma, [indices_usados]).
    """
    # Cria dict chaveado por índice
    valores = {str(i): v for i, v in enumerate(valores_lista)}
    valores = preprocessar(valores, alvo)
    if not valores:
        return []

    brutos = []
    g = greedy(valores, alvo, max_n)
    if g:
        brutos.append(g)
    mitm = meet_in_the_middle(valores, alvo, max_n)
    brutos.extend(mitm)

    if not brutos:
        brutos.extend(solver_milp(valores, alvo, max_n))

    # Deduplicar por VALORES ordenados — evita mostrar a mesma composição
    # de valores com índices diferentes (ex: dois 497,95 com idx 4 e 28)
    vistas, resultado = set(), []
    for dif, combo in brutos:
        parcelas = tuple(sorted(round(float(valores.get(k, 0)), 2) for k in combo))
        if parcelas in vistas:
            continue
        vistas.add(parcelas)
        soma = round(sum(parcelas), 2)
        resultado.append((dif, soma, sorted(combo, key=lambda x: int(x))))

    resultado.sort(key=lambda x: (x[0], len(x[2])))
    return resultado[:top]

def aba_calculadora():
    st.markdown("""
    <div style="background:#0a1628; border:1px solid #1e293b; border-radius:12px;
         padding:20px 24px; margin-bottom:24px;">
        <div style="font-family:'IBM Plex Sans',sans-serif; font-weight:700;
             font-size:15px; color:#e2e8f0; margin-bottom:6px;">
            🧮 Calculadora de Combinações
        </div>
        <div style="font-size:12px; color:#64748b;">
            Cole uma lista de valores, informe o alvo e encontre quais combinações somam exatamente esse valor.
            Útil para conciliar qualquer conjunto de números sem precisar de planilha.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detectar se o texto colado tem tabs (formato planilha) ──
    # Fazemos isso antes dos widgets para mostrar o radio só quando relevante

    c1, c2 = st.columns([2, 1])

    with c1:
        texto_vals = st.text_area(
            "Cole os valores aqui",
            height=220,
            placeholder=(
                "Formato simples (um por linha):\n"
                "124.170,17\n56.307,30\n47.724,75\n\n"
                "── ou formato planilha (Saldo  IR  ISS) ──\n"
                "629,94\t-14,40\t-\n"
                "1.464,85\t-3,30\t-\n"
                "147.436,07\t-7.051,24\t-"
            ),
            help=(
                "Aceita valores simples (um por linha) ou copiados direto da planilha "
                "com 3 colunas: Saldo, IR e ISS separados por tab."
            ),
            key="calc_texto"
        )

        tem_tabs = "\t" in (texto_vals or "")
        if tem_tabs:
            modo = st.radio(
                "Usar como valor de cada linha:",
                ["Saldo bruto", "Líquido (Saldo + IR + ISS)"],
                horizontal=True,
                key="calc_modo",
                help="'Saldo bruto' ignora IR/ISS. 'Líquido' soma as três colunas (IR e ISS já são negativos)."
            )
            modo_key = "liquido" if "Líquido" in modo else "saldo"

            # Preview em tempo real dos valores reconhecidos
            if texto_vals.strip():
                preview = _parse_valores_livres(texto_vals, modo=modo_key)
                if preview:
                    st.caption(f"✅ {len(preview)} valores reconhecidos — soma: {brl(sum(preview))}")
                    with st.expander("Ver valores que serão usados", expanded=False):
                        df_prev = pd.DataFrame({
                            "Nº":    range(1, len(preview)+1),
                            "Valor": [brl(v) for v in preview],
                        })
                        st.dataframe(df_prev, use_container_width=True, hide_index=True)
                else:
                    st.caption("⚠️ Nenhum valor reconhecido ainda.")
        else:
            modo_key = "saldo"

    with c2:
        alvo_str = st.text_input(
            "Valor alvo (R$)",
            placeholder="Ex: 436.611,53",
            key="calc_alvo"
        )
        max_n = st.slider(
            "Máx. de parcelas por combinação",
            min_value=1, max_value=100, value=8,
            key="calc_maxn"
        )
        top_n = st.slider(
            "Máx. de resultados",
            min_value=1, max_value=20, value=10,
            key="calc_top"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        calc_btn = st.button("🔍 Calcular combinações", use_container_width=True, type="primary", key="calc_btn")

    if calc_btn:
        if not texto_vals.strip():
            st.error("Cole pelo menos um valor.")
            return
        if not alvo_str.strip():
            st.error("Informe o valor alvo.")
            return

        try:
            alvo = float(alvo_str.replace(".", "").replace(",", "."))
        except Exception:
            st.error("Valor alvo inválido.")
            return

        vals = _parse_valores_livres(texto_vals, modo=modo_key)
        if not vals:
            st.error("Nenhum valor numérico reconhecido. Verifique o formato.")
            return

        st.markdown("---")

        # Stats rápidos
        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(f'<div class="stat-box"><div class="stat-label">Valores colados</div><div class="stat-value">{len(vals)}</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-box"><div class="stat-label">Soma total</div><div class="stat-value" style="font-size:15px;">{brl(sum(vals))}</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-box"><div class="stat-label">Valor alvo</div><div class="stat-value green" style="font-size:15px;">{brl(alvo)}</div></div>', unsafe_allow_html=True)
        s4.markdown(f'<div class="stat-box"><div class="stat-label">Diferença</div><div class="stat-value {"green" if abs(sum(vals)-alvo)<0.01 else "yellow"}" style="font-size:15px;">{brl(round(sum(vals)-alvo,2))}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("Buscando combinações..."):
            t0 = time.time()
            resultados = _buscar_combinacoes_livres(vals, alvo, max_n, top_n)
            elapsed = round((time.time() - t0) * 1000)

        if not resultados:
            st.error("Nenhuma combinação encontrada. Tente aumentar o máximo de parcelas ou ajustar a tolerância.")
            return

        st.markdown(f"""
        <div class="alert-multi">
            ⚡ <b>{len(resultados)} combinação(ões)</b> encontrada(s) em <b>{elapsed}ms</b>.
            A #1 usa o menor número de parcelas.
        </div>
        """, unsafe_allow_html=True)

        for i, (dif, soma, indices) in enumerate(resultados, 1):
            is_best  = (i == 1)
            parcelas = [vals[int(idx)] for idx in indices]

            # Cabeçalho do card
            borda    = "1.5px solid #3b82f6" if is_best else "1px solid #1e293b"
            bg_head  = "#1e3a5f"             if is_best else "#0a0f1a"
            cor_num  = "#3b82f6"             if is_best else "#334155"
            melhor   = " &nbsp; ★ MELHOR OPÇÃO" if is_best else ""
            cor_soma = "#4ade80" if dif == 0 else "#f59e0b"
            cor_dif  = "#4ade80" if dif == 0 else "#f87171"

            st.markdown(
                "<div style='border:" + borda + "; border-radius:12px 12px 0 0; overflow:hidden;'>"
                "<div style='background:" + bg_head + "; padding:12px 20px; "
                "display:flex; align-items:center; gap:12px; flex-wrap:wrap;'>"
                "<span style='background:" + cor_num + "; color:#fff; border-radius:6px; "
                "padding:2px 10px; font-size:13px; font-weight:800; font-family:monospace;'>"
                "#" + str(i) + melhor + "</span>"
                "<span style='margin-left:auto; font-size:12px; color:#94a3b8; font-family:monospace;'>"
                + str(len(parcelas)) + " parcela(s) &nbsp;|&nbsp; "
                "Soma: <b style='color:" + cor_soma + ";'>" + brl(soma) + "</b> &nbsp;|&nbsp; "
                "Diferença: <b style='color:" + cor_dif + ";'>" + brl(dif) + "</b></span>"
                "</div></div>",
                unsafe_allow_html=True
            )

            # Tabela de parcelas
            df_parc = pd.DataFrame({
                "Nº":    [int(idx) + 1 for idx in indices],
                "Valor": [brl(vals[int(idx)]) for idx in indices],
            })
            df_total = pd.DataFrame({"Nº": ["TOTAL"], "Valor": [brl(soma)]})
            df_parc  = pd.concat([df_parc, df_total], ignore_index=True)
            st.dataframe(df_parc, use_container_width=True, hide_index=True)
            st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)



# ============================================================
# INTERFACE PRINCIPAL — com três abas
# ============================================================

st.markdown("""
<div class="header-box">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:36px;">⚖️</div>
        <div>
            <div style="font-family:'IBM Plex Sans',sans-serif; font-weight:800;
                 font-size:22px; letter-spacing:-0.02em; color:#f1f5f9;">
                Motor de Conciliação Financeira
            </div>
            <div style="font-size:12px; color:#475569; letter-spacing:0.05em; margin-top:2px;">
                MAXIFROTA — ANÁLISE DE TÍTULOS VENCIDOS
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

aba_conciliacao, aba_calc, aba_ret, aba_itau = st.tabs([
    "⚖️  Conciliação com Planilha",
    "🧮  Calculadora de Combinações",
    "🎯  Liquidação por Retenção",
    "🏦  Conciliação Itaú",
])

# ══════════════════════════════════════════════════════════════
# ABA 1 — CONCILIAÇÃO COM PLANILHA
# ══════════════════════════════════════════════════════════════
with aba_conciliacao:

    uploaded = st.file_uploader(
        "📂 Carregar planilha de pendências",
        type=["xlsx", "xls"],
        help="Arraste o arquivo .xlsx com os títulos vencidos"
    )

    if uploaded:
        with st.spinner("Lendo planilha..."):
            df = carregar_planilha(uploaded.read(), uploaded.name)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="stat-box"><div class="stat-label">Arquivo</div><div class="stat-value" style="font-size:14px;">{uploaded.name}</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="stat-box"><div class="stat-label">Títulos</div><div class="stat-value green">{len(df):,}</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="stat-box"><div class="stat-label">Colunas mapeadas</div><div class="stat-value">{len([c for c in COLUNAS if c in df.columns])}/{len(COLUNAS)}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        c1, c2, c3, c4 = st.columns([1.2, 2, 1.8, 1])

        with c1:
            busca_por = st.selectbox("Buscar por", ["Cidade", "CNPJ", "Nome"], index=0)

        with c2:
            placeholder = {
                "Cidade": "Ex: Abdon Batista",
                "CNPJ":   "00.000.000/0001-00",
                "Nome":   "Ex: Municipio de...",
            }[busca_por]
            identificador = st.text_input(busca_por, placeholder=placeholder)

        with c3:
            valor_str = st.text_input("Valor alvo (R$)", placeholder="Ex: 31.452,88")

        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            executar = st.button("Conciliar →", use_container_width=True, type="primary")

        if executar:
            if not identificador:
                st.error("Informe o identificador de busca.")
            else:
                try:
                    valor = float(valor_str.replace(".", "").replace(",", "."))
                except Exception:
                    st.error("Valor alvo inválido. Use vírgula como decimal (ex: 31.452,88).")
                    st.stop()

                bp = busca_por.upper()

                if bp == "CNPJ":
                    limpo  = identificador.replace(".", "").replace("/", "").replace("-", "").lstrip("0")
                    col_c  = "cnpj_limpo" if "cnpj_limpo" in df.columns else "cnpj"
                    df_cli = df[df[col_c].str.lstrip("0") == limpo]
                    escopo = "direto"

                elif bp == "CIDADE":
                    df_cli = df[df["cidade"].str.strip().str.upper() == identificador.strip().upper()]
                    escopo = "cidade"

                else:
                    df_cli = df[df["nome"].str.strip().str.upper().str.contains(
                        identificador.strip().upper(), na=False)]
                    escopo = "direto"

                if df_cli.empty:
                    st.error(f"Nenhum registro encontrado para {busca_por}: **{identificador}**")
                    st.stop()

                with st.spinner(f"Buscando combinações em {len(df_cli)} nota(s)..."):
                    resultados, meta = conciliar(df_cli, valor, escopo=escopo)

                st.markdown("---")
                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(f'<div class="stat-box"><div class="stat-label">Notas analisadas</div><div class="stat-value">{len(df_cli)}</div></div>', unsafe_allow_html=True)
                s2.markdown(f'<div class="stat-box"><div class="stat-label">Combinações</div><div class="stat-value {"yellow" if len(resultados) > 1 else "green"}">{len(resultados)}</div></div>', unsafe_allow_html=True)
                s3.markdown(f'<div class="stat-box"><div class="stat-label">Tempo</div><div class="stat-value">{meta["tempo_ms"]}ms</div></div>', unsafe_allow_html=True)
                s4.markdown(f'<div class="stat-box"><div class="stat-label">Algoritmo</div><div class="stat-value" style="font-size:14px;">{meta.get("camada","—")}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if bp == "CIDADE" and resultados:
                    origem = meta.get("origem", "cnpj")
                    label, cor, descricao = ORIGEM_CONFIG.get(origem, ("", "gray", ""))
                    escopo_desc = meta.get("escopo_descricao", "")
                    css_class = f"alert-{'escopo' if cor == 'purple' else 'multi' if cor == 'yellow' else 'retencao' if cor == 'red' else 'multi'}"
                    if descricao:
                        st.markdown(f"""
                        <div class="{css_class}">
                            <b>{label}</b> — {descricao}<br>
                            <span style="opacity:0.7; font-size:11px;">{escopo_desc}</span>
                        </div>
                        """, unsafe_allow_html=True)

                if not resultados:
                    st.error("Nenhuma combinação encontrada. Verifique o valor ou tente ajustar a tolerância.")
                else:
                    if len(resultados) > 1:
                        st.markdown(f"""
                        <div class="alert-multi">
                            ⚡ <b>{len(resultados)} composições distintas</b> encontradas.
                            A combinação <b>#1</b> é a mais simples e financeiramente recomendada.
                        </div>
                        """, unsafe_allow_html=True)

                    for i, combo in enumerate(resultados):
                        exibir_combo(combo, i, len(resultados))

    else:
        st.info("⬆️ Carregue a planilha de títulos vencidos para iniciar a conciliação.")
        st.markdown("""
**Como usar:**
1. Faça o upload da planilha `.xlsx` com os títulos vencidos
2. Selecione o tipo de busca (Cidade, CNPJ ou Nome)
3. Digite o identificador do cliente
4. Informe o valor recebido a conciliar
5. Clique em **Conciliar →**

**Busca por Cidade — cascata automática:**
- 🎯 Tenta primeiro fechar o valor com um único CNPJ
- 🔗 Se não achar, tenta dentro do mesmo grupo RBASE
- ⚠️ Só abre para a cidade inteira se as anteriores falharem (com aviso)

**O motor identifica automaticamente:**
- ✅ A composição de notas mais simples para o valor informado
- ⚠️ Possíveis retenções indevidas de IR/ISS
- ⚡ Múltiplas composições quando existirem
        """)

# ══════════════════════════════════════════════════════════════
# ABA 2 — CALCULADORA DE COMBINAÇÕES LIVRES
# ══════════════════════════════════════════════════════════════
with aba_calc:
    aba_calculadora()

# ============================================================
# ABA 3 — LIQUIDAÇÃO POR RETENÇÃO (funções auxiliares)
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_retencao(file_bytes, file_name):
    """Lê planilha de títulos em aberto e normaliza colunas de retenção."""
    import io
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()

    # Mapear colunas com nomes alternativos
    MAP = {
        "NOME":          ["NOME", "NOME CLIENTE"],
        "CNPJ":          ["CNPJ"],
        "CIDADE":        ["CIDADE"],
        "UF":            ["UF"],
        "NF":            ["NF", "NR NFEM"],
        "VLR SALDO":     ["SALDO", "VLR SALDO"],
        "IR":            ["IR", "VLR RETIDO"],
        "ISS":           ["ISS", "ISS RETIDO"],
        "LIQ CORRETO":   ["CORRETO", "LIQ CORRETO"],
        "VENC":          ["VENCIMENTO", "VENC", "DT VENCIMENTO"],
        "ATRASO":        ["ATRASO"],
        "RBASE":         ["RBASE RAIZ", "RBASE"],
    }
    rename = {}
    cols_up = {c.upper(): c for c in df.columns}
    for dest, candidates in MAP.items():
        for cand in candidates:
            if cand.upper() in cols_up and dest not in df.columns:
                rename[cols_up[cand.upper()]] = dest
    df.rename(columns=rename, inplace=True)

    for c in ["VLR SALDO", "IR", "ISS", "ATRASO"]:  # já renomeado pelo MAP acima
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(2)

    # Calcular campos derivados
    if "LIQ CORRETO" not in df.columns:
        df["LIQ CORRETO"] = (df["VLR SALDO"] + df["IR"] + df["ISS"]).round(2)

    df["RETENCAO_IR"]  = df["IR"].abs().round(2)
    df["RETENCAO_ISS"] = df["ISS"].abs().round(2)
    df["RETENCAO_TOT"] = (df["RETENCAO_IR"] + df["RETENCAO_ISS"]).round(2)
    df["DIFERENCA"]    = (df["VLR SALDO"] - df["LIQ CORRETO"]).round(2)  # = RETENCAO_TOT

    # Classificação
    def classif(row):
        if row["RETENCAO_IR"] > 0 and row["RETENCAO_ISS"] > 0:
            return "IR + ISS"
        elif row["RETENCAO_IR"] > 0:
            return "IR"
        elif row["RETENCAO_ISS"] > 0:
            return "ISS"
        return "Sem retenção"
    df["TIPO_RETENCAO"] = df.apply(classif, axis=1)

    if "CNPJ" in df.columns:
        df["CNPJ"] = df["CNPJ"].astype(str).str.strip()

    df.index = range(len(df))
    return df


def aba_retencao():
    st.markdown("""
    <div style="background:#0a1628; border:1px solid #1e293b; border-radius:12px;
         padding:20px 24px; margin-bottom:24px;">
        <div style="font-family:'IBM Plex Sans',sans-serif; font-weight:700;
             font-size:15px; color:#e2e8f0; margin-bottom:6px;">
            🎯 Notas Candidatas a Liquidação
        </div>
        <div style="font-size:12px; color:#64748b;">
            Identifica notas fiscais onde o saldo em aberto é próximo do total retido (IR + ISS),
            ou seja, a retenção cobre quase todo o valor — tornando viável a liquidação com mínimo ajuste.
        </div>
    </div>
    """, unsafe_allow_html=True)

    up = st.file_uploader(
        "📂 Planilha de títulos em aberto (.xlsx)",
        type=["xlsx","xls"],
        key="ret_upload",
    )

    if not up:
        st.info("⬆️ Carregue a planilha para identificar as notas candidatas à liquidação.")
        return

    # ── Leitura e normalização ─────────────────────────────────
    df = pd.read_excel(up)
    df.columns = df.columns.str.strip()

    # Mapa de colunas: aceita nome novo e antigo
    MAPA_RET = {
        "NF":        ["NF", "NR NFEM"],
        "NOME":      ["NOME CLIENTE", "NOME"],
        "CIDADE":    ["CIDADE"],
        "CNPJ":      ["CNPJ"],
        "VLR SALDO": ["SALDO", "VLR SALDO"],
        "IR":        ["IR", "VLR RETIDO"],
        "ISS":       ["ISS", "ISS RETIDO"],
        "VENC":      ["VENCIMENTO", "VENC", "DT VENCIMENTO"],
        "ATRASO":    ["ATRASO"],
    }
    cols_up = {c.upper(): c for c in df.columns}
    rename_map = {}
    for destino, candidatos in MAPA_RET.items():
        if destino not in df.columns:
            for cand in candidatos:
                if cand.upper() in cols_up:
                    rename_map[cols_up[cand.upper()]] = destino
                    break
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Garantir colunas numéricas
    for c in ["VLR SALDO","IR","ISS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(2)
        else:
            df[c] = 0.0

    # ── Cálculo central ────────────────────────────────────────
    # Remover linhas sem saldo (evita zeros/None poluírem o resultado)
    df = df[df["VLR SALDO"] > 0].copy()

    df["RETENCAO"] = (df["IR"].abs() + df["ISS"].abs()).round(2)
    # Só faz sentido considerar notas que TÊM alguma retenção
    df = df[df["RETENCAO"] > 0].copy()
    df["DIFERENÇA"] = (df["VLR SALDO"] - df["RETENCAO"]).abs().round(2)

    # ── Controle de tolerância ─────────────────────────────────
    tolerancia = st.slider(
        "Tolerância máxima de diferença (R$)",
        min_value=0.0, max_value=50.0, value=10.0, step=0.5,
        format="R$ %.2f",
        key="ret_tol",
        help="Notas onde |Saldo − Retenção| ≤ este valor serão listadas."
    )

    df_cand = df[df["DIFERENÇA"] <= tolerancia].copy()
    df_cand = df_cand.sort_values("DIFERENÇA")

    # ── Métricas ───────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="stat-box"><div class="stat-label">Notas candidatas</div><div class="stat-value green">{len(df_cand):,}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stat-box"><div class="stat-label">Total saldo a liquidar</div><div class="stat-value" style="font-size:13px;">{brl(df_cand["VLR SALDO"].sum())}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stat-box"><div class="stat-label">Total retido nessas NFs</div><div class="stat-value" style="font-size:13px;">{brl(df_cand["RETENCAO"].sum())}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="stat-box"><div class="stat-label">Ajuste total necessário</div><div class="stat-value yellow" style="font-size:13px;">{brl(df_cand["DIFERENÇA"].sum())}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df_cand.empty:
        st.warning(f"Nenhuma nota com diferença ≤ R$ {tolerancia:.2f} encontrada.")
        return

    # ── Tabela de exibição (só colunas que existem) ────────────
    COLS_EXIB = ["NF","NOME","CIDADE","VLR SALDO","IR","ISS","RETENCAO","DIFERENÇA","VENC","ATRASO"]
    cols_ok = [c for c in COLS_EXIB if c in df_cand.columns]
    df_exib = df_cand[cols_ok].copy()

    # Formatar NF como inteiro
    if "NF" in df_exib.columns:
        df_exib["NF"] = df_exib["NF"].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("nan","None","") else "")

    # Guardar versão numérica para export antes de formatar
    df_export = df_exib.copy()

    # Formatar monetários para exibição
    for col in ["VLR SALDO","IR","ISS","RETENCAO","DIFERENÇA"]:
        if col in df_exib.columns:
            df_exib[col] = df_exib[col].apply(lambda v: brl(v) if pd.notna(v) else "")

    # Renomear para exibição
    df_exib.rename(columns={
        "VLR SALDO": "Saldo", "RETENCAO": "Total Retido",
        "DIFERENÇA": "Diferença", "VENC": "Vencimento", "ATRASO": "Atraso (dias)"
    }, inplace=True)

    st.markdown(
        f"**{len(df_exib):,} nota(s)** onde a retenção cobre o saldo com diferença de até **{brl(tolerancia)}**"
        f" — ordenadas da menor para a maior diferença:",
        unsafe_allow_html=True,
    )
    st.dataframe(df_exib, use_container_width=True, hide_index=True, height=480)

    # ── Export Excel ───────────────────────────────────────────
    import io as _io
    buf = _io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_export.rename(columns={
            "VLR SALDO": "Saldo", "RETENCAO": "Total Retido",
            "DIFERENÇA": "Diferença", "VENC": "Vencimento", "ATRASO": "Atraso (dias)"
        }, inplace=True)
        df_export.to_excel(writer, index=False, sheet_name="Liquidação Retenção")
        wb  = writer.book
        ws  = writer.sheets["Liquidação Retenção"]
        fmt_brl  = wb.add_format({"num_format": "R$ #,##0.00", "align": "right"})
        fmt_head = wb.add_format({"bold": True, "bg_color": "#0a1628", "font_color": "#e2e8f0", "border": 1})
        for col_idx, col_name in enumerate(df_export.columns):
            ws.set_column(col_idx, col_idx, max(18, len(str(col_name))+4))
            if col_name in ("Saldo","IR","ISS","Total Retido","Diferença"):
                ws.set_column(col_idx, col_idx, 18, fmt_brl)
            ws.write(0, col_idx, col_name, fmt_head)
    buf.seek(0)

    st.download_button(
        label="⬇️ Exportar para Excel",
        data=buf,
        file_name="liquidacao_retencao.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════
# ABA 3 — NOTAS CANDIDATAS A LIQUIDAÇÃO
# ══════════════════════════════════════════════════════════════
with aba_ret:
    aba_retencao()

# ============================================================
# ABA 4 — CONCILIAÇÃO BANCÁRIA ITAÚ (PDF × CSV)
# ============================================================

def extrair_liquidacoes_pdf(file_bytes):
    """Extrai registros de liquidação do PDF do Itaú."""
    try:
        import pdfplumber
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "pdfplumber", "--break-system-packages", "-q"])
        import pdfplumber
    import io, re

    COL_SEU_NUM_MIN    = 280
    COL_SEU_NUM_MAX    = 360
    COL_TIPO_MIN       = 195
    COL_TIPO_MAX       = 240
    COL_OPERACAO_MIN   = 620
    COL_OPERACAO_MAX   = 760
    COL_VALOR_FINAL_MIN = 760
    COL_VALOR_FINAL_MAX = 850
    COL_JUROS_VAL_MIN   = 695
    HEADER_Y_MAX = 150

    def parse_val(t):
        return float(t.replace('.','').replace(',','.'))

    registros = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for num_pag, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            linhas = {}
            for w in words:
                y = round(w['top'] / 2) * 2
                linhas.setdefault(y, []).append(w)

            ultimo = None
            for y in sorted(linhas.keys()):
                if y < HEADER_Y_MAX:
                    continue
                palavras = linhas[y]
                textos = [w['text'] for w in palavras]

                # Linha de juros
                if 'juros' in textos and ultimo is not None:
                    for w in palavras:
                        if w['x0'] > COL_JUROS_VAL_MIN and w['text'] != 'juros':
                            try:
                                ultimo['juros'] += parse_val(w['text'])
                            except ValueError:
                                pass
                    continue

                seu_num = operacao = valor_final = valor_inicial = None
                pagador_words = []

                for w in palavras:
                    x, t = w['x0'], w['text']
                    if COL_SEU_NUM_MIN <= x <= COL_SEU_NUM_MAX and re.match(r'^\d{9,10}$', t):
                        seu_num = t
                    if COL_OPERACAO_MIN <= x <= COL_OPERACAO_MAX and t not in ('----',):
                        operacao = t
                    if x >= COL_VALOR_FINAL_MIN and re.match(r'^[\d.,]+$', t) and t != '----':
                        try:
                            valor_final = parse_val(t)
                        except ValueError:
                            pass
                    if 45 <= x <= 195:
                        pagador_words.append(t)

                if seu_num and operacao == 'liquidação' and valor_final is not None:
                    reg = {
                        'seu_num':     seu_num,
                        'pagador':     ' '.join(pagador_words),
                        'valor_final': valor_final,
                        'juros':       0.0,
                        'pagina':      num_pag,
                    }
                    for w in palavras:
                        if 460 <= w['x0'] <= 600 and re.match(r'^[\d.,]+$', w['text']):
                            try:
                                reg['valor_inicial'] = parse_val(w['text'])
                            except ValueError:
                                pass
                    if 'valor_inicial' not in reg:
                        reg['valor_inicial'] = None
                    registros.append(reg)
                    ultimo = reg
                else:
                    if not (seu_num and operacao):
                        ultimo = None
    return registros


def ler_csv_itau(file_bytes):
    """Lê o CSV do sistema Itaú (separador ;)."""
    import csv, io
    dados = {}
    text = file_bytes.decode('utf-8', errors='replace')
    reader = csv.DictReader(io.StringIO(text), delimiter=';')
    for row in reader:
        doc = row.get('NUM_DOCUMENTO','').strip().strip('"')
        if not doc:
            continue
        def pv(k):
            try:
                return float(row.get(k,'0').strip().strip('"').replace('.','').replace(',','.'))
            except:
                return 0.0
        dados[doc] = {
            'nome':     row.get('NOME','').strip().strip('"'),
            'parcela':  pv('VLR_PARCELA_MOEDA_CORRENTE'),
            'recebido': pv('VLR_RECEB_MOEDA_CORRENTE'),
        }
    return dados


def render_aba_itau():
    st.markdown("""
    <div style="background:#0a1628; border:1px solid #1e293b; border-radius:12px;
         padding:20px 24px; margin-bottom:24px;">
        <div style="font-family:'IBM Plex Sans',sans-serif; font-weight:700;
             font-size:15px; color:#e2e8f0; margin-bottom:6px;">
            🏦 Conciliação Bancária Itaú — PDF × CSV
        </div>
        <div style="font-size:12px; color:#64748b;">
            Compara o extrato de cobrança do Itaú (PDF) com os recebimentos do sistema (CSV)
            e identifica boletos não lançados e diferenças de valor por juros/mora.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_pdf, col_csv = st.columns(2)
    with col_pdf:
        up_pdf = st.file_uploader("📄 Extrato Itaú (.pdf)", type=["pdf"], key="itau_pdf")
    with col_csv:
        up_csv = st.file_uploader("📋 Recebimentos do sistema (.csv)", type=["csv"], key="itau_csv")

    if not up_pdf or not up_csv:
        st.info("⬆️ Carregue o PDF do banco e o CSV do sistema para iniciar a conciliação.")
        return

    with st.spinner("Lendo PDF do Itaú..."):
        try:
            pdf_regs = extrair_liquidacoes_pdf(up_pdf.read())
        except Exception as e:
            st.error(f"Erro ao ler PDF: {e}")
            return

    with st.spinner("Lendo CSV do sistema..."):
        try:
            csv_dados = ler_csv_itau(up_csv.read())
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            return

    if not pdf_regs:
        st.warning("Nenhuma liquidação encontrada no PDF. Verifique se o arquivo é o extrato de cobrança Itaú correto.")
        return

    # ── Calcular resultados ────────────────────────────────────
    faltantes  = [r for r in pdf_regs if r['seu_num'] not in csv_dados]
    diferencas = []
    for r in pdf_regs:
        if r['seu_num'] in csv_dados:
            diff = round(r['valor_final'] - csv_dados[r['seu_num']]['recebido'], 2)
            if abs(diff) > 0.009:
                diferencas.append({**r,
                    'csv_recebido': csv_dados[r['seu_num']]['recebido'],
                    'diff': diff})

    total_pdf       = round(sum(r['valor_final'] for r in pdf_regs), 2)
    total_csv       = round(sum(v['recebido'] for v in csv_dados.values()), 2)
    total_faltantes = round(sum(r['valor_final'] for r in faltantes), 2)
    total_diff      = round(sum(d['diff'] for d in diferencas), 2)
    gap_total       = round(total_pdf - total_csv, 2)

    # ── Cards de resumo ────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(f'<div class="stat-box"><div class="stat-label">Liquidações PDF</div><div class="stat-value">{len(pdf_regs):,}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stat-box"><div class="stat-label">Registros CSV</div><div class="stat-value">{len(csv_dados):,}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stat-box"><div class="stat-label">Não lançados</div><div class="stat-value {"red" if faltantes else "green"}">{len(faltantes):,}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="stat-box"><div class="stat-label">Com diferença</div><div class="stat-value {"yellow" if diferencas else "green"}">{len(diferencas):,}</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="stat-box"><div class="stat-label">GAP Total</div><div class="stat-value {"red" if abs(gap_total)>0.01 else "green"}" style="font-size:13px;">{brl(gap_total)}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Seção 1: Faltantes ─────────────────────────────────────
    st.markdown("### 1 · Liquidações no banco não encontradas no sistema")
    if faltantes:
        st.markdown(
            f'<div class="alert-retencao">⚠️ <b>{len(faltantes)} boleto(s)</b> liquidados pelo banco '
            f'não estão no CSV do sistema — total de <b>{brl(total_faltantes)}</b> não lançado.</div>',
            unsafe_allow_html=True)
        df_falt = pd.DataFrame([{
            "Seu Número":    r['seu_num'],
            "Pagador":       r['pagador'],
            "Vlr Inicial":   brl(r['valor_inicial']) if r['valor_inicial'] else "—",
            "Vlr Final":     brl(r['valor_final']),
            "Juros":         brl(r['juros']) if r['juros'] > 0 else "—",
            "Página PDF":    r['pagina'],
        } for r in faltantes])
        st.dataframe(df_falt, use_container_width=True, hide_index=True)
        st.markdown(f"**Total não lançado: {brl(total_faltantes)}**")
    else:
        st.success("✔ Nenhum boleto faltante — todos os registros do PDF estão no sistema.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Seção 2: Diferenças de valor ──────────────────────────
    st.markdown("### 2 · Diferenças de valor entre banco e sistema (juros/mora)")
    if diferencas:
        st.markdown(
            f'<div class="alert-multi">⚡ <b>{len(diferencas)} registro(s)</b> com valor diferente '
            f'entre o banco e o sistema — diferença total de <b>{brl(total_diff)}</b>.</div>',
            unsafe_allow_html=True)
        df_diff = pd.DataFrame([{
            "Seu Número":   d['seu_num'],
            "Pagador":      d['pagador'],
            "CSV Recebido": brl(d['csv_recebido']),
            "PDF Vlr Final":brl(d['valor_final']),
            "Diferença":    brl(d['diff']),
            "Página PDF":   d['pagina'],
        } for d in diferencas])
        st.dataframe(df_diff, use_container_width=True, hide_index=True)
        st.markdown(f"**Total das diferenças: {brl(total_diff)}**")
    else:
        st.success("✔ Nenhuma diferença de valor encontrada.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Seção 3: Totais gerais ─────────────────────────────────
    st.markdown("### 3 · Resumo geral")
    df_resumo = pd.DataFrame([
        {"Origem": "PDF (banco)",   "Registros": len(pdf_regs),      "Total": brl(total_pdf)},
        {"Origem": "CSV (sistema)", "Registros": len(csv_dados),     "Total": brl(total_csv)},
        {"Origem": "Não lançados",  "Registros": len(faltantes),     "Total": brl(total_faltantes)},
        {"Origem": "Diferenças",    "Registros": len(diferencas),    "Total": brl(total_diff)},
        {"Origem": "GAP Total",     "Registros": "—",                "Total": brl(gap_total)},
    ])
    st.dataframe(df_resumo, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# ABA 4 — CONCILIAÇÃO ITAÚ
# ══════════════════════════════════════════════════════════════
with aba_itau:
    render_aba_itau()
