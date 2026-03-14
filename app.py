import streamlit as st
import pandas as pd
import numpy as np
import time
import re as _re
import unicodedata as _unicodedata
import io

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(page_title="Motor de Conciliação", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=IBM+Plex+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }
.header-box {
    background: linear-gradient(135deg, #0f2044 0%, #0a1628 100%);
    border: 1px solid #1e3a5f; border-radius: 12px; padding: 24px 32px; margin-bottom: 28px;
}
.stat-box {
    background: #0a1628; border: 1px solid #1e293b; border-radius: 10px;
    padding: 16px 20px; text-align: center;
}
.stat-label { font-size: 10px; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.stat-value { font-size: 22px; font-weight: 800; color: #e2e8f0; font-family: 'IBM Plex Mono', monospace; }
.stat-value.green  { color: #4ade80; }
.stat-value.yellow { color: #f59e0b; }
.stat-value.red    { color: #f87171; }
.alert-retencao { background: #ef444410; border: 1px solid #ef444430; border-radius: 8px; padding: 14px 18px; color: #fca5a5; font-size: 13px; margin-top: 12px; }
.alert-multi { background: #f59e0b10; border: 1px solid #f59e0b30; border-radius: 8px; padding: 12px 16px; color: #fcd34d; font-size: 13px; margin-bottom: 16px; }
div[data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

def brl(v):
    try: return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except: return "—"

# ============================================================
# ALGORITMO — para Calculadora de Combinações
# ============================================================

TOLERANCIA = 0.01
MAX_NOTAS  = 30
TOP_RESULTADOS = 10

def preprocessar(valores, alvo):
    limite = round(alvo + TOLERANCIA, 2)
    return dict(sorted({k: v for k, v in valores.items() if 0 < round(v, 2) <= limite}.items(), key=lambda x: x[1], reverse=True))

def greedy(valores, alvo, max_notas):
    alvo_r = round(alvo, 2); soma, combo = 0.0, []
    for chave, val in valores.items():
        if len(combo) >= max_notas: break
        if round(soma + val, 2) <= round(alvo_r + TOLERANCIA, 2):
            soma = round(soma + val, 2); combo.append(chave)
        if abs(soma - alvo_r) <= TOLERANCIA:
            return (round(abs(soma - alvo_r), 2), tuple(combo))
    return None

def _somas_grupo(items, max_n, cap=None):
    dp = {0: [()]}
    MAX_DP_SIZE = 50_000
    for i, (chave, val) in enumerate(items):
        new_e = {}
        for sa, combos in dp.items():
            nv = sa + val
            novos = [c + (chave,) for c in combos if len(c) < max_n]
            if novos:
                if nv not in new_e: new_e[nv] = []
                new_e[nv].extend(novos)
        for nv, novos in new_e.items():
            if nv not in dp: dp[nv] = novos[:cap] if cap else novos
            else:
                merged = dp[nv] + novos
                if cap and len(merged) > cap: merged.sort(key=len); merged = merged[:cap]
                dp[nv] = merged
        if len(dp) > MAX_DP_SIZE: break
    return dp

def meet_in_the_middle(valores, alvo, max_notas):
    n = len(valores)
    if   n <= 20: cap = None
    elif n <= 35: cap = 5
    elif n <= 60: cap = 2
    else:         cap = 1
    alvo_cents = round(round(alvo, 2) * 100)
    tol_cents  = round(TOLERANCIA * 100)
    items = [(k, round(round(v, 2) * 100)) for k, v in valores.items() if round(v, 2) > 0]
    if not items: return []
    mid = len(items) // 2
    left, right = items[:mid], items[mid:]
    max_l = max(1, max_notas * len(left) // len(items))
    max_r = max(1, max_notas * len(right) // len(items))
    somas_l = _somas_grupo(left, max_l, cap)
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
                    dif = round(abs(alvo_cents - total) / 100, 2)
                    resultados.append((dif, tuple(sorted(merged))))
    resultados.sort(key=lambda x: (x[0], len(x[1])))
    return resultados[:TOP_RESULTADOS * 3]

def solver_milp_simples(valores, alvo, max_notas):
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        keys = list(valores.keys()); vals = np.array([round(valores[k], 2) for k in keys])
        n = len(keys)
        c1 = LinearConstraint(vals.reshape(1, -1), lb=alvo - TOLERANCIA, ub=alvo + TOLERANCIA)
        c2 = LinearConstraint(np.ones((1, n)), lb=0, ub=max_notas)
        res = milp(np.ones(n), constraints=[c1, c2], integrality=np.ones(n),
                   bounds=Bounds(lb=np.zeros(n), ub=np.ones(n)))
        if res.success and res.x is not None:
            xi = np.round(res.x).astype(int)
            combo = tuple(keys[i] for i in range(n) if xi[i] == 1)
            soma = round(sum(valores[k] for k in combo), 2)
            return [(round(abs(soma - round(alvo, 2)), 2), combo)]
    except: pass
    return []

# ============================================================
# CALCULADORA DE COMBINAÇÕES
# ============================================================

def _parse_num(tok):
    tok = tok.strip().replace("\xa0", "").replace(" ", "")
    tok = _re.sub(r"[R$]", "", tok)
    if tok in ("-", "–", "—", ""): return 0.0
    if _re.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$", tok):
        tok = tok.replace(".", "").replace(",", ".")
    else: tok = tok.replace(",", ".")
    try: return float(tok)
    except: return None

def _parse_valores_livres(texto, modo="saldo"):
    rows = []
    for line in texto.splitlines():
        cols = _re.split(r"\t", line) if "\t" in line else _re.split(r"[;|]", line)
        parsed = [v for c in cols if (v := _parse_num(c)) is not None]
        if not parsed: continue
        val = round(sum(parsed), 2) if modo == "liquido" else next((v for v in parsed if v > 0), None)
        if val is None: continue
        val = round(val, 2)
        if val > 0: rows.append(val)
    return rows

def _buscar_combinacoes_livres(valores_lista, alvo, max_n, top):
    valores = {str(i): v for i, v in enumerate(valores_lista)}
    valores = preprocessar(valores, alvo)
    if not valores: return []
    brutos = []
    g = greedy(valores, alvo, max_n)
    if g: brutos.append(g)
    brutos.extend(meet_in_the_middle(valores, alvo, max_n))
    if not brutos: brutos.extend(solver_milp_simples(valores, alvo, max_n))
    vistas, resultado = set(), []
    for dif, combo in brutos:
        parcelas = tuple(sorted(round(float(valores.get(k, 0)), 2) for k in combo))
        if parcelas in vistas: continue
        vistas.add(parcelas)
        resultado.append((dif, round(sum(parcelas), 2), sorted(combo, key=lambda x: int(x))))
    resultado.sort(key=lambda x: (x[0], len(x[2])))
    return resultado[:top]

def aba_calculadora():
    st.markdown("""
    <div style="background:#0a1628;border:1px solid #1e293b;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
        <div style="font-weight:700;font-size:15px;color:#e2e8f0;margin-bottom:6px;">🧮 Calculadora de Combinações</div>
        <div style="font-size:12px;color:#64748b;">Cole valores, informe o alvo e encontre combinações que somam exatamente esse valor.</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        texto_vals = st.text_area("Cole os valores aqui", height=220,
            placeholder="Um por linha:\n124.170,17\n56.307,30\n47.724,75", key="calc_texto")
        tem_tabs = "\t" in (texto_vals or "")
        if tem_tabs:
            modo = st.radio("Valor da linha:", ["Saldo bruto", "Líquido (Saldo+IR+ISS)"], horizontal=True, key="calc_modo")
            modo_key = "liquido" if "Líquido" in modo else "saldo"
            if texto_vals.strip():
                preview = _parse_valores_livres(texto_vals, modo=modo_key)
                if preview: st.caption(f"✅ {len(preview)} valores — soma: {brl(sum(preview))}")
        else: modo_key = "saldo"
    with c2:
        alvo_str = st.text_input("Valor alvo (R$)", placeholder="Ex: 436.611,53", key="calc_alvo")
        max_n = st.slider("Máx. parcelas", 1, 100, 8, key="calc_maxn")
        top_n = st.slider("Máx. resultados", 1, 20, 10, key="calc_top")
        st.markdown("<br>", unsafe_allow_html=True)
        calc_btn = st.button("🔍 Calcular", use_container_width=True, type="primary", key="calc_btn")

    if calc_btn:
        if not texto_vals.strip(): st.error("Cole pelo menos um valor."); return
        if not alvo_str.strip(): st.error("Informe o valor alvo."); return
        try: alvo = float(alvo_str.replace(".", "").replace(",", "."))
        except: st.error("Valor inválido."); return
        vals = _parse_valores_livres(texto_vals, modo=modo_key)
        if not vals: st.error("Nenhum valor reconhecido."); return
        st.markdown("---")
        s1,s2,s3,s4 = st.columns(4)
        s1.markdown(f'<div class="stat-box"><div class="stat-label">Valores</div><div class="stat-value">{len(vals)}</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-box"><div class="stat-label">Soma</div><div class="stat-value" style="font-size:15px;">{brl(sum(vals))}</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-box"><div class="stat-label">Alvo</div><div class="stat-value green" style="font-size:15px;">{brl(alvo)}</div></div>', unsafe_allow_html=True)
        d = "green" if abs(sum(vals)-alvo)<0.01 else "yellow"
        s4.markdown(f'<div class="stat-box"><div class="stat-label">Diferença</div><div class="stat-value {d}" style="font-size:15px;">{brl(round(sum(vals)-alvo,2))}</div></div>', unsafe_allow_html=True)
        with st.spinner("Buscando..."):
            t0 = time.time()
            resultados = _buscar_combinacoes_livres(vals, alvo, max_n, top_n)
            elapsed = round((time.time() - t0) * 1000)
        if not resultados: st.error("Nenhuma combinação encontrada."); return
        st.markdown(f'<div class="alert-multi">⚡ <b>{len(resultados)} combinação(ões)</b> em <b>{elapsed}ms</b></div>', unsafe_allow_html=True)
        for i, (dif, soma, indices) in enumerate(resultados, 1):
            parcelas = [vals[int(idx)] for idx in indices]
            is_best = (i == 1)
            brd = "1.5px solid #3b82f6" if is_best else "1px solid #1e293b"
            bg_h = "#1e3a5f" if is_best else "#0a0f1a"
            cor_s = "#4ade80" if dif == 0 else "#f59e0b"
            st.markdown(
                f"<div style='border:{brd};border-radius:12px;overflow:hidden;margin-bottom:12px;'>"
                f"<div style='background:{bg_h};padding:10px 16px;display:flex;align-items:center;gap:10px;'>"
                f"<b style='color:#fff;font-family:monospace;'>#{i}</b>"
                f"<span style='margin-left:auto;font-size:12px;color:#94a3b8;font-family:monospace;'>"
                f"{len(parcelas)} parcela(s) | Soma: <b style='color:{cor_s};'>{brl(soma)}</b></span>"
                f"</div></div>", unsafe_allow_html=True)
            df_p = pd.DataFrame({"Nº": [int(idx)+1 for idx in indices]+["TOTAL"], "Valor": [brl(v) for v in parcelas]+[brl(soma)]})
            st.dataframe(df_p, use_container_width=True, hide_index=True)


# ============================================================
# LIQUIDAÇÃO POR RETENÇÃO
# ============================================================

def aba_retencao():
    st.markdown("""
    <div style="background:#0a1628;border:1px solid #1e293b;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
        <div style="font-weight:700;font-size:15px;color:#e2e8f0;margin-bottom:6px;">🎯 Notas Candidatas a Liquidação</div>
        <div style="font-size:12px;color:#64748b;">Identifica notas onde o saldo ≈ total retido (IR + ISS).</div>
    </div>""", unsafe_allow_html=True)
    up = st.file_uploader("📂 Planilha de títulos (.xlsx)", type=["xlsx","xls"], key="ret_upload")
    if not up: st.info("⬆️ Carregue a planilha."); return
    df = pd.read_excel(up); df.columns = df.columns.str.strip()
    MAP = {"NF":["NF","NR NFEM"],"NOME":["NOME CLIENTE","NOME"],"CIDADE":["CIDADE"],"CNPJ":["CNPJ"],
           "VLR SALDO":["SALDO","VLR SALDO"],"IR":["IR","VLR RETIDO"],"ISS":["ISS","ISS RETIDO"],
           "VENC":["VENCIMENTO","VENC","DT VENCIMENTO"],"ATRASO":["ATRASO"]}
    cu = {c.upper(): c for c in df.columns}
    for dest, cands in MAP.items():
        if dest not in df.columns:
            for cand in cands:
                if cand.upper() in cu: df.rename(columns={cu[cand.upper()]: dest}, inplace=True); break
    for c in ["VLR SALDO","IR","ISS"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(2)
        else: df[c] = 0.0
    df = df[df["VLR SALDO"] > 0].copy()
    df["RETENCAO"] = (df["IR"].abs() + df["ISS"].abs()).round(2)
    df = df[df["RETENCAO"] > 0].copy()
    df["DIFERENÇA"] = (df["VLR SALDO"] - df["RETENCAO"]).abs().round(2)
    tol = st.slider("Tolerância (R$)", 0.0, 50.0, 10.0, 0.5, format="R$ %.2f", key="ret_tol")
    df_cand = df[df["DIFERENÇA"] <= tol].sort_values("DIFERENÇA")
    st.markdown("---")
    m1,m2,m3,m4 = st.columns(4)
    m1.markdown(f'<div class="stat-box"><div class="stat-label">Candidatas</div><div class="stat-value green">{len(df_cand):,}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stat-box"><div class="stat-label">Saldo</div><div class="stat-value" style="font-size:13px;">{brl(df_cand["VLR SALDO"].sum())}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stat-box"><div class="stat-label">Retido</div><div class="stat-value" style="font-size:13px;">{brl(df_cand["RETENCAO"].sum())}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="stat-box"><div class="stat-label">Ajuste</div><div class="stat-value yellow" style="font-size:13px;">{brl(df_cand["DIFERENÇA"].sum())}</div></div>', unsafe_allow_html=True)
    if df_cand.empty: st.warning(f"Nenhuma nota com diferença ≤ R$ {tol:.2f}."); return
    COLS_E = ["NF","NOME","CIDADE","VLR SALDO","IR","ISS","RETENCAO","DIFERENÇA","VENC","ATRASO"]
    cols_ok = [c for c in COLS_E if c in df_cand.columns]
    df_e = df_cand[cols_ok].copy(); df_exp = df_e.copy()
    for col in ["VLR SALDO","IR","ISS","RETENCAO","DIFERENÇA"]:
        if col in df_e.columns: df_e[col] = df_e[col].apply(lambda v: brl(v) if pd.notna(v) else "")
    st.dataframe(df_e, use_container_width=True, hide_index=True, height=480)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w: df_exp.to_excel(w, index=False, sheet_name="Liquidação")
    buf.seek(0)
    st.download_button("⬇️ Exportar", buf, "liquidacao_retencao.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


# ============================================================
# CONCILIAÇÃO ITAÚ (PDF × CSV)
# ============================================================

def extrair_liquidacoes_pdf(file_bytes):
    try: import pdfplumber
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "--break-system-packages", "-q"])
        import pdfplumber
    COL_SEU_NUM_MIN=280; COL_SEU_NUM_MAX=360; COL_OPERACAO_MIN=620; COL_OPERACAO_MAX=760
    COL_VALOR_FINAL_MIN=760; COL_JUROS_VAL_MIN=695; HEADER_Y_MAX=150
    def pv(t): return float(t.replace('.','').replace(',','.'))
    registros = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for np_, page in enumerate(pdf.pages, 1):
            words = page.extract_words(); linhas = {}
            for w in words: y = round(w['top']/2)*2; linhas.setdefault(y,[]).append(w)
            ultimo = None
            for y in sorted(linhas.keys()):
                if y < HEADER_Y_MAX: continue
                palavras = linhas[y]; textos = [w['text'] for w in palavras]
                if 'juros' in textos and ultimo:
                    for w in palavras:
                        if w['x0'] > COL_JUROS_VAL_MIN and w['text'] != 'juros':
                            try: ultimo['juros'] += pv(w['text'])
                            except: pass
                    continue
                seu_num=op=vf=None; pw=[]
                for w in palavras:
                    x,t = w['x0'],w['text']
                    if COL_SEU_NUM_MIN<=x<=COL_SEU_NUM_MAX and _re.match(r'^\d{9,10}$',t): seu_num=t
                    if COL_OPERACAO_MIN<=x<=COL_OPERACAO_MAX and t not in ('----',): op=t
                    if x>=COL_VALOR_FINAL_MIN and _re.match(r'^[\d.,]+$',t) and t!='----':
                        try: vf=pv(t)
                        except: pass
                    if 45<=x<=195: pw.append(t)
                if seu_num and op=='liquidação' and vf is not None:
                    reg = {'seu_num':seu_num,'pagador':' '.join(pw),'valor_final':vf,'juros':0.0,'pagina':np_}
                    for w in palavras:
                        if 460<=w['x0']<=600 and _re.match(r'^[\d.,]+$',w['text']):
                            try: reg['valor_inicial']=pv(w['text'])
                            except: pass
                    if 'valor_inicial' not in reg: reg['valor_inicial']=None
                    registros.append(reg); ultimo=reg
                else:
                    if not (seu_num and op): ultimo=None
    return registros

def ler_csv_itau(file_bytes):
    import csv
    dados = {}
    text = file_bytes.decode('utf-8', errors='replace')
    for row in csv.DictReader(io.StringIO(text), delimiter=';'):
        doc = row.get('NUM_DOCUMENTO','').strip().strip('"')
        if not doc: continue
        def p(k):
            try: return float(row.get(k,'0').strip().strip('"').replace('.','').replace(',','.'))
            except: return 0.0
        dados[doc] = {'nome':row.get('NOME','').strip().strip('"'),'parcela':p('VLR_PARCELA_MOEDA_CORRENTE'),'recebido':p('VLR_RECEB_MOEDA_CORRENTE')}
    return dados

def render_aba_itau():
    st.markdown("""
    <div style="background:#0a1628;border:1px solid #1e293b;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
        <div style="font-weight:700;font-size:15px;color:#e2e8f0;margin-bottom:6px;">🏦 Conciliação Itaú — PDF × CSV</div>
        <div style="font-size:12px;color:#64748b;">Compara extrato Itaú (PDF) com recebimentos (CSV).</div>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: up_pdf = st.file_uploader("📄 Extrato (.pdf)", type=["pdf"], key="itau_pdf")
    with c2: up_csv = st.file_uploader("📋 Recebimentos (.csv)", type=["csv"], key="itau_csv")
    if not up_pdf or not up_csv: st.info("⬆️ Carregue PDF e CSV."); return
    with st.spinner("Lendo PDF..."):
        try: pdf_regs = extrair_liquidacoes_pdf(up_pdf.read())
        except Exception as e: st.error(f"Erro PDF: {e}"); return
    with st.spinner("Lendo CSV..."):
        try: csv_dados = ler_csv_itau(up_csv.read())
        except Exception as e: st.error(f"Erro CSV: {e}"); return
    if not pdf_regs: st.warning("Nenhuma liquidação no PDF."); return
    faltantes = [r for r in pdf_regs if r['seu_num'] not in csv_dados]
    diferencas = []
    for r in pdf_regs:
        if r['seu_num'] in csv_dados:
            diff = round(r['valor_final']-csv_dados[r['seu_num']]['recebido'],2)
            if abs(diff) > 0.009: diferencas.append({**r,'csv_recebido':csv_dados[r['seu_num']]['recebido'],'diff':diff})
    gap = round(sum(r['valor_final'] for r in pdf_regs) - sum(v['recebido'] for v in csv_dados.values()),2)
    st.markdown("---")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.markdown(f'<div class="stat-box"><div class="stat-label">PDF</div><div class="stat-value">{len(pdf_regs)}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stat-box"><div class="stat-label">CSV</div><div class="stat-value">{len(csv_dados)}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stat-box"><div class="stat-label">Faltantes</div><div class="stat-value {"red" if faltantes else "green"}">{len(faltantes)}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="stat-box"><div class="stat-label">Diferenças</div><div class="stat-value {"yellow" if diferencas else "green"}">{len(diferencas)}</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="stat-box"><div class="stat-label">GAP</div><div class="stat-value {"red" if abs(gap)>0.01 else "green"}" style="font-size:13px;">{brl(gap)}</div></div>', unsafe_allow_html=True)
    st.markdown("### 1 · Faltantes")
    if faltantes:
        st.dataframe(pd.DataFrame([{"Seu Número":r['seu_num'],"Pagador":r['pagador'],"Vlr Final":brl(r['valor_final'])} for r in faltantes]), use_container_width=True, hide_index=True)
    else: st.success("✔ Nenhum faltante.")
    st.markdown("### 2 · Diferenças")
    if diferencas:
        st.dataframe(pd.DataFrame([{"Seu Número":d['seu_num'],"CSV":brl(d['csv_recebido']),"PDF":brl(d['valor_final']),"Diff":brl(d['diff'])} for d in diferencas]), use_container_width=True, hide_index=True)
    else: st.success("✔ Sem diferenças.")


# ============================================================
# CONCILIAÇÃO BB — Enriquecimento de Extrato
# ============================================================

_STOP = {"MUN","MUNIC","MUNICIPAL","MUNICIPIO","PREFEITURA","PREF","FUNDO","SAUDE","SECRETARIA","CAMARA","SERVICO","SAMAE","FMS","SME","PM","CME","DE","DO","DA","DOS","DAS"}
_AMBIG = {"SAUDE","ITA","CUSTODIA","BELA","VERDE","ALEGRE","NOVA","ALTO","ALTA","BOA","SOL","MAR"}

def _bb_norm(t):
    if not t: return ""
    t = str(t).upper().strip()
    t = _unicodedata.normalize("NFD", t)
    return "".join(c for c in t if _unicodedata.category(c) != "Mn")

def _bb_extract_cnpj(t):
    if not t: return None
    s = str(t)
    m = _re.search(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", s)
    if m: return _re.sub(r"\D", "", m.group())
    for tok in s.split():
        clean = _re.sub(r"[.\-/]", "", tok)
        if _re.match(r"^\d{14}$", clean): return clean
    return None

def _bb_consultar(cnpj, cache):
    if cnpj in cache: return cache[cnpj]
    try:
        import requests
        r = requests.get(f"https://brasilapi.com.br/api/cnpj/v1/{cnpj}", timeout=8)
        result = (r.json().get("municipio","").upper(), r.json().get("uf","").upper()) if r.status_code == 200 else ("","")
    except: result = ("","")
    cache[cnpj] = result; time.sleep(0.3)
    return result

@st.cache_data(show_spinner=False)
def _bb_carregar_ag(fb, fn):
    df = pd.read_excel(io.BytesIO(fb), sheet_name="AGENCIAS")
    am, ml, seen = {}, [], set()
    for _, row in df.iterrows():
        if not pd.notna(row.get("AGENCIA")): continue
        try: ag = str(int(float(str(row["AGENCIA"])))).zfill(4)
        except: ag = str(row["AGENCIA"]).strip().zfill(4)
        mun = str(row["MUNICIPIO"]).strip().upper() if pd.notna(row.get("MUNICIPIO")) else ""
        uf = str(row["UF"]).strip().upper() if pd.notna(row.get("UF")) else ""
        if ag and mun: am[ag] = (mun, uf)
        if mun:
            n = _bb_norm(mun)
            if n not in seen: seen.add(n); ml.append((n, mun, uf))
    ml.sort(key=lambda x: -len(x[0]))
    return am, ml

def _bb_buscar_mun(texto, ml):
    if not texto: return "", ""
    n = _bb_norm(texto)
    n = _re.sub(r"^\d{2}/\d{2}\s+\d{2}:\d{2}\s+", "", n)
    for nm, mun, uf in ml:
        if len(nm) < 5 or nm in _AMBIG: continue
        if _re.search(r"(?<![A-Z])"+_re.escape(nm)+r"(?![A-Z])", n): return mun, uf
    return "", ""

@st.cache_data(show_spinner=False)
def _bb_ler_ext(fb, fn):
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(fb)); ws = wb.active
    hr, hv = None, []
    for i, row in enumerate(ws.iter_rows(values_only=True), 1):
        if len([v for v in row if v is not None]) >= 3: hr, hv = i, list(row); break
    if hr is None: raise ValueError("Cabeçalho não encontrado.")
    rd = [list(row) for i, row in enumerate(ws.iter_rows(values_only=True), 1) if i > hr and any(v is not None for v in row)]
    cn, sc = [], {}
    for v in hv:
        nm = str(v).strip().upper() if v else "COL"
        if nm in sc: sc[nm] += 1; nm = f"{nm}_{sc[nm]}"
        else: sc[nm] = 0
        cn.append(nm)
    df = pd.DataFrame(rd, columns=cn)
    df = df[df.apply(lambda r: any(str(v).strip() not in ("","None","nan","NAN") for v in r), axis=1)].copy()
    df.index = range(len(df))
    ca = next((c for c in df.columns if "AGENCI" in c), None)
    cd = next((c for c in df.columns if "DETAL" in c), None) or next((c for c in df.columns if "HISTOR" in c), None)
    def nag(v):
        if v is None or str(v).strip() in ("","nan","None"): return "0000"
        try: return str(int(float(str(v).strip()))).zfill(4)
        except: return str(v).strip().zfill(4)
    df["_AG"] = df[ca].apply(nag) if ca else "0000"
    df["_DET"] = df[cd].fillna("").astype(str).str.strip().str.upper() if cd else ""
    return df

def render_aba_bb():
    st.markdown("""
    <div style="background:#0a1628;border:1px solid #1e293b;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
        <div style="font-weight:700;font-size:15px;color:#e2e8f0;margin-bottom:6px;">🏛️ Conciliação BB — Extrato</div>
        <div style="font-size:12px;color:#64748b;">Enriquece extrato BB com UF e Município.</div>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: ue = st.file_uploader("📄 Extrato BB (.xlsx)", type=["xlsx","xls"], key="bb_ext")
    with c2: ua = st.file_uploader("📋 Agências (.xlsx)", type=["xlsx","xls"], key="bb_ag")
    if not ue or not ua: st.info("⬆️ Carregue extrato e agências."); return
    with st.spinner("Carregando..."): am, ml = _bb_carregar_ag(ua.read(), ua.name)
    with st.spinner("Lendo extrato..."): df = _bb_ler_ext(ue.read(), ue.name)
    st.markdown("---")
    m1,m2,m3 = st.columns(3)
    m1.markdown(f'<div class="stat-box"><div class="stat-label">Agências</div><div class="stat-value">{len(am)}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stat-box"><div class="stat-label">Municípios</div><div class="stat-value">{len(ml)}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stat-box"><div class="stat-label">Lançamentos</div><div class="stat-value green">{len(df)}</div></div>', unsafe_allow_html=True)
    if not st.button("🔍 Processar", type="primary", use_container_width=True, key="bb_proc"): return
    df["UF"]=""; df["MUNICIPIO"]=""
    cc={}; nt=nc=na=nv=0
    bar = st.progress(0)
    for i, (idx, row) in enumerate(df.iterrows()):
        det, ag = row["_DET"], row["_AG"]
        mun, uf = _bb_buscar_mun(det, ml)
        if mun: nt += 1
        if not mun:
            cnpj = _bb_extract_cnpj(det)
            if cnpj: mun, uf = _bb_consultar(cnpj, cc);
            if mun: nc += 1
        if not mun: mun, uf = am.get(ag, ("","")); na += 1 if mun else 0
        if not mun: mun = "VERIFICAR"; nv += 1
        df.at[idx,"UF"]=uf; df.at[idx,"MUNICIPIO"]=mun
        bar.progress(int((i+1)/len(df)*100))
    bar.empty()
    st.markdown("---")
    r1,r2,r3,r4 = st.columns(4)
    r1.markdown(f'<div class="stat-box"><div class="stat-label">Texto</div><div class="stat-value green">{nt}</div></div>', unsafe_allow_html=True)
    r2.markdown(f'<div class="stat-box"><div class="stat-label">CNPJ</div><div class="stat-value green">{nc}</div></div>', unsafe_allow_html=True)
    r3.markdown(f'<div class="stat-box"><div class="stat-label">Agência</div><div class="stat-value">{na}</div></div>', unsafe_allow_html=True)
    r4.markdown(f'<div class="stat-box"><div class="stat-label">Verificar</div><div class="stat-value {"red" if nv else "green"}">{nv}</div></div>', unsafe_allow_html=True)
    oc = [c for c in df.columns if not c.startswith("_") and c not in ("UF","MUNICIPIO")]
    st.dataframe(df[(oc[:5]+["UF","MUNICIPIO"])].head(100), use_container_width=True, hide_index=True, height=400)


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

st.markdown("""
<div class="header-box">
    <div style="display:flex;align-items:center;gap:16px;">
        <div style="font-size:36px;">⚖️</div>
        <div>
            <div style="font-weight:800;font-size:22px;letter-spacing:-0.02em;color:#f1f5f9;">Motor de Conciliação Financeira</div>
            <div style="font-size:12px;color:#475569;letter-spacing:0.05em;margin-top:2px;">MAXIFROTA — MOTOR v5</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;padding:12px 18px;margin-bottom:20px;font-size:12px;color:#94a3b8;">
    📓 <b>Conciliação Unificada</b> e <b>Conciliação em Massa</b> agora rodam no
    <a href="https://colab.research.google.com/" style="color:#60a5fa;">Google Colab</a> (sem limite de memória/tempo).
    Acesse os notebooks no repositório GitHub.
</div>
""", unsafe_allow_html=True)

aba_calc, aba_ret, aba_itau, aba_bb = st.tabs([
    "🧮  Calculadora de Combinações",
    "🎯  Liquidação por Retenção",
    "🏦  Conciliação Itaú",
    "🏛️  Conciliação BB",
])

with aba_calc:
    aba_calculadora()

with aba_ret:
    aba_retencao()

with aba_itau:
    render_aba_itau()

with aba_bb:
    render_aba_bb()
