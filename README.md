# ⚖️ Motor de Conciliação Financeira — Maxifrota v5

## 📓 Notebooks (Google Colab)

| Notebook | Função | Abrir |
|----------|--------|-------|
| `Conciliacao_Unificada_v5.ipynb` | Busca individual por Cidade/CNPJ/Nome + Valor | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/SEU_REPO/blob/main/Conciliacao_Unificada_v5.ipynb) |
| `Conciliacao_em_Massa_v5.ipynb` | Cruza TODOS pagamentos vs títulos | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU_USUARIO/SEU_REPO/blob/main/Conciliacao_em_Massa_v5.ipynb) |

> Substitua `SEU_USUARIO/SEU_REPO` pela URL do seu repositório GitHub.

## 🌐 App Streamlit

| Aba | Função |
|-----|--------|
| 🧮 Calculadora | Combinações livres de valores colados |
| 🎯 Liquidação por Retenção | Notas onde retenção ≈ saldo |
| 🏦 Conciliação Itaú | PDF × CSV do Itaú |
| 🏛️ Conciliação BB | Enriquece extrato BB com município |

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Motor v5 — Solver MILP multi-valor

Testa 4 variantes por nota:
- 🟢 **Saldo** — valor integral
- 🟡 **Saldo − IR** — pagador deduziu IR
- 🟡 **Saldo − ISS** — pagador deduziu ISS
- 🔴 **Líquido** — pagador deduziu IR + ISS

Restrição: máximo 1 variante por nota. Solver: scipy/HiGHS (MILP).
