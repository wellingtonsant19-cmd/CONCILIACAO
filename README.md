# ⚖️ Motor de Conciliação Financeira — Maxifrota v5

Aplicação Streamlit para conciliação de títulos vencidos.  
Motor v5: solver multi-valor (Saldo, Saldo−IR, Saldo−ISS, Líquido), destaque célula-a-célula, conciliação em massa.

---

## 🚀 Deploy no Streamlit Cloud

1. Fork/push deste repositório
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Selecione `app.py` e deploy

---

## 💻 Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📋 Abas disponíveis

| Aba | Função |
|-----|--------|
| ⚖️ Conciliação por Planilha | Busca individual por Cidade/CNPJ/Nome + valor alvo |
| 📊 Conciliação em Massa BB | Cruza TODOS pagamentos vs títulos automaticamente |
| 🧮 Calculadora | Combinações livres de valores colados |
| 🎯 Liquidação por Retenção | Identifica notas onde retenção ≈ saldo |
| 🏦 Conciliação Itaú | PDF × CSV do Itaú |
| 🏛️ Conciliação BB | Enriquece extrato BB com município |

---

## 🧠 Motor v5 — O que mudou

| Problema v4 | Solução v5 |
|-------------|-----------|
| Solver usava só SALDO | Testa 4 variantes por nota (Saldo, −IR, −ISS, Líquido) |
| Mostrava 28 notas (verde/vermelho) | Mostra só as notas da composição |
| Verde na linha inteira | Verde só nas células usadas (Saldo, IR, ISS) |
| Solução compartilhada entre pagamentos | Cada pagamento tem sua composição |

---

## 🧠 Algoritmo — 5 camadas (busca individual)

| Camada | Técnica | Quando aciona |
|--------|---------|---------------|
| 1 | Pré-processamento | Sempre — reduz base em até 90% |
| 2 | Greedy Descending | Sempre — ms para soluções simples |
| 3 | Meet-in-the-Middle | Se Greedy não resolver |
| 4 | MILP (scipy) | Fallback — Branch & Bound |
| 5 | Ranking inteligente | Sempre — ordena por notas / RBASE / data |

## 🧠 Algoritmo — Conciliação em massa

| Etapa | Técnica |
|-------|---------|
| Seleção | Município exato → fuzzy → histórico vs nome cliente |
| Solver | Subset Sum multi-valor com verificação de unicidade |
| Match | Exato → Aproximado (≤2%) → Ambíguo/Sem match |
| Output | Excel com blocos visuais, destaque por célula |
