# ⚖️ Motor de Conciliação Financeira — Maxifrota

Aplicação Streamlit para conciliação de títulos vencidos.  
Identifica a composição de notas fiscais que compõe um valor recebido, detecta retenções indevidas de IR/ISS e sinaliza quando existem múltiplas composições possíveis.

---

## 🚀 Deploy no Streamlit Cloud

1. Faça fork ou push deste repositório para o seu GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app**
4. Selecione o repositório e o arquivo `app.py`
5. Clique em **Deploy**

---

## 💻 Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Algoritmo — 5 camadas

| Camada | Técnica | Quando aciona |
|--------|---------|---------------|
| 1 | Pré-processamento | Sempre — reduz base em até 90% |
| 2 | Greedy Descending | Sempre — ms para soluções simples |
| 3 | Meet-in-the-Middle | Se Greedy não resolver |
| 4 | MILP (scipy) | Fallback — Branch & Bound |
| 5 | Ranking inteligente | Sempre — ordena por notas / RBASE / data |

---

## 📋 Colunas esperadas na planilha

| Coluna | Alternativa | Descrição |
|--------|-------------|-----------|
| `NOME` | — | Nome do cliente |
| `CNPJ` | — | CNPJ (aceita com ou sem formatação) |
| `CIDADE` | — | Cidade do cliente |
| `VLR SALDO` | `VALOR SALDO` | Saldo em aberto da nota |
| `VLR TITULO` | `VALOR TITULO` | Valor integral da nota |
| `IR` | — | IR (negativo = dedução) |
| `ISS` | — | ISS |
| `RBASE RAIZ` | `RBASE` | Base de cálculo |
| `NF` | — | Número da nota fiscal |
| `VENC` | — | Data de vencimento |
| `ATRASO` | — | Dias de atraso |
