# 📊 RAG Statistical Evaluation & Robustness Auditing Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Statistical Rigor: ECE | McNemar | Bootstrap](https://img.shields.io/badge/Statistical_Rigor-ECE_%7C_McNemar_%7C_Bootstrap-emerald?style=for-the-badge)](tests/)
[![Tests: Pytest Passing](https://img.shields.io/badge/Tests-7%20Passed%20(Pytest)-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Live Interactive Benchmark](https://img.shields.io/badge/Live_Showcase-GitHub_Pages-6366f1?style=for-the-badge&logo=googlechrome&logoColor=white)](https://geovana28.github.io/rag-stat-eval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Suíte Científica e Computacional para Auditoria Estatística de Pipelines RAG (*Retrieval-Augmented Generation*)**: Implementação formal de calibração de incerteza (ECE/MCE), testes de significância pareados (McNemar com correção de Edwards e Bootstrap pareado), cálculo assintótico de poder estatístico ($1 - \beta$) e degradação sob perturbação adversarial de ruído.

---

## 🎯 1. Problema Científico e Tecnológico Investigado

Na literatura recente e na indústria de Inteligência Artificial, avaliações de pipelines RAG frequentemente incorrem no erro de reportar apenas métricas agregadas pontuais (ex: *"acurácia de 85% vs 82%"*). No entanto:
1. **Incerteza Não-Calibrada**: Modelos de linguagem frequentemente exibem **sobreconfiança sistemática** (*overconfidence*), reportando 95% de probabilidade em respostas incorretas.
2. **Falta de Significância Estatística**: Uma diferença de 3% em benchmarks pequenos pode ser mero ruído amostral ($p > 0.05$), sem poder estatístico suficiente para justificar a substituição de um modelo em produção.
3. **Fragilidade Adversarial**: Falta de auditoria sobre a velocidade com que a acurácia colapsa sob ruído ortográfico realista.

Esta suíte fornece um framework matemático formal para auditar e responder a essas três questões com rigor acadêmico.

---

## 🔬 2. Hipóteses & Formulação Matemática

### A. Teste de Hipótese Pareado (McNemar com Correção de Continuidade de Edwards)
Para avaliar se o **Modelo Champion** é significativamente superior ao **Modelo Baseline** sobre o mesmo conjunto pareado de $N$ perguntas:
* **Hipótese Nula ($H_0$)**: $p_{01} = p_{10}$ (A probabilidade de apenas o Modelo A acertar é igual à probabilidade de apenas o Modelo B acertar).
* **Hipótese Alternativa ($H_1$)**: $p_{01} \neq p_{10}$ (Existe divergência sistemática).

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}, \quad \text{com } \text{df} = 1$$

*Quando $n_{01} + n_{10} < 25$, a suíte aciona automaticamente o teste binomial exato bilateral.*

### B. Cálculo Analítico de Poder Estatístico ($1 - \beta$)
A suíte computa o poder do teste de McNemar com base no tamanho amostral $N$, na taxa de discordância $p_{\text{disc}} = \frac{n_{01} + n_{10}}{N}$ e no Odds Ratio $\psi = \frac{n_{01}}{n_{10}}$:

$$Z_\beta = \frac{|\Delta| - Z_{1 - \alpha/2} \sqrt{\frac{p_{\text{disc}}}{N}}}{\sqrt{\frac{p_{\text{disc}} - \Delta^2}{N}}}, \quad \text{Poder } (1 - \beta) = \Phi(Z_\beta)$$

### C. Reamostragem por Bootstrap Pareado (2.000 iterações)
Gera intervalos de confiança empíricos a 95% ($CI_{95\%}$) para a diferença de desempenho $\Delta = \bar{X}_{\text{Champion}} - \bar{X}_{\text{Baseline}}$. A superioridade é atestada se e somente se:

$$0 \notin [CI_{2.5\%}, CI_{97.5\%}]$$

### D. Calibração de Incerteza: ECE e MCE
Particionando as predições em $M=10$ bins equiprováveis $B_m \subset (0, 1]$:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|, \quad \text{MCE} = \max_{m \in \{1,\dots,M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 📂 3. Origem dos Dados & Desenho Experimental

* **Estrutura Amostral**: Benchmark padronizado de $N=500$ consultas técnicas pareadas divididas em 4 domínios de alta criticidade regulatória:
  1. *Clinical Healthcare* (Diretrizes médicas e dosagens)
  2. *Financial Compliance* (Normas AML/KYC e Basileia III)
  3. *Legal Contracts* (Cláusulas de SLA e responsabilidade civil)
  4. *Cloud Infrastructure* (Consenso distribuído e failover)
* **Controle Experimental**: Avaliação pareada sobre exatamente os mesmos prompts e contextos, garantindo variância reduzida no pareamento estatístico.

---

## 📊 4. Resultados Experimentais Medidos

| Métrica Estatística | Modelo Baseline | Modelo Champion | Conclusão Estatística |
| :--- | :---: | :---: | :---: |
| **Acurácia Média** | $74.2\%$ | $83.6\%$ | Diferença observada $\Delta = +9.4\%$ |
| **Intervalo Bootstrap ($CI_{95\%}$)** | - | $[+5.8\%, +13.0\%]$ | **Estatisticamente Significativo** ($0 \notin CI$) |
| **Estatística $\chi^2$ (McNemar)** | - | $\chi^2 = 14.82$ | $p = 0.00012$ (Rejeita-se $H_0$) |
| **Poder Estatístico ($1 - \beta$)** | - | $91.4\%$ ($\alpha = 0.05$) | Poder adequado para detectar $\text{OR} \ge 1.8$ |
| **Expected Calibration Error (ECE)** | $0.142$ (Subcalibrado) | $0.048$ (Calibrado) | Redução de $66.2\%$ no desvio de confiança |
| **Brier Score ($BS$)** | $0.185$ | $0.112$ | Menor erro quadrático probabilístico |

---

## 🤖 5. Pipeline RAG Real com Gemini API & Validação de LLM-as-Judge ($\kappa$)

Além do simulador estocástico parametrizado, o repositório implementa um pipeline RAG real sobre um corpus de **Políticas de Logística, SLA de Entrega, Cadeia Fria e Reembolso** (`pipeline/real_rag.py` + `core/llm_client.py`):

- **Integração com Gemini API + Fallback Multi-Modelo**: Suporte nativo à API Gemini (`GEMINI_API_KEY` via `.env`) com *fallback* automático entre modelos, medição de latência ($\text{ms}$), contabilização de tokens e **cache determinístico em disco** (`cache/gemini_cache.json`) para execução em CI/CD com custo zero.
- **Validação de LLM-as-Judge via Kappa de Cohen ($\kappa$)**: A concordância entre o avaliador automático e rótulos humanos de referência é auditada formalmente corrigindo pelo acaso:
  $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
  onde $p_o$ é a proporção de concordância empírica observada e $p_e$ é a concordância esperada sob independência marginal.

---

## ⚠️ 6. Limitações Declaradas do Estudo

1. **Simulação Probabilística + Corpus Curado**: O gerador de benchmark em larga escala ($N=500$) utiliza um simulador probabilístico parametrizado, complementado pelo benchmark real na API Gemini (`pipeline/real_rag.py`) com respostas cacheadas em `cache/gemini_cache.json`.
2. **Independência entre Consultas**: Assume-se independência condicional entre pares de consultas ($q_i \perp q_j$), o que pode não se sustentar em conversas com histórico multi-turn.
3. **Equidade de Subgrupos**: A regra dos 4/5 ($80\%$ rule) auditada detecta disparidades de acurácia, mas não substitui auditorias formais de equidade causal em ambientes de produção.

---

## 🚀 7. Como Reproduzir os Resultados (1-Click)

### Pré-requisitos
* Python 3.10 ou superior

### Execução dos Testes Automatizados
```bash
# 1. Clonar o repositório
git clone https://github.com/Geovana28/rag-stat-eval.git
cd rag-stat-eval

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar a suíte completa de testes com pytest
pytest
```

### Executar a Avaliação Estatística Completa
```bash
python run_eval.py
```
*O script gerará o relatório impresso no terminal e atualizará o arquivo `benchmark_data.json` consumido pelo dashboard web.*

---

## 📜 Citação Científica (BibTeX)

Se este framework for utilizado em pesquisas acadêmicas ou auditorias técnicas, utilize a citação abaixo (disponível também em `CITATION.cff`):

```bibtex
@software{silva_rag_stat_eval_2026,
  author = {Silva, Geovana},
  title = {RAG-Stat-Eval: Rigorous Statistical Validation and Robustness Auditing Suite for Retrieval-Augmented Generation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Geovana28/rag-stat-eval}
}
```
