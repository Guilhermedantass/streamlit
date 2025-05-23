import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind
import kagglehub
import numpy as np
import scipy.stats as stats

# Configuração da página
st.set_page_config(page_title="Dashboard Manutenção Preditiva", layout="wide")
st.title("Dashboard de Manutenção Preditiva")

# Carregamento dos Dados
try:
    path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
    file_path = os.path.join(path, "predictive_maintenance.csv")
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Erro ao carregar o dataset: {e}")
    st.stop()

# Padronização dos nomes das colunas
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

# Pré-processamento
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Sidebar - Filtros
st.sidebar.header('Filtros')
produto = st.sidebar.multiselect(
    'Selecione o Produto', 
    options=df['type'].unique(), 
    default=df['type'].unique()
)
df_filt = df[df['type'].isin(produto)]


st.subheader("Distribuição dos Tipos de Falha")

falhas = df[df['failure_type'] != 'No Failure']
contagem = falhas['failure_type'].value_counts().reset_index()
contagem.columns = ['failure_type', 'qtd']

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=contagem, x='failure_type', y='qtd', palette='Set3', ax=ax)

ax.set_title("Quantidade de Cada Tipo de Falha")
ax.set_xlabel("Tipo de Falha")
ax.set_ylabel("Quantidade")

# 🔧 Rotaciona os textos do eixo X
plt.xticks(rotation=30, ha='right')

plt.tight_layout()  # 📐 Ajusta o layout para evitar sobreposição
st.pyplot(fig)

st.markdown("---")

# Proporção de falhas por tipo de produto
st.subheader('Proporção de Falhas por Tipo de Produto')

total_por_produto = df_filt['type'].value_counts().rename('total_produto')
falhas = df_filt[df_filt['failure_type'] != 'No Failure']
contagem_falhas = falhas.groupby(['type', 'failure_type']).size().rename('qtd_falhas').reset_index()
contagem_falhas = contagem_falhas.merge(total_por_produto, left_on='type', right_index=True)
contagem_falhas['proporcao'] = contagem_falhas['qtd_falhas'] / contagem_falhas['total_produto']

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=contagem_falhas, x='type', y='proporcao', hue='failure_type', ax=ax, palette='Set2')
ax.set_title('Proporção de Falhas por Produto')
ax.set_ylabel('Proporção (%)')
ax.set_xlabel('Tipo de Produto')
ax.set_ylim(0, contagem_falhas['proporcao'].max() * 1.1)
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("---")

# Relação entre variáveis e tipos de erro
st.subheader('Relação entre Variáveis e Tipos de Falha')

vars_interesse = ['torque_[nm]', 'air_temperature_[k]', 'process_temperature_[k]', 'tool_wear_[min]']
tipo_falha = 'failure_type'

# Bloco 1: torque e air_temperature
fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5))
for i, ax in enumerate(axs1):
    var = vars_interesse[i]
    sns.stripplot(x=tipo_falha, y=var, data=df_filt, jitter=True, ax=ax, palette="Set1")
    ax.set_title(f'Dispersão entre {var} e Tipo de Falha')
    ax.set_xlabel('Tipo de Falha')
    ax.set_ylabel(var)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig1)

st.markdown("")

# Bloco 2: process_temperature e tool_wear
fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))
for i, ax in enumerate(axs2):
    var = vars_interesse[i + 2]
    sns.stripplot(x=tipo_falha, y=var, data=df_filt, jitter=True, ax=ax, palette="Set1")
    ax.set_title(f'Dispersão entre {var} e Tipo de Falha')
    ax.set_xlabel('Tipo de Falha')
    ax.set_ylabel(var)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# Tempo médio até a falha (tool wear)
st.subheader("Tempo Médio Até a Falha (Tool Wear [min])")
falhas_ativas = df_filt[df_filt[tipo_falha] != 'No Failure']

if not falhas_ativas.empty:
    tempo_medio = falhas_ativas['tool_wear_[min]'].mean()
    st.write(f"**Tempo médio até uma falha:** {tempo_medio:.2f} minutos")
else:
    st.write("Nenhuma falha encontrada para o filtro selecionado.")

st.markdown("---")

# ==== Análises adicionais ====
st.subheader("Tempo Médio Até a Falha por Produto")

falhas_tool = df[df['failure_type'] != 'No Failure']
tempo_medio_por_tipo = falhas_tool.groupby('type')['tool_wear_[min]'].mean().reset_index()

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=tempo_medio_por_tipo, x='type', y='tool_wear_[min]', palette="Blues", ax=ax)

ax.set_title("Tempo Médio de Tool Wear Até a Falha por Produto")
ax.set_xlabel("Produto")
ax.set_ylabel("Tempo Médio (min)")
plt.tight_layout()

st.pyplot(fig)