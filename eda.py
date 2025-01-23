import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unicodedata import numeric

ds = pd.read_csv('teste.csv')

#visualizar as primeiras linhas
print(ds.head())

#visualizar dimensões do arquivo.
print(ds.shape)

#info gerais sobre o dataset
print(ds.info())

#estatisticas descritivas (media, mediana, desvio padrão, valores min e máx.)
print(ds.describe())

#substituindo os valores nulos (dados missing) por 0
ds['nome'] = ds['nome'].fillna(0)
ds['host_name'] = ds['host_name'].fillna(0)
ds['ultima_review'] = ds['ultima_review'].fillna(0)
ds['reviews_por_mes'] = ds['reviews_por_mes'].fillna(0)
print(ds.isnull().sum())



#2 a)
# média de preço por bairro
ds = ds[ds['price'] > 0] # remove preços 0 ou invalido
ds = ds[ds['price'] < ds['price'].quantile(0.99)] #tirando outliers extremos

preco_bairro = ds.groupby('bairro_group')['price'].mean().sort_values(ascending=False)

melhor_bairro = preco_bairro.head(10)
print("Melhores bairros para compra com maior preço médio:", melhor_bairro)
plt.figure(figsize=(8, 5))
preco_bairro.head(10).plot(kind='bar', color='lightcoral')
plt.title('Média de preço por bairro:')
plt.xlabel('Bairro')
plt.ylabel('Preço médio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show



#2 b)
correlacao = ds[['price', 'disponibilidade_365', 'reviews_por_mes']].corr()

#scatter plot para explorar a relação entre disponibilidade e preço (mapa de calor)
sns.lmplot(x='disponibilidade_365', y='price', data=ds, aspect=2, height=4, scatter_kws={'color': 'skyblue', 's':20}, line_kws={'color': 'orange'})
plt.title('Disponibilidade x Preço')
plt.xlabel('Disponibilidade (dias)')
plt.ylabel('Preço')
plt.show()


#scatter plot para explorar a relação entre reviews por mês e preço (mapa de calor)
sns.lmplot(x='reviews_por_mes', y='price', data=ds, aspect=2, height=4, scatter_kws={'color':'skyblue', 's':20}, line_kws={'color':'orange'})
plt.title('Reviews por Mês x Preço')
plt.xlabel('Reviews por Mês')
plt.ylabel('Preço')
plt.show()



#2 c)
from sklearn.feature_extraction.text import CountVectorizer


bairros_top = preco_bairro.head(10).index
bairros_luxo = ds[ds['bairro_group'].isin(bairros_top)]
bairros_luxo['nome'] = bairros_luxo ['nome'].astype(str)

# Criando o vetor de contagem
vector = CountVectorizer(stop_words='english', max_features=10)

# Transformando os nomes dos bairros em uma matriz esparsa
X = vector.fit_transform(bairros_luxo['nome'])

# Convertendo a matriz esparsa para um array denso e criando um DataFrame para somar as frequências
palavras_freq = pd.DataFrame(X.toarray(), columns=vector.get_feature_names_out()).sum().sort_values(ascending=False)

# Exibindo as palavras mais frequentes
print("Para locais de alto valor, as palavras mais frequentes foram:")
print(palavras_freq)


#grafico de barras: Palavras x Lugares
plt.figure(figsize=(8, 6))
palavras_freq.head(10).plot(kind='bar', color='gray')
plt.title("Frequencia de palavras por lugares")
plt.xlabel("Palavra")
plt.ylabel("Frequencia")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()