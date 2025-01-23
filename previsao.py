import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor 


ds = pd.read_csv('teste.csv')


# O random Forest foi escolhido por sua capacidade de capturar relações não lineares nos dados e por ser menos sensível a overfitting
# O Gradient Boosting é um pouco mais preciso, porem exige mais tempo para ajustes.


#Transformar varaiveis categoricas em numerocas(one hot encoding)
X= pd.get_dummies(ds[['bairro_group', 'reviews_por_mes', 'disponibilidade_365']])
y= ds['price']

#dividindo dados (treino e teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#modelo de regressão (Randoom Forest)
model = RandomForestRegressor()
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print('MSE: {mse}')
print('RMSE: {rmse}')



# --------------------------------------------- RANDOM FOREST ---------------------------------------------------

#tratando valores nulos
ds['nome'] = ds['nome'].fillna('Unknowm')
ds['host_name'] = ds['host_name'].fillna('Unkown')
ds['ultima_review'] = ds['ultima_review'].fillna('No Review')
ds['reviews_por_mes'] = ds['reviews_por_mes'].fillna('0')


#separando features (X) e o alvo (Y)
X = ds.drop(columns=['price'])
y = ds['price']

#dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #dividir dados entre treino e teste

# transformador para viariáveis categóricas e numéricas
    #padronizar variaveis numericas com StandardScaler
        
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['latitude', 'longitude', 'minimo_noites', 'numero_de_reviews','reviews_por_mes', 'calculado_host_listings_count', 'disponibilidade_365']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['bairro_group', 'room_type'])
        ])

#criando pipeline do modelo
pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor), 
            ('model', GradientBoostingRegressor(random_state=42))
])


#trteinando o modelo
pipeline.fit(X_train, y_train)

# previsões no teste
y_pred = pipeline.predict(X_test)

# Avaliando o modelo
#avaliação para medir a performace do modelo criado
rmse = mean_squared_error(y_test, y_pred, squared=False) 
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")

# Salvando o modelo Random Forest
joblib.dump(model, 'random_forest_model.pkl')
print("Modelo Random Forest salvo como 'random_forest_model.pkl'")



# ---------------------------------------------------- GRADIENT BOOSTING -----------------------------------------------------------------------
#tratando valores nulos
ds['nome'] = ds['nome'].fillna('Unknowm')
ds['host_name'] = ds['host_name'].fillna('Unkown')
ds['ultima_review'] = ds['ultima_review'].fillna('No Review')
ds['reviews_por_mes'] = ds['reviews_por_mes'].fillna('0')

## utilizamos o modelo treinado para prever o preço do apartamento com características fornecidas
apartamento = {
 'id': 2595,
 'nome': 'Skylit Midtown Castle',
 'host_id': 2845,
 'host_name': 'Jennifer',
 'bairro_group': 'Manhattan',
 'bairro': 'Midtown',
 'latitude': 40.75362,
 'longitude': -73.98377,
 'room_type': 'Entire home/apt',
 'minimo_noites': 1,
 'numero_de_reviews': 45,
 'ultima_review': '2019-05-21',
 'reviews_por_mes': 0.38,
 'calculado_host_listings_count': 2,
 'disponibilidade_365': 355
 }


#convertando em dataframe
apartamento_ds = pd.DataFrame([apartamento])

#veririca se as colunas estão presentes, caso não, assume o valor 0 para colunas ausentes
for col in X_train.columns:
    if col not in apartamento_ds:
        apartamento_ds[col] = 0


#ordenando colunas para coincidir com o pipe line
apartamento_ds = apartamento_ds[X_train.columns]
    
#prevendo preço do ap
previsao_preco = pipeline.predict(apartamento_ds)
print(f"Minha sugestão de preço é de: ${previsao_preco[0]: .2f}")

# Salvando o pipeline do Gradient Boosting
joblib.dump(pipeline, 'gradient_boosting_pipeline.pkl')
print("Pipeline Gradient Boosting salvo como 'gradient_boosting_pipeline.pkl'")