#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Premios-Goya" data-toc-modified-id="Premios-Goya-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Premios Goya</a></span><ul class="toc-item"><li><span><a href="#Datos-Premios-Goya" data-toc-modified-id="Datos-Premios-Goya-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Datos Premios Goya</a></span></li><li><span><a href="#Datos-Premios-Feroz" data-toc-modified-id="Datos-Premios-Feroz-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Datos Premios Feroz</a></span></li><li><span><a href="#Datos-Premios-Forqué" data-toc-modified-id="Datos-Premios-Forqué-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Datos Premios Forqué</a></span></li><li><span><a href="#Datos-Premios-CEC" data-toc-modified-id="Datos-Premios-CEC-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Datos Premios CEC</a></span></li><li><span><a href="#Analytical-Base-Table" data-toc-modified-id="Analytical-Base-Table-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Analytical Base Table</a></span></li><li><span><a href="#Variables-dummy.-Porcentajes-por-año-de-variables-numéricas" data-toc-modified-id="Variables-dummy.-Porcentajes-por-año-de-variables-numéricas-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Variables dummy. Porcentajes por año de variables numéricas</a></span></li><li><span><a href="#Modelo-'2000'-o-'Todos-los-años'.-Búsqueda-de-parámetros" data-toc-modified-id="Modelo-'2000'-o-'Todos-los-años'.-Búsqueda-de-parámetros-1.7"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Modelo '2000' o 'Todos los años'. Búsqueda de parámetros</a></span><ul class="toc-item"><li><span><a href="#Ajustar:" data-toc-modified-id="Ajustar:-1.7.1"><span class="toc-item-num">1.7.1&nbsp;&nbsp;</span>Ajustar:</a></span></li></ul></li><li><span><a href="#Se-ajusta-el-modelo-con-train-+-test" data-toc-modified-id="Se-ajusta-el-modelo-con-train-+-test-1.8"><span class="toc-item-num">1.8&nbsp;&nbsp;</span>Se ajusta el modelo con train + test</a></span><ul class="toc-item"><li><span><a href="#Ajustar:" data-toc-modified-id="Ajustar:-1.8.1"><span class="toc-item-num">1.8.1&nbsp;&nbsp;</span>Ajustar:</a></span></li></ul></li><li><span><a href="#Predicción-2023" data-toc-modified-id="Predicción-2023-1.9"><span class="toc-item-num">1.9&nbsp;&nbsp;</span>Predicción 2023</a></span></li></ul></li></ul></div>

# # Premios Goya

# ## Datos Premios Goya

# In[ ]:


def manejar_cookies():
    try:
        cookies = driver.find_element(By.CLASS_NAME, "css-v43ltw")
        cookies.click()
        return
    except:
        print("Botón de cookies no encontrado")
        return


# In[ ]:


################
# Premios Goya #
################

from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re


def datos1():
    """
    Primer bloque de datos:
    --> Título de la película.
    --> Nº de nominaciones.
    --> Ganador (1/0).
    """
    # Listas para títulos y nominaciones recibidas
    tp, nomi = [], []
    
    # Título de la película ganadora del Goya
    tp.append(html1.find(id='goya_best_picture').parent.select('span.movie-title-link')[0].text.strip())

    # Título de las películas nominadas
    for n in html1.find(id='goya_best_picture').parent.select('span.aw-mc.nominee-title'):
        tp.append(n.a.text.strip())

    # Nominaciones de todas las películas finalistas incluida la ganadora
    for m in html1.find(id='goya_best_picture').parent.select('div.nom-wins'):
        if m.b == None:
            nomi.append('0')
            continue
        nomi.append(m.b.text)

    # Lista con ganador (1) o no (0)
    gdor = [1] + [0] * (len(tp) - 1)
    
    return tp, nomi, gdor


def datos2():
    """
    Segundo bloque de datos:
    --> Puntuación media.
    --> Nº de votos recibidos.
    --> Nº de críticas recibidas.
    --> Coproducción (1/0).
    --> Género (lista de géneros principales).
    Cada lista contiene datos homogéneos.
    """
    puntuacion = driver.find_element_by_css_selector('#movie-rat-avg').text.replace(",", ".")
    votos = driver.find_element_by_css_selector('#movie-count-rat > span').text.replace(".", "")
    criticas = driver.find_element_by_css_selector('#movie-reviews-box').text.split()[0]
    coproduccion = [1 if "Coproducción" in i.text else 0 for i in html2.select('dd div.credits span.nb span')]
    if 1 in coproduccion:
        coproduccion = 1
    else:
        coproduccion = 0
    genero = [i.text for i in html2.select('dd span[itemprop="genre"]')]  # Lista de géneros principales.
    return [puntuacion, votos, criticas, coproduccion, genero]


# Listas iniciales
ttl, nmn, gnd, yr = [], [], [], []
dts2 = []

# Abrir Chrome y manejar cookies
driver = webdriver.Chrome()
driver.get("https://www.filmaffinity.com/es/award_data.php?award_id=goya")

# Cargar páginas años
for ed in range(1987, 2023):
    driver.get("https://www.filmaffinity.com/es/awards.php?award_id=goya&year=" + str(ed))

    # Bloque 1 de datos
    html1 = BeautifulSoup(driver.page_source, 'html.parser')
    titulo, nominaciones, ganador = datos1()
    year = [ed] * len(titulo)
    
    # Unión de listas del bloque 1
    ttl.extend(titulo)
    nmn.extend(nominaciones)
    gnd.extend(ganador)
    yr.extend(year)

    # Bloque 2 de datos
    for ti in titulo:
        link = html1.find(id='goya_best_picture').parent.find(
            href=True, string=re.compile(ti.split("(")[0].strip())).get('href')
        driver.get("https://www.filmaffinity.com/es/film" + link[link.rfind("=")+1:] + ".html")
        html2 = BeautifulSoup(driver.page_source, 'html.parser')
        dts2.append(datos2())

# Unir datos
d = {'ganador': gnd, 'titulo': ttl, 'year': yr, 'nominaciones': nmn}
df1 = pd.DataFrame(d)
df2 = pd.DataFrame(dts2, columns=['puntuacion', 'votos', 'criticas', 'coproduccion', 'genero'])
df = df1.join(df2)

# Cerrar driver
driver.close()

# Guardar tabla de datos
df.to_csv('C:/Users/danie/Desktop/PYTHON/Goyas/Goyas_2023/df_goyas_data.csv', index=False)
df


# ## Datos Premios Feroz

# In[ ]:


#################
# Premios Feroz #
#################

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def datos_feroz():
    """
    Bloque de datos:
    --> Título de la película drama y comedia.
    --> Ganador drama y comedia (1/0).
    """
    # Listas para títulos y ganador/nominado de drama y comedia
    tp_drama, tp_comedia = [], []
    
    # Búsqueda de ganadores y nominados por categoría drama
    tp_drama.append(html1.find(id='feroz_mejor_pelicula_drama').parent.select('span.movie-title-link')[0].text.strip())
    for n in html1.find(id='feroz_mejor_pelicula_drama').parent.select('span.aw-mc.nominee-title'):
        tp_drama.append(n.a.text.strip())
    gdor_drama = ["ganador"] + ["nominado"] * (len(tp_drama) - 1)
    
    # Búsqueda de ganadores y nominados por categoría comedia
    tp_comedia.append(html1.find(id='feroz_mejor_comedia').parent.select('span.movie-title-link')[0].text.strip())
    for n in html1.find(id='feroz_mejor_comedia').parent.select('span.aw-mc.nominee-title'):
        tp_comedia.append(n.a.text.strip())
    gdor_comedia = ["ganador"] + ["nominado"] * (len(tp_comedia) - 1)
    
    return tp_drama, gdor_drama, tp_comedia, gdor_comedia


# Listas iniciales
drama_feroz, comedia_feroz, gs_drama, gs_comedia = [], [], [], []

# Cargar páginas años
driver = webdriver.Chrome()
for ed in range(2014, 2023):
    driver.get("https://www.filmaffinity.com/es/awards.php?award_id=feroz&year=" + str(ed))

    # Datos Premios Feroz
    html1 = BeautifulSoup(driver.page_source, 'html.parser')
    dr_feroz, g_dr_feroz, co_feroz, g_co_feroz = datos_feroz()

    # Unión de listas Premios Feroz
    drama_feroz.extend(dr_feroz)
    comedia_feroz.extend(co_feroz)
    gs_drama.extend(g_dr_feroz)
    gs_comedia.extend(g_co_feroz)

# Cerrar driver
driver.close()

# Coordinación con Premios Goya
df = pd.read_csv('df_goyas_data.csv')
dra, com = [], []
fedrlow = [t.lower() for t in drama_feroz]
fecolow = [t.lower() for t in comedia_feroz]
for i in df.titulo.tolist():
    if i.lower() in fedrlow:
        dra.extend([gs_drama[drama_feroz.index(i)]])
    else:
        dra.extend(["no_clasificado"])
    if i.lower() in fecolow:
        com.extend([gs_comedia[comedia_feroz.index(i)]])
    else:
        com.extend(["no_clasificado"])

# Unión Premios Feroz a Premios Goya
df['feroz_drama'] = dra
df['feroz_comedia'] = com

# Guardar tabla de datos Goya + Feroz
df.to_csv('df_goyas_feroz_data.csv', index=False)
df


# ## Datos Premios Forqué

# In[ ]:


##################
# Premios Forqué #
##################

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def datos_forque():
    """
    Bloque de datos:
    --> Título de la película.
    --> Ganador (1/0).
    """
    # Lista títulos películas premios Forque
    tp = []
    
    # Ganador
    tp.append(html1.find(id='forque_best_picture').parent.select('span.movie-title-link')[0].text.strip())
    
    # Nominadas
    for n in html1.find(id='forque_best_picture').parent.select('span.aw-mc.nominee-title'):
        tp.append(n.a.text.strip())

    # Lista ganadores y nominados
    gdor = ["ganador"] + ["nominado"] * (len(tp) - 1)
    
    return tp, gdor


# Listas iniciales
ti_forque, ga_forque = [], []

# Cargar páginas años
# Ajuste: añadido 2023 como "https://www.filmaffinity.com/es/award-edition.php?edition-id=forque_2022_dic"
driver = webdriver.Chrome()
for ed in range(1996, 2023):
    if ed != 2023:
        driver.get("https://www.filmaffinity.com/es/awards.php?award_id=forque&year=" + str(ed))
    else:
        driver.get("https://www.filmaffinity.com/es/award-edition.php?edition-id=forque_2022_dic")
        
    # Datos Premios Feroz
    html1 = BeautifulSoup(driver.page_source, 'html.parser')
    tit_fq, gan_fq = datos_forque()
    
    # Unión de listas Premios Feroz
    ti_forque.extend(tit_fq)
    ga_forque.extend(gan_fq)
    
# Cerrar driver
driver.close()

# Coordinación con Premios Goya
df = pd.read_csv('df_goyas_feroz_data.csv')
fq = []
tfq = [t.lower() for t in ti_forque]
for i in df.titulo.tolist():
    if i.lower() in tfq:
        fq.extend([ga_forque[ti_forque.index(i)]])
    else:
        fq.extend(["no_clasificado"])

# Unión Premios Feroz a Premios Goya
df['forque'] = fq

# Guardar tabla de datos Goya + Feroz
df.to_csv('df_goyas_feroz_forque_data.csv', index=False)
df


# ## Datos Premios CEC

# In[ ]:


###############
# Premios CEC #
###############

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd


def gan_cec():
    """
    Bloque de datos:
    --> Título de la película ganadora.
    """
    # Ganadores
    tp = html1.find('a', string=re.compile(str(ed))).parent.parent.parent.parent.find_all('span')[3].text
    return tp


def nom_cec():
    """
    Bloque de datos:
    --> Título de las películas nominadas.
    """
    # Lista para nominadas
    nomi = []
    
    # Búsqueda nominadas en tabla, fila, columna y tag 'i'
    tabla = html2.find_all('table', class_='wikitable')[0]
    fila = tabla.find_all('tr')[1]
    for col in fila.find_all('td')[3:]:
        for p in col.find_all('i'):
            nomi.append(p.text)
    return nomi


# Lista ganadores y nominados
ti_cec, nomi_cec = [], []

# Títulos con nombre incompleto en web CEC
ti_incompleto = {'NADIE HABLARÁ DE NOSOTRAS...': 'nadie hablará de nosotras cuando hayamos muerto', 
                     'VIVIR ES FÁCIL CON LOS OJOS...\xa0': 'vivir es fácil con los ojos cerrados'}

# Recopilación ganadores
driver = webdriver.Chrome()
driver.get("https://archivocine.com/index.php/premios/circulo-de-escritores-cinematograficos-cec")
html1 = BeautifulSoup(driver.page_source, 'html.parser')

# Ajuste de años no celebrados
todos = set(range(1945, 2022))
no_celebrados = {1972, 1980, 1985, 1986, 1987, 1988, 1989}
edi_celebradas = list(todos.difference(no_celebrados))

for ed in edi_celebradas:
    ti_cec.append(gan_cec())
#g_cec = ["ganador"] * len(ti_cec)

# Títulos incompletos sustituidos
for i in ti_incompleto:
    ti_cec[ti_cec.index(i)] = ti_incompleto[i]

# Recopilación finalistas (ediciones válidas: 2003-2012, 2017-2019)
edi_validas = list(range(2003, 2013)) + list(range(2017, 2020))
for ed in edi_validas:
    driver.get("https://es.wikipedia.org/wiki/Anexo:Medallas_del_CEC_de_" + str(ed))
    html2 = BeautifulSoup(driver.page_source, 'html.parser')
    nomi_cec.extend(nom_cec())
#n_cec = ["nominado"] * len(nomi_cec)

# Cerrar driver
driver.close()

# Se añaden manualmente las ediciones 2013, 2014, 2015, 2016, 2020, 2021
nomi_manual = ["15 años y un día", "Caníbal", "Stockholm",
               "El niño", "Relatos salvajes", "Loreak (Flores)",
               "Techo y comida", "La novia", "El desconocido", "Un día pefecto",
               "El hombre de las mil caras", "Julieta", "El olivo",
               "Las niñas", "Adú", "Uno para todos",
               "El buen patrón", "Maixabel", "Libertad", "Las leyes de la frontera"]
nomi_cec.extend(nomi_manual)
#n_cec = ["nominado"] * len(nomi_cec)



# Coordinación con Premios Goya
df = pd.read_csv('df_goyas_feroz_forque_data.csv')
cec = []
tcw = [t.lower() for t in ti_cec]
tcn = [n.lower() for n in nomi_cec]
for i in df.titulo.tolist():
    if i.lower() in tcn:  # Nominados. Todo en minúsculas.
        cec.extend(["nominado"])
    elif i.lower() in tcw:  # Ganadores. Todo en minúsculas.
        cec.extend(["ganador"])
    else:
        cec.extend(["no_clasificado"])

# Unión Premios CEC a Premios Goya
df['cec'] = cec

# Guardar tabla de datos Goya + CEC
df.to_csv('df_goyas_feroz_forque_cec_data.csv', index=False)
df


# ## Analytical Base Table

# In[ ]:


import pandas as pd
import pickle


df = pd.read_csv('df_goyas_feroz_forque_cec_data.csv')

# Columna genero de string a lista
df.genero = df.genero.apply(lambda x: [r.strip("[").strip("]").strip(" ").strip("'") for r in x.split(",")])

# Lista de géneros posibles
y = []
[y.extend(i) for i in df.genero]
lgp = list(set(y))

# Dummy géneros
dummy_generos = pd.DataFrame(columns=lgp, index=df.index)
for n, i in zip(df.genero, df.index):
    for j in n:
        if j in lgp:
            dummy_generos.at[i, j] = 1
dummy_generos.fillna(0, inplace=True)

# Dummy Premios Feroz
dummy_fe_dr = pd.get_dummies(df.feroz_drama, prefix='feroz_drama')
dummy_fe_co = pd.get_dummies(df.feroz_comedia, prefix='feroz_comedia')

# Dummy Premios Forqué
dummy_forque = pd.get_dummies(df.forque, prefix='forque')

# Dummy Premios CEC
dummy_cec = pd.get_dummies(df.cec, prefix='cec')

# Unión df y df_dummy
df = pd.concat([df, dummy_fe_dr, dummy_fe_co, dummy_forque, dummy_cec, dummy_generos], axis=1, sort=False)

# Parche manual: una película en los Goyas al menos tiene una nominación
df.at[df[df.titulo == '27 horas'].index[0], 'nominaciones'] = 1

# Guardar ABT de Goyas
df.to_csv('ABT_Goyas.csv', index=False)

# Guardar lista de géneros
with open("lista_generos.txt", "wb") as f:
    pickle.dump(lgp, f)
df


# ## Variables dummy. Porcentajes por año de variables numéricas

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


df = pd.read_csv('ABT_Goyas.csv')
with open("lista_generos.txt", "rb") as f:
    lgp = pickle.load(f)

# Películas por género
for feature in lgp:
    print(feature, ":", df[feature].sum())
    
# Eliminación columnas usadas para generar dummies
df.drop(columns=['genero', 'feroz_drama', 'feroz_comedia', 'forque', 'cec'], inplace=True)

# Fusión de Premios Feroz drama y comedia (pocos datos en comedia)
#df['feroz_ganador'] = df.feroz_drama_ganador + df.feroz_comedia_ganador
#df['feroz_nominado'] = df.feroz_drama_nominado + df.feroz_comedia_nominado

# Se eliminan columnas no usadas
df.drop(columns=[#'feroz_drama_ganador', 
                 'feroz_drama_no_clasificado', 
                 #'feroz_drama_nominado', 
                 'feroz_comedia_no_clasificado', 
                 #'feroz_comedia_nominado', 
                 'forque_no_clasificado', 
                 'cec_no_clasificado'], inplace=True)

# Muchos géneros (13). Hacemos 4 grupos:
# Grupo más frecuente: Drama
df['G_drama'] = df.Drama
# Grupo de frecuencia media-alta: Comedia, Thriller, Romance e Intriga.
df['G_med_alto'] = np.where((df.Comedia == 1) 
                               | (df.Thriller == 1) 
                               | (df.Romance == 1) 
                               | (df.Intriga == 1), 1, 0)
# Grupo de frecuencia media-baja: Fantástico, Terror, Acción y Aventuras.
df['G_med_bajo'] = np.where((df['Fantástico'] == 1) 
                               | (df.Terror == 1) 
                               | (df['Acción'] == 1) 
                               | (df.Aventuras == 1), 1, 0)
# Grupo de frecuencia baja: Western, Ciencia ficción, Musical y Cine negro.
df['G_bajo'] = np.where((df.Western == 1) 
                               | (df['Ciencia ficción'] == 1) 
                               | (df.Musical == 1) 
                               | (df['Cine negro'] == 1), 1, 0)

# Eliminación de géneros ya agrupados
df.drop(columns=lgp, inplace=True)

# Control de Nan en premios sin ediciones (parche). Referencia inicial Goyas: 1987
num_tit = df.groupby('year').count()
n_feroz = num_tit[num_tit.index > 2013].sum()['titulo']
n_forque = num_tit[num_tit.index > 1995].sum()['titulo']
n_cec = num_tit[num_tit.index > 1986].sum()['titulo']
feroz_nan =  [0] * (len(df) - n_feroz) + [1] * n_feroz  # 0 = nan, 1 = sí hay dato.
forque_nan = [0] * (len(df) - n_forque) + [1] * n_forque
cec_nan = [0] * (len(df) - n_cec) + [1] * n_cec
df['feroz_nan'] = feroz_nan
df['forque_nan'] = forque_nan
df['cec_nan'] = cec_nan

# Porcentaje anual de variables cuantitativas
t_anual = df.groupby('year')[['nominaciones', 'puntuacion', 'votos', 'criticas']].sum()  # Totales.
df.set_index('year', inplace=True)
ptj = df[['nominaciones', 'puntuacion', 'votos', 'criticas']] / t_anual  # Porcentaje.
df[['nominaciones', 'puntuacion', 'votos', 'criticas']] = ptj
df.reset_index(inplace=True)

# Eliminamos columnas que no vamos a usar como variables independientes
df.drop(columns=['year', 'titulo'], inplace=True)

# Matriz final: 145 registros, 21 variables X. "ganador" variable Y.
# Guardar matriz de datos de Premios Goyas
df.to_csv('Datos_Goyas.csv', index=False)
df


# ## Modelo '2000' o 'Todos los años'. Búsqueda de parámetros

# ### Ajustar:
# - Año de inicio: desde 2000 ó desde el principio

# In[ ]:


import pickle
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Cargar tabla final de variables
df = pd.read_csv('Datos_Goyas.csv')

# Descomentar para datos SOLO desde el año 2000 incluido (44:)
#df = df[44:]

# Separar variable target y variables predictoras
y = df.ganador
X = df.drop('ganador', axis=1)

# Dividir X e y en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=df.ganador)

# Comprobar train y test
print("Se comprueba train y test:")
print(len(X_train), len(X_test), len(y_train), len(y_test))
print(y_train.mean(), y_test.mean())

# Estandarización
scaler_X = StandardScaler().fit(X_train)
X_train_std = scaler_X.transform(X_train)
X_test_std = scaler_X.transform(X_test)

# Algoritmo
clf_gb = GradientBoostingClassifier(random_state=123)

# Hiperparámetros
hyperparameters = {'n_estimators': [100, 200, 500, 1000, 1500, 2000, 3000], 
                   'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2], 
                   'max_depth': [1, 3, 5, 7, 9]}

# Modelo
modelo = GridSearchCV(clf_gb, hyperparameters, n_jobs=-1, cv=10)

# Ajustar modelo
modelo.fit(X_train_std, y_train)
print("\nBest Train Score:", modelo.best_score_)

# Matriz de confusión. Predicción de categorías (y_test)
pred = modelo.predict(X_test_std)
print("MC sobre test:", "\n", confusion_matrix(pred, y_test))

# Curva ROC. Predicción de probabilidades (y_test)
pred = modelo.predict_proba(X_test_std)
pred = pred[:, 1]  # Se toma sólo la clase positiva
fpr, tpr, threshols = roc_curve(y_test, pred)

# Dibujar la curva ROC
# Valores gráfico
fig = plt.figure(figsize=(8, 8))
plt.title('Receiver Operating Characteristic (ROC)')
# Plot ROC curve
plt.plot(fpr, tpr, label='gradient boosting')
plt.legend(loc='lower right')
# Diagonal 45 degree line
plt.plot([0, 1], [0, 1], 'k--')
# Axes limits and labels
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.show()

# AUROC
print("Área bajo la curva ROC", auc(fpr, tpr))

# Parámetros del modelo
print("\nParámetros del modelo ajustado:", modelo.best_estimator_, "\n", modelo.best_estimator_.get_params())

# Importancia de las variables
print("\nImportancia sobre 1:\n")
for nom, val in zip(X, modelo.best_estimator_.feature_importances_):
    print(nom, "\t", round(val, 3))

# Guardar el modelo
with open('modelo_goyas_gb.pkl', 'wb') as f:
    pickle.dump(modelo.best_estimator_, f)


# ## Se ajusta el modelo con train + test

# ### Ajustar:
# - Variables a cargar: usecols
# 
# - Parámetros del mejor modelo: mod_final_goyas
# 
# - Nombre de importancia de características: nom_var

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Datos: todos o solo features relevantes (Importancia >= 0.01)
#df = pd.read_csv('Datos_Goyas.csv')
df = pd.read_csv('Datos_Goyas.csv', 
                 usecols=['ganador',
                          'nominaciones',
                          'puntuacion',
                          'votos',
                          'criticas',
                          'feroz_comedia_nominado',
                          'forque_ganador',
                          'forque_nominado',
                          'cec_ganador',
                          'G_med_bajo',
                          'forque_nan'
                        ])

# Separar variable target y variables predictoras
y = df.ganador
X = df.drop('ganador', axis=1)

# Estandarización de todo X
scaler_X = StandardScaler().fit(X)
X_scaler = scaler_X.transform(X)

# Algoritmo con parámetros del mejor modelo (cambiar cuando se reajusta el modelo)
mod_final_goyas = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.001, max_depth=3, random_state=123)

# Ajuste del modelo sobre todos los datos
mod_final_goyas.fit(X_scaler, y)

# Guardar el modelo sobre todas las películas
with open('modelo_goyas_gb_todas.pkl', 'wb') as f:
    pickle.dump(mod_final_goyas, f)

# Diagramas de barras con importancia de características: todas o solo relevantes
# nom_var = ['Nº Nominaciones', 'Puntuación media', 'Nº Votos','Nº Críticas',
#            'Coproducción', 'Ganador Premios Feroz Drama', 'Nominado Premios Feroz Drama',
#            'Ganador Premios Feroz Comedia', 'Nominado Premios Feroz Comedia',
#            'Ganador Premios Forqué', 'Nominado Premios Forqué',
#            'Ganador Premios CEC', 'Nominado Premios CEC',
#            'Género Drama', 'Género de frecuencia media-alta',
#            'Género de frecuencia media-baja', 'Género de frecuencia baja',
#            'Feroz nan', 'Forqué nan', 'CEC nan']
nom_var = ['Nº Nominaciones',
           'Puntuación media',
           'Nº Votos',
           'Nº Críticas',
           'Nom. P. Feroz Comed.',
           'Ganador P. Forqué',
           'Nominado P. Forqué',
           'Ganador P. CEC',
           'Género frec. med.-baja',
           'Forqué nan'
           ]

# Data frame con valores mayores que 0, y ordenado
features = {'Variable': nom_var, 'Porcentaje': 100 * mod_final_goyas.feature_importances_}
a = pd.DataFrame(features)
b = a[a.Porcentaje > 0].sort_values(by=['Porcentaje'])
b.index = list(range(len(b)))  # Renombrando índice tras ordenar por valores.

# Gráfico features
fig_features = plt.figure(figsize=(16, 9))
colores = ['aqua', 'yellow', 'red', 'green', 'silver', 'olive', 'navy', 'fuchsia', 'maroon']
plt.barh(b.Variable, b.Porcentaje, align='center', alpha=0.6, color=colores)
# Eje x. Título
plt.xlabel('Porcentaje')
#plt.title('RELEVANCIA DE VARIABLES (desde 2000)')
plt.title('RELEVANCIA DE VARIABLES (1987-2022)')  # Para el años 2023 se usan todos los años.
# Etiquetas de datos
for i, fila in b.iterrows():
    plt.text(x=fila.Porcentaje - 1.2, y=i, s="{0:.1%}".format(fila.Porcentaje/100), size=16, color='black')
plt.tick_params(axis='y', labelrotation=30)
plt.show()
# Salvar gráfico
fig_features.savefig('features importances.jpg')

# Matriz de confusión. Predicción de categorías (y_test)
pred = mod_final_goyas.predict(X_scaler)
print("Matriz Confusión sobre total:", "\n", confusion_matrix(pred, y))

# Curva ROC. Predicción de probabilidades (y_test)
pred = mod_final_goyas.predict_proba(X_scaler)
pred = pred[:, 1]  # Se toma sólo la clase positiva
fpr, tpr, threshols = roc_curve(y, pred)

# Dibujar la curva ROC
# Valor AUROC
valor_auroc = auc(fpr, tpr)
# Valores gráfico
fig_roc = plt.figure(figsize=(9, 9))
plt.title('Receiver Operating Characteristic (ROC)')
# Plot ROC curve
plt.plot(fpr, tpr, label="AUROC: " + "{0:.4}".format(valor_auroc))
plt.legend(loc='lower right')
# Diagonal 45 degree line
plt.plot([0, 1], [0, 1], 'k--')
# Axes limits and labels
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.show()
# Salvar gráfico
fig_roc.savefig('ROC_curve')


# ## Predicción 2023

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


# PREDICCIÓN 2023
# Cargar datos para predicción
df_2023 = pd.read_excel('datos_para_prediccion_2023.xlsx')
titulos_2023 = df_2023.titulo
X_2023 = df_2023.drop('titulo', axis=1)

# Variables cuantitativas en porcentaje anual
v_c = ['nominaciones', 'puntuacion', 'votos', 'criticas']
v_d = ['feroz_comedia_nominado',
       'forque_ganador',
       'forque_nominado',
       'cec_ganador',
       'G_med_bajo',
       'forque_nan']

X_2023 = pd.concat([X_2023[v_c] / X_2023[v_c].sum(), X_2023[v_d]], axis=1)

# Estandarización (se usa 'scaler_X' del modelo completo)
X_2023_std = scaler_X.transform(X_2023)

# Predicción de probabilidades
pred_2023 = mod_final_goyas.predict_proba(X_2023_std)
pred_2023 = pred_2023[:, 1]

# Ponderación probabilidades sobre 100 (lineal)
probs = pred_2023 / pred_2023.sum()

# Mostrar probabilidades ponderadas sobre 100 para cada título
print("\nPredicciones películas cargadas...\n")
tamano = 1 + len(titulos_2023.max())  # Longitud del título más largo para luego rellenar con espacios.
for tit, pre, prob in zip(titulos_2023, pred_2023, probs):
    print(tit.ljust(tamano), "\t", "{0:.1%}".format(pre), "\t", "{0:.1%}".format(prob))

# Gráfico probabilidades nominados 2023
# Data frame
n_2023 = pd.DataFrame({'titulo': titulos_2023, 'prob_absoluta': pred_2023, 'prob_edicion': probs})
n_2023 = n_2023.sort_values(by=['prob_edicion'], ascending=False)
n_2023.index = list(range(len(n_2023)))
# Gráfico dimensiones y supra-título
fig, ax = plt.subplots(figsize=(11, 9), facecolor='w')
fig.suptitle('Premios Goya - Mejor Película 2023', fontsize=14, fontweight='bold')
clrs = ['aqua', 'yellow', 'red', 'green', 'orange']
plt.bar(n_2023.titulo, n_2023.prob_edicion, align='center', alpha=0.6, color=clrs)
# Eje x. Título
plt.ylabel('Probabilidad de ganar')
plt.title('¿QUIÉN GANARÁ EL GOYA?')
# Etiquetas de datos
for i, fila in n_2023.iterrows():
    plt.text(x=i, y = 0.03, s="{0:.1%}".format(fila.prob_edicion), 
             size=16, horizontalalignment='center', color='black')    
plt.show()
# Salvar gráfico
fig.savefig('Probs_Goya_2023')

