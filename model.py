import base64
import datetime
import io
from multiprocessing.sharedctypes import Value
from pydoc import classname

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
from dash import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import grasia_dash_components as gdc
from dash import Dash, html, Input, Output, callback_context, State
import numpy as np
import os
import sys
from collections import OrderedDict
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.graph_objs as go
from functools import partial
from dash import callback
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


filename = 'random_forest_stock.sav'
classifier_r_stock = pickle.load(open(filename, 'rb'))
filename1 = 'extratrees_stock.sav'
classifier_e_stock = pickle.load(open(filename1, 'rb'))
filename2 = 'xgboost_stock.sav'
classifier_x_stock = pickle.load(open(filename2, 'rb'))

filenamee = 'random_forest_flux.sav'
classifier_r_flux = pickle.load(open(filenamee, 'rb'))
filename11 = 'extratrees_flux.sav'
classifier_e_flux  = pickle.load(open(filename11, 'rb'))
filename22 = 'xgboost_flux.sav'
classifier_x_flux = pickle.load(open(filename22, 'rb'))


sc2_path = 'scaler_flux.pkl'
sc2  = pickle.load(open(sc2_path, 'rb'))
sc1_path = 'scaler_stock.pkl'
sc1  = pickle.load(open(sc1_path, 'rb'))

liste_classifier = ['Random Forest', 'Extratrees', 'Xgboost']

dict_classifier_stock = {
    'Random Forest' : classifier_r_stock,
    'Extratrees' : classifier_e_stock,
    'Xgboost' : classifier_x_stock,
}

dict_classifier_flux  = {
    'Random Forest' : classifier_r_flux ,
    'Extratrees' : classifier_e_flux ,
    'Xgboost' : classifier_x_flux ,
}

dict_classifier = {
    'Stock' : dict_classifier_stock,
    'Flux' : dict_classifier_flux,
}


def barcode(x):
    if x < 0.25 :
        return 'Très Faible'
    elif 0.25 <= x < 0.5 :
        return 'Faible'
    elif 0.5 <= x < 0.75 :
        return 'Moyen'
    else : 
        return 'Fort'




def testing(df, algo, type) : 
    if 'Unnamed: 0' in df.columns :
        df =  df.drop(['Unnamed: 0'], axis = 1)
    if 'id_projet' in df.columns :
        projet = df['id_projet']
        df = df.drop(['id_projet'], axis = 1)
    else :
        projet = [i for i in range(df.shape[0])]
    if 'BARCODE' in df.columns : 
        df = df.drop(['BARCODE'], axis = 1)
    if type == 'Stock' :
        df1 = df[['CA_REMISE_T', 'PNB', 'NB_CONX_WEB_12M', 'NB_MVT_DBTR__DAV_12M', 'AGE','ANCIENNETE_MOIS', 'TX_CRED', 'DUREE_RESTANTE',
        'MT_ACCORDE_PRET_PAR_CR', 'MT_PROCH_ECHCE_THRQ']]
        X = df1.values
        X1 = sc1.transform(X)
        cls = dict_classifier[type][algo]
        y_pred = cls.predict_proba(X1)
        df.insert(0, 'Risque', y_pred[:, 1])
        df.insert(0, 'Nv_Risque', y_pred[:, 1])
        df = df.sort_values(by=['Nv_Risque'], ascending = False)
        df['Nv_Risque'] = df['Nv_Risque'].apply(lambda x : barcode(x))
        df.insert(0, 'id_projet', projet)
        df = np.round(df,decimals=2)
    else : 
        df1 = df[['CA_REMISE_T', 'PNB', 'NB_CONX_WEB_12M', 'NB_MVT_DBTR__DAV_12M','ANCIENNETE_MOIS', 'TX_CRED', 'DUREE_RESTANTE',
        'MT_ACCORDE_PRET_PAR_CR']]
        X = df1.values
        X1 = sc2.transform(X)
        cls = dict_classifier[type][algo]
        y_pred = cls.predict_proba(X1)
        df.insert(0, 'Risque', y_pred[:, 1])
        df.insert(0, 'Nv_Risque', y_pred[:, 1])
        df = df.sort_values(by=['Nv_Risque'], ascending = False)
        df['Nv_Risque'] = df['Nv_Risque'].apply(lambda x : barcode(x))
        df.insert(0, 'id_projet', projet)
        df = np.round(df,decimals=2)
    return df

def mt(x) :
    if x < 50000 :
        return '< 50K'
    elif 50000 <= x < 2e5 :
        return 'Entre 50K et 200K'
    elif 2e5 <= x < 5e5 :
        return 'Entre 200K et 500K'
    else : 
        return '> 500K'





def graphing(df) :
    color = ['rgb(25, 195, 164)', 'rgb(13, 129, 108)', 'rgb(6, 67, 56)', 'rgb(99, 172, 145)']
    fig1 = px.histogram(df, x="Nv_Risque", color="Nv_Risque", color_discrete_sequence=color,title ='Nombre de projet Pour Chaque Catégorie') 

    data3 = df[['MT_ACCORDE_PRET_PAR_CR', 'Nv_Risque']]
    data3['MT'] = data3['MT_ACCORDE_PRET_PAR_CR'].apply(lambda x : mt(x))

    x,y =  'MT', 'Nv_Risque'

    data3 = data3.groupby(y)[x].value_counts()
    data3 = data3.rename('Pourcentage').reset_index()
    data3['Pourcentage']=data3['Pourcentage'].apply(lambda x:round(x,2))
    fig2 = px.bar(data3, x=y, y='Pourcentage', color = x, text='Pourcentage',color_discrete_sequence=color, hover_data={'Pourcentage':':.2f'},
                title ='Categories des Montant par Niveau Risque')

    data77 = df.groupby(['Nv_Risque'])['CA_REMISE_T'].sum()
    data77 = data77.rename('Sum_CA').reset_index()
    fig3 = px.bar(data77, x='Nv_Risque', y='Sum_CA', color =  'Nv_Risque', color_discrete_sequence=color,hover_data={'Sum_CA':':.2f'},
             title = "Somme de Chiffre D'affaires Pour Chaque Catégorie")
    return fig1, fig2, fig3
    


liste_graph = ['Evolution Du Phénomène de subtitution',"Evolution des Nombre D'Assurance Externe", "Evolution du CA par rapport Par Type D'Ass",
 "Montant du Pret Par Type D'Ass", "Age Par Type D'Ass",  "Nombre de Connexion Par Type D'Ass", "Durée Restante du Projet par Type D'Ass", 
"Anciennte Par Type D'Ass", "Date de signature par type D'Ass", "PNB par type D'Ass"]


def age(x) :
    if x < 30 :
        return '< 30 Ans'
    elif 30 <= x < 45 :
        return 'Entre 30 et 45 Ans'
    elif 45 <= x < 60 :
        return 'Entre 45 et 60 Ans'
    else : 
        return '> 60 Ans'

def web(x) :
    if x < 100 :
        return '< 100 Connexions'
    elif 100 <= x < 250 :
        return 'Entre 100 et 250 Connexions'
    elif 250 <= x < 500 :
        return 'Entre 250 et 500 Connexions'
    else : 
        return '> 500 Connexions'

def duree(x) :
    if x < 50 :
        return '< 50'
    elif 50 <= x < 150 :
        return 'Entre 50 et 150'
    elif 150 <= x < 250 :
        return 'Entre 150 et 250'
    else : 
        return '> 250'

def An(x) :
    if x < 100 :
        return '< 100 Mois'
    elif 100 <= x < 200 :
        return 'Entre 100 et 200 Mois'
    elif 200 <= x < 500 :
        return 'Entre 200 et 500 Mois'
    else : 
        return '> 500 Mois'

def sign_crdt(x) :
    if x < 2015 :
        return 'Avant 2015'
    elif 2015 <= x < 2018 :
        return 'Entre 2015 et 2018'
    else : 
        return 'Apres 2018'

def pnb1(x) :
    if x < 0 :
        return '< 0'
    elif 0 <= x < 500 :
        return 'Entre 0 et 500'
    elif 500 <= x < 1000 :
        return 'Entre 500 et 1000'
    else : 
        return '> 1000'

def generate_dataset_graph(df, type):
    color = ['rgb(25, 195, 164)', 'rgb(13, 129, 108)', 'rgb(6, 67, 56)', 'rgb(99, 172, 145)']
    if type == 'Evolution Du Phénomène de subtitution' :
        data3 = df[df['DD_HISTO'] > 2009]
        x,y =  'BARCODE', 'DD_HISTO'
        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('percent').reset_index()
        df1[x] = df1[x].astype(str)
        fig = fig = px.line(df1, x=y, y='percent',color = x,color_discrete_sequence=color, title = type)
        return fig
    elif type == "Evolution des Nombre D'Assurance Externe" :
        data3 = df[df['DD_HISTO'] > 2009]
        x,y =  'BARCODE', 'DD_HISTO'
        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('percent').reset_index()
        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='percent', color = x, color_discrete_sequence=color,hover_data={'percent':':.2f'},
             title = type)
        return fig
    elif type == "Evolution du CA par rapport Par Type D'Ass" :
        data3 = df[df['DD_HISTO'] > 2009]
        data77 = data3.groupby(['DD_HISTO', 'BARCODE'])['CA_REMISE_T'].sum()
        data77 = data77.rename('sum_CA').reset_index()
        data77['BARCODE'] = data77['BARCODE'].astype(str)
        fig = px.bar(data77, x='DD_HISTO', y='sum_CA', color =  'BARCODE', color_discrete_sequence=color,hover_data={'sum_CA':':.2f'},
                    title = type)
        return fig
    elif type == "Montant du Pret Par Type D'Ass" :
        data3 = df[['MT_ACCORDE_PRET_PAR_CR', 'BARCODE']]
        data3['MT'] = data3['MT_ACCORDE_PRET_PAR_CR'].apply(lambda x : mt(x))
        x,y =  'MT', 'BARCODE'

        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('num').reset_index()
        df1['num']=df1['num'].apply(lambda x:round(x,2))

        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='num', color = x, text='num',color_discrete_sequence=color, hover_data={'num':':.2f'},
                    title =type)
        return fig
    elif type == "Age Par Type D'Ass" :
        data3 = df[['AGE', 'BARCODE']]

        data3['Age'] = data3['AGE'].apply(lambda x : age(x))

        x,y =  'Age', 'BARCODE'

        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('num').reset_index()
        df1['num']=df1['num'].apply(lambda x:round(x,2))

        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='num', color = x, text='num',color_discrete_sequence=color, hover_data={'num':':.2f'},
                    title =type)
        return fig
    elif type == "Nombre de Connexion Par Type D'Ass" :
        data3 = df[['NB_CONX_WEB_12M', 'BARCODE']]

        data3['Connex'] = data3['NB_CONX_WEB_12M'].apply(lambda x : web(x))

        x,y =  'Connex', 'BARCODE'

        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('num').reset_index()
        df1['num']=df1['num'].apply(lambda x:round(x,2))

        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='num', color = x, text='num',color_discrete_sequence=color, hover_data={'num':':.2f'},
                    title =type)
        return fig
    elif type == "Durée Restante du Projet par Type D'Ass" :
        data3 = df[['DUREE_RESTANTE', 'BARCODE']]

        data3['Duree'] = data3['DUREE_RESTANTE'].apply(lambda x : duree(x))

        x,y =  'Duree', 'BARCODE'

        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('num').reset_index()
        df1['num']=df1['num'].apply(lambda x:round(x,2))

        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='num', color = x, text='num',color_discrete_sequence=color, hover_data={'num':':.2f'},
                    title =type)
        return fig
    elif type == "Anciennte Par Type D'Ass" :
        data3 = df[['ANCIENNETE_MOIS', 'BARCODE']]

        data3['ANCIENNETE'] = data3['ANCIENNETE_MOIS'].apply(lambda x : An(x))

        x,y =  'ANCIENNETE', 'BARCODE'

        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('num').reset_index()
        df1['num']=df1['num'].apply(lambda x:round(x,2))

        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='num', color = x, text='num',color_discrete_sequence=color, hover_data={'num':':.2f'},
                    title =type)
        return fig
    elif type == "Date de signature par type D'Ass" :
        data3 = df[df['DT_SIGNER_CTR_CRED'] > 2009]
        data3 = data3[['DT_SIGNER_CTR_CRED', 'BARCODE']]
        data3['sign_crt'] = data3['DT_SIGNER_CTR_CRED'].apply(lambda x : sign_crdt(x))
        x,y =  'sign_crt', 'BARCODE'

        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('num').reset_index()
        df1['num']=df1['num'].apply(lambda x:round(x,2))

        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='num', color = x, text='num',color_discrete_sequence=color,
                    title =type)
        return fig
    elif type == "PNB par type D'Ass" :
        data3 = df[['PNB', 'BARCODE']]

        data3['pnb'] = data3['PNB'].apply(lambda x : pnb1(x))

        x,y =  'pnb', 'BARCODE'

        df1 = data3.groupby(y)[x].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('num').reset_index()
        df1['num']=df1['num'].apply(lambda x:round(x,2))

        df1[x] = df1[x].astype(str)
        fig = px.bar(df1, x=y, y='num', color = x, text='num',color_discrete_sequence=color, hover_data={'num':':.2f'},
                    title =type)
        return fig
    
    