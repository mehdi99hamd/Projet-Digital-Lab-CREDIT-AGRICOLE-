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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

import webbrowser
from threading import Timer

import model


def generate_columns(df):
    children = []
    i = 0
    if 'Unnamed: 0' in df.columns :
        df =  df.drop(['Unnamed: 0'], axis = 1)
    for col in df.columns : 
        children.append(html.P([html.I(className="fas fa-columns icons"), ' ' + col], className="icon-a", id="column" + str(i), n_clicks=0))
        children.append(dcc.Input(id="collumn" + str(i), type="hidden", placeholder="", debounce=True, value = col))
        i += 1
    return children

def show(df, filename) :
    if 'Unnamed: 0' in df.columns :
        df =  df.drop(['Unnamed: 0'], axis = 1)
    df = np.round(df,decimals=2)
    df1 = df.iloc[:50,:]
    children = [
    html.P(filename[:-4], className='dataname'),
    html.Div([
    dash_table.DataTable(
        data = df1.to_dict('records'),
        columns = [{'name': i, 'id': i} for i in df1.columns],
         style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['Date', 'Region']
        ],
        style_data={
            'color': 'white',
            'backgroundColor': '#1EA58C',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#176D5D',
            }
        ],
        style_header={
        'backgroundColor': '#064338',
        'color': 'white',
        'fontWeight': 'bold'
        },
        style_table={'overflowX': 'auto', 'border-collapse': 'collapse', 'width':'98%', 'overflowY': 'auto', 'height' : '40%'},
        page_size=10
    )], className='table')
    ]
    return children

def show1(df, filename) :
    df1 = df.copy()
    children = [
    html.P(filename[:-4], className='dataname'),
    html.Div([
    dash_table.DataTable(
        data = df1.to_dict('records'),
        columns = [{'name': i, 'id': i} for i in df1.columns],
        style_data={
            'color': 'white',
            'backgroundColor': '#1EA58C',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd' , 'column_id' : c},
                'backgroundColor': '#176D5D',
            } for c in list(df1.columns) if c not in ['Nv_Risque']
        ] + [
            {
            'if': {
                'filter_query': '{Nv_Risque} = "Très Faible"',
                'column_id': 'Nv_Risque',
                
            },
            'backgroundColor': '#1EA58C',
            'color': 'white'
        }, 
        {
            'if': {
                'filter_query': '{Nv_Risque} = "Faible"',
                'column_id': 'Nv_Risque',
                
            },
            'backgroundColor': '#CFC50A',
            'color': 'white'
        }, 
        {
            'if': {
                'filter_query': '{Nv_Risque} = "Moyen"',
                'column_id': 'Nv_Risque',
                
            },
            'backgroundColor': '#E9A90D',
            'color': 'white'
        }, 
        {
            'if': {
                'filter_query': '{Nv_Risque} = "Fort"',
                'column_id': 'Nv_Risque',
                
            },
            'backgroundColor': '#EF220A',
            'color': 'white'
        }, 
        ],
        style_header={
        'backgroundColor': '#064338',
        'color': 'white',
        'fontWeight': 'bold'
        },
        style_table={'overflowX': 'auto', 'border-collapse': 'collapse', 'width':'98%', 'overflowY': 'auto', 'height' : '40%'},
        page_size=10
    )], className='table')
    ]
    return children

fig = go.Figure()
fig.update_layout(height=550, width=530)

fig1 = go.Figure()
fig1.update_layout(height=550, width=950)

liste_graph = ['Evolution Du Phénomène de subtitution', "Evolution du CA par rapport Par Type D'Ass", "Montant du Pret Par Type D'Ass", 
"Age Par Type D'Ass", "Evolution des Nombre D'Assurance Externe", "Nombre de Connexion Par Type D'Ass", "Durée Restante du Projet par Type D'Ass", 
"Anciennte Par Type D'Ass", "Date de signature par type D'Ass", "PNB par type D'Ass"]

liste_classifier = ['Random Forest', 'Extratrees', 'Xgboost']

external_stylesheets = [dbc.themes.BOOTSTRAP, 
    "https://use.fontawesome.com/releases/v5.0.6/css/all.css",
]

external_scripts = [
    {'src': "https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"}
]

hidden = ' hidden'

visible = ' visible'

color = ['rgb(25, 195, 164)', 'rgb(13, 129, 108)', 'rgb(6, 67, 56)', 'rgb(99, 172, 145)']

def find_data_file(filename):
    if getattr(sys, 'frozen', False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)

port =  8050 # or simply open on the default `8050` port

def open_browser():
	webbrowser.open_new("http://localhost:{}".format(port))


app = dash.Dash(__name__, assets_folder=find_data_file('assets/'), external_scripts = external_scripts, 
                external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.title='Crédit Analyse'

server = app.server

app.layout = html.Div([html.Div([
    html.Div([
            html.Img(src='/assets/image.png', className="logo-img"),
            html.H1('Analyse Des Données', className='header11'), 
            html.Img(src='/assets/image2.png', className="logo-img2"),
        ], className='nav'),
    html.Div([
        html.Div([
            html.Div([]),
            html.Div([]),
            html.Div([]),
            html.Div([]),
            html.Div([]),
            html.Div([]),
            html.Div([]),
            html.Div([]),
            html.Div([]),
            html.Div([]),
        ], className = 'box-animated'),
        html.Div([
            html.Div([], className = 'design1'),
            html.Div([], className = 'design2'),
        ], className = 'design'),
        html.Div([
            html.Div([
                html.Div([html.P("Téléchargement Des Fichiers")], className="button-upload1"),
                dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            [html.I(className="fas fa-upload icon"),html.Br(),"Glisser Et Déposer Le Fichier Ici",
                            html.Br(),html.Span("Ou"),
                            html.Br(), html.Button('Selectioner Les Fichiers', className="button-upload2")]
                        ), multiple=False, className="upload-data"
                ),
                html.Div([], id = 'output-tt', className="button-upload-e572 hidden"),
            ], className="left"),
            html.Div([
                html.Div([html.P("Lecture Du Chemin Des Fichiers")], className="button-upload"),
                html.H5("Entrer Le Chemin Des Fichiers", className="title1"),
                dcc.Input(id="input2", type="text", placeholder="", debounce=True, className="input-path"),
                html.Button('Prediction', id='prediction',className="start", n_clicks = 0),
                html.Button('Suivant', id='start',className="start1", n_clicks = 0),
                html.Div([], id = 'error1', className = 'error1'),
            ], className="right")
        ],id = 'contact-box', className="contact-box visible"),
    ], className = "container"),
], className ="body1", id = 'body1'),

###############################################2eme page #######################################################################################################
html.Div([
    html.Div([
        html.P(['Colonnes', html.I(className = 'fa fa-bars menu1', n_clicks = 0, id='menu1')], className = 'logo1', id = 'logo1'),
        html.P([html.I(className="fa fa-bars menu", id = 'menu', n_clicks = 0)], className = 'logo', n_clicks = 0, id='logo'),
        html.Div(children = [], className = 'columns', id = 'generate_columns')
    ], className = "mySidenav", id = 'mySidenav'),
    
    
    
    html.Div([
        html.Div([
            html.Img(src='/assets/image.png', className="logo-img"),
            html.H1('Analyse Des Données', className='header'), 
            html.Img(src='/assets/image2.png', className="logo-img2"),
        ], className='nav1', id = 'nav'),
        
        html.Br(),
        
        html.Div([], className='clearfix'),

        html.Br(),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='crossfilter-xaxis-column',
                    options = [{
                        'label': i,
                        'value': i
                    } for i in liste_graph],
                    className = 'dropdown'
                )
            ], className = 'drop')
        ], className = 'box blanc', id = 'box5', n_clicks = 0),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='crossfilter-xaxis-column1',
                    options = [{
                        'label': i,
                        'value': i
                    } for i in liste_graph],
                    className = 'dropdown'
                )
            ], className = 'drop')
        ], className = 'box blanc', id = 'box3', n_clicks = 0),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='crossfilter-xaxis-column2',
                    options = [{
                        'label': i,
                        'value': i
                    } for i in liste_graph],
                    className = 'dropdown'
                )
            ], className = 'drop')
        ], className = 'box blanc', id = 'box4', n_clicks = 0),

        
        html.Div([
             html.Div([dcc.Graph(figure = fig,id ='graph')],className = 'content-cards'),
             html.Div([dcc.Graph(figure = fig,id ='graph1')],className = 'content-cards1'),
        ], className = "cards", id = "card"),

        html.Div([
             html.Div([dcc.Graph(figure = fig1,id ='graph2')],className = 'content-cards3')
        ], className = "cards", id = "card1"),

        

        html.Div([], className='clearfix'),

        html.Br(),
        dcc.Input(type = 'hidden', id = 'number_of_graphs', value = 0),
        html.Div([], id = 'graphs'),

        html.Div([], className='clearfix'),

        html.Br(),

        html.Div([
            html.Div([
                html.Button("Télécharger", id='download1',className="finish1", n_clicks=0),
                Download(id="download-dataframe-csv"),
                html.Div(children = [], className = 'content-box', id = 'showdata')
            ], className = 'box-8')
        ], className = 'table-list'),
        
        html.A([html.Button("Revenir à L'acceuil", id='home1',className="next1", n_clicks=0)],href='/'),
        html.Button('Prédiction', id='next',className="next", n_clicks = 0),
    ], className = 'main', id = 'main')
], className ="body hidden", id = 'body'),


###############################################Page Prediction###################################################################################""
html.Div([
    html.Div([
        html.P(['Colonnes', html.I(className = 'fa fa-bars menu1', n_clicks = 0, id='menu2')], className = 'logo1', id = 'logo2'),
        html.P([html.I(className="fa fa-bars menu", id = 'menu3', n_clicks = 0)], className = 'logo', n_clicks = 0, id='logo3'),
        html.Div(children = [], className = 'columns', id = 'generate_columns22')
    ], className = "mySidenav"),
    
    
    
    html.Div([
        html.Div([
            html.Img(src='/assets/image.png', className="logo-img"),
            html.H1('Prédiction Des Substitutions', className='header'), 
            html.Img(src='/assets/image2.png', className="logo-img2"),
        ], className='nav1', id = 'nav5'),
        html.Br(),
        
        html.Div([], className='clearfix'),

        html.Br(),

        html.Div([
            dcc.RadioItems(
                id='choosing',
                options = [{
                        'label': i,
                        'value': i
                } for i in ['Télécharger Les Données', 'Entrer Les Données Manuellement']],
               value = 'Télécharger Les Données',
               className = 'choosing'
            )
        ], className="form-control"),

        html.Br(),

        html.Div([
            dcc.Upload(
                    id="upload-data1",
                    children=html.Div(
                        [html.I(className="fas fa-upload iconnn"),"Glisser Et Déposer Le Fichier Ici", 
                        html.Span("Ou"), html.Button('Selectioner Les Fichiers', className="button-upload22")]
                    ), multiple=False, className="upload-data11"
            ),
            html.Div([], id = 'output-tt1', className="button-upload-e5722 hidden"),
        ], className="left1",  id = 'telech'),

        html.Div([
            dcc.Input(id='my-input1', placeholder='Montant Pro Echeance', type='text', className="input-path1"),
            dcc.Input(id='my-input2', placeholder='Montant Accordé du Pret', type='text', className="input-path1"), 
            dcc.Input(id='my-input3', placeholder='Durée Restante', type='text', className="input-path1"), 
            dcc.Input(id='my-input4', placeholder='Taux Crédit', type='text', className="input-path1"), 
            dcc.Input(id='my-input5', placeholder='Anciennté en Mois', type='text', className="input-path1"), 
            dcc.Input(id='my-input6', placeholder='Age', type='text', className="input-path1"), 
            dcc.Input(id='my-input7', placeholder='Nbr Mvt DBTV DAV 12M', type='text', className="input-path1"), 
            dcc.Input(id='my-input8', placeholder='Nbr De Connex 12M', type='text', className="input-path1"), 
            dcc.Input(id='my-input9', placeholder='PNB', type='text', className="input-path1"), 
            dcc.Input(id='my-input0', placeholder="Chiffre D'affaire Remisé", type='text', className="input-path1"), 
        ], className="left2 hidden", id = 'manuelle'),
        
        html.Br(),
        
        html.Div([], className='clearfix'),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='model',
                    options = [{
                        'label': i,
                        'value': i
                    } for i in liste_classifier],
                    className = 'dropdown1'
                )
            ], className = 'drop1'),
        ], className = 'box88 blanc', n_clicks = 0),


        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='type_model', 
                    options = [{
                            'label': i,
                            'value': i
                    } for i in ['Stock', 'Flux']],
                    value = 'Stock',
                    className = 'type_model'
                )
            ], className="form-control1"),
        ], className = 'box888 blanc', n_clicks = 0),

        html.Div([
            html.Button("Lancer", id='lancer',className="finish11", n_clicks=0)
        ], className = 'box8888 blanc', n_clicks = 0),

        html.Div([], className='clearfix'),

        html.Br(),

        html.Div([
            html.Div([
                html.Button("Télécharger", id='download2',className="finish1", n_clicks=0),
                Download(id="download-dataframe-csv1"),
                html.Div(children = [], className = 'content-box', id = 'showdata1')
            ], className = 'box-8')
        ], className = 'table-list'),

        html.Div([], className='clearfix'),

        html.Br(),
        
        html.Div([
             html.Div([dcc.Graph(figure = fig,id ='graph3')],className = 'content-cards'),
             html.Div([dcc.Graph(figure = fig,id ='graph4')],className = 'content-cards1'),
        ], className = "cards", id = "card5"),

        html.Div([
             html.Div([dcc.Graph(figure = fig1,id ='graph5')],className = 'content-cards3')
        ], className = "cards", id = "card6"),

        html.Div([], className='clearfix'),

        html.Br(),
        html.A([html.Button("Revenir à L'acceuil", id='home',className="next", n_clicks=0)],href='/'),

        
    ], className = 'main')
], className ="body hidden", id = 'body3'),


dcc.Store(id='data_up3'), 
dcc.Store(id='filename_up3'), 

dcc.Store(id='data_up4'), 
dcc.Store(id='filename_up4'),


dcc.Store(id='data_up5'), 
dcc.Store(id='filename_up5'),



dcc.Store(id='data_up'), 
dcc.Store(id='filename_up'), 

dcc.Store(id='data_up1'), 
dcc.Store(id='filename_up1'),


dcc.Store(id='data_up2'), 
dcc.Store(id='filename_up2'),
















])


@app.callback(
    Output('showdata1', 'children'),
    Output('data_up5', 'data'),
    Output('filename_up5', 'data'),
    Output('graph3', 'figure'),
    Output('graph4', 'figure'),
    Output('graph5', 'figure'),
    State('data_up3', 'data'),
    State('filename_up3', 'data'),
    State('model', 'value'),
    State('type_model', 'value'),
    State('upload-data1', 'filename'),
    State('choosing', 'value'),
    State('my-input1', 'value'),
    State('my-input2', 'value'),
    State('my-input3', 'value'),
    State('my-input4', 'value'),
    State('my-input5', 'value'),
    State('my-input6', 'value'),
    State('my-input7', 'value'),
    State('my-input8', 'value'),
    State('my-input9', 'value'),
    State('my-input0', 'value'),
    Input('lancer', 'n_clicks'))

def predict(data, columns, modell, type, filename, choosing, input1,input2, input3, input4, input5, input6, input7, input8, input9, input0,  click) :
    children = []
    inputt = [input1,input2, input3, input4, input5, input6, input7, input8, input9, input0]
    if choosing == 'Entrer Les Données Manuellement' :
        if None not in inputt and click > 0 :
            inputt = [int(i) for i in inputt]
            df = pd.DataFrame([inputt])
            df.columns = ['MT_PROCH_ECHCE_THRQ','MT_ACCORDE_PRET_PAR_CR', 'DUREE_RESTANTE','TX_CRED', 'ANCIENNETE_MOIS', 'AGE','NB_MVT_DBTR__DAV_12M','NB_CONX_WEB_12M',
            'PNB','CA_REMISE_T']
            filename = 'projet_x.csv'
            df = model.testing(df, modell, type)
            children = show1(df, filename)
            fig1, fig2, fig3 = model.graphing(df)
            columns1 = df.columns
            df = df.values
            return children, df, columns1, fig1, fig2, fig3
    else :
        if click > 0 and data is not None :
            df = pd.DataFrame(data)
            df.columns = columns
            df = model.testing(df, modell, type)
            children = show1(df, filename)
            fig1, fig2, fig3 = model.graphing(df)
            columns1 = df.columns
            df = df.values
            return children, df, columns1, fig1, fig2, fig3
        else : 
            df = None
            columns1 = None
            fig1, fig2, fig3 = go.Figure(), go.Figure() , go.Figure()
            return children, df, columns1, fig1, fig2, fig3

@app.callback(
    Output("download-dataframe-csv1", "data"),
    State('data_up5', 'data'),
    State('upload-data1', 'filename'),
    State('filename_up5', 'data'),
    Input("download2", "n_clicks"),
    prevent_initial_call=True,
)
def func(data, filename, columns,  n_clicks):
    if data is not None and n_clicks > 0 :
        df = pd.DataFrame(data)
        df.columns = columns
        return send_data_frame(df.to_csv, filename[:-4] + "_predicted.csv")


@app.callback(
    Output('telech', 'className'),
    Output('manuelle', 'className'),
    Input('choosing', 'value'))

def update_choosing(type) :
    class_left1 = 'left1'
    class_left2 = 'left2 hidden'
    if type is not None :
        if type == 'Entrer Les Données Manuellement' :
            class_left1 = 'left1 hidden'
            class_left2 = 'left2'
            return class_left1, class_left2
        else : 
            return class_left1, class_left2
    else : 
        return class_left1, class_left2


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename :
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        return None


@app.callback(Output('output-tt1', 'children'),
              Output('output-tt1', 'className'),
              Output('data_up3', 'data'),
              Output('filename_up3', 'data'),
              State('upload-data1', 'contents'),
              Input('upload-data1', 'filename'))

def update_output1(contents, filename):
    children = []
    class_file = "button-upload-e5722 hidden"
    if contents is not None :
        if 'csv' in filename  or "xls" in filename:
            class_file = 'button-upload-e5722'
            children.append(html.P("Le Fichier " + filename + " est Téléchargé"))
            df = parse_contents(contents, filename)
            data = list(df.values)
            columns = list(df.columns)
            return  children, class_file, data, columns
        else :
            children.append(html.P("Veuillez Choisir Un Fichier csv Ou xls"))
            class_file = 'background-red1'
            data = None
            columns = None
            return  children, class_file, data, columns
    else : 
        data = None
        columns = None
        class_file = 'button-upload-e5722 hidden'
        return  children, class_file, data, columns










    
@app.callback(
    Output('graph', 'figure'),
    State('data_up', 'data'),
    State('filename_up', 'data'),
    Input('crossfilter-xaxis-column', 'value'))

def update_graph1(data, columns,type) :
    if type is not None :
        df = pd.DataFrame(data)
        df.columns = columns
        fig = model.generate_dataset_graph(df, type)
        return fig
    else : 
        return go.Figure()
        
@app.callback(
    Output('graph1', 'figure'),
    State('data_up', 'data'),
    State('filename_up', 'data'),
    Input('crossfilter-xaxis-column1', 'value'))

def update_graph2(data, columns, type) :
    if type is not None :
        df = pd.DataFrame(data)
        df.columns = columns
        fig = model.generate_dataset_graph(df, type)
        return fig
    else : 
        return go.Figure()



@app.callback(
    Output('graph2', 'figure'),
    State('data_up', 'data'),
    State('filename_up', 'data'),
    Input('crossfilter-xaxis-column2', 'value'))

def update_graph3(data, columns,type) :
    if type is not None :
        df = pd.DataFrame(data)
        df.columns = columns
        fig = model.generate_dataset_graph(df, type)
        return fig
    else : 
        return go.Figure()

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename :
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        return None

@app.callback(
    Output("download-dataframe-csv", "data"),
    State('data_up', 'data'),
    State('upload-data', 'filename'),
    State('filename_up', 'data'),
    Input("download1", "n_clicks"),
    prevent_initial_call=True,
)
def func(data, filename, columns,  n_clicks):
    if data is not None and n_clicks > 0 :
        df = pd.DataFrame(data)
        df.columns = columns
        return send_data_frame(df.to_csv, filename[:-4] + "_projet.csv")

@app.callback(Output('output-tt', 'children'),
              Output('output-tt', 'className'),
              Output('data_up', 'data'),
              Output('filename_up', 'data'),
              State('upload-data', 'contents'),
              Input('upload-data', 'filename'),
              State('output-tt', 'className'))

def update_output1(contents, filename, classes):
    children = []
    if contents is not None :
        if 'csv' or "xls" in filename:
            class_file = 'button-upload-e572'
            children.append(html.P("Le Fichier " + filename + " est Téléchargé"))
            df = parse_contents(contents, filename)
            data = list(df.values)
            columns = list(df.columns)
            return  children, class_file, data, columns
        else :
            children.append(html.P("Veuillez Choisir Un Fichier csv Ou xls"))
            class_file = 'background-red'
            data = None
            columns = None
            data1 = None
            columns1 = None
            data2 = None
            columns2 = None
            return  children, class_file, data, columns
    else : 
        data = None
        columns = None
        class_file = 'button-upload-e572 hidden'
        return  children, class_file, data, columns

@app.callback(Output('error1', 'children'),
             Output('body', 'className'),
             Output('body1', 'className'),
              Output('body3', 'className'),
             Output('generate_columns', 'children'),
             Output('showdata', 'children'),
             Output('prediction', 'n_clicks'),
             Output('next', 'n_clicks'),
             State('data_up', 'data'),
             State('upload-data', 'filename'),
             State('filename_up', 'data'),
              State('input2', 'value'),
              Input('start', 'n_clicks'),
              Input('prediction', 'n_clicks'),
              Input('next', 'n_clicks'))

def update_output2(data, filename,columns, path, click, click1,click2):
    class_body1 = 'body1'
    class_body = 'body hidden'
    class_body3 = 'body hidden'
    children = []
    children2 = []
    children3 = []
    if click2 > 0 :
        class_body = 'body1 hidden'
        class_body3 = 'body'
        class_body1 = 'body1 hidden'
        click2 = 0
        return children, class_body , class_body1,class_body3,  children2, children3, click1, click2
    elif click1 > 0 : 
        class_body1 = 'body1 hidden'
        class_body3 = 'body'
        click1 = 0
        return children, class_body , class_body1,class_body3,  children2, children3, click1, click2
    elif data is not None and click > 0:
        if 'csv' or 'xls' in filename :
            df = pd.DataFrame(data)
            df.columns = columns
            class_body1 = 'body1 hidden'
            class_body = 'body'
            children2 = generate_columns(df)
            children3 = show(df, filename)
            children4 = list(df.values)
            children5 = list(df.columns)
            return children, class_body , class_body1,class_body3,  children2, children3, click1, click2
        else :
            children = [dbc.Alert("Veuillez Entrer Un Fichier valide", color="danger")]
            return children, class_body , class_body1,class_body3,  children2, children3, click1, click2
    elif data is None and path is None and click > 0:
            children = [dbc.Alert("Veuillez Selectionner Un Fichier", color="danger")]
            return children, class_body , class_body1, class_body3, children2, children3, click1, click2
    elif path is None and click > 0:
        children = [dbc.Alert("Veuillez Sélectionner Le Chemin De Sortie Des Fichiers !", color="danger")]
        return  children, class_body , class_body1,class_body3,  children2, children3, click1, click2
    elif path is not None and click > 0:
        output = str(path)
        output = output.replace('\\','/') 
        if os.path.exists(output) :
            class_body1 = 'body1 hidden'
            class_body = 'body'
            return children, class_body , class_body1,class_body3,  children2, children3, click1, click2
        else : 
            children = [dbc.Alert("Veuillez Entrer Un Chemin Valide", color="danger")]
            return children, class_body , class_body1,class_body3,  children2, children3, click1, click2
    else :
        return children, class_body , class_body1,class_body3,  children2, children3, click1, click2

@app.callback(Output('menu1', 'className'),
             Output('menu', 'className'),
             Output('main', 'className'),
             Output('generate_columns', 'className'),
             Output('nav', 'className'),
             Output('logo', 'className'),
             Output('logo1', 'className'),
             Output('mySidenav', 'className'),
             Output('menu1', 'n_clicks'),
             Output('menu', 'n_clicks'),
              Input('menu', 'n_clicks'),
              Input('menu1', 'n_clicks'))

def update_menu(click, click2):
    menu1 = 'fa fa-bars menu1'
    menu = 'fa fa-bars menu'
    main = 'main'
    columns = 'columns'
    nav = 'nav1'
    logo = 'logo'
    logo1 = 'logo1'
    mySidenav = 'mySidenav'
    if click > 0:
        menu1 = 'fa fa-bars menu2'
        menu = 'fa fa-bars menu1'
        main = 'main1'
        columns = 'columns1'
        nav = 'nav2'
        logo = 'logo1'
        logo1 = 'logo'
        mySidenav = 'mySidenav1'
        click = 0
        return menu1, menu, main, columns, nav, logo, logo1, mySidenav, click2, click
    elif click2 > 0:
        click2 = 0
        return menu1, menu, main, columns, nav, logo, logo1, mySidenav, click2, click
    else :
        return menu1, menu, main, columns, nav, logo, logo1, mySidenav, click2, click




if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug = True)