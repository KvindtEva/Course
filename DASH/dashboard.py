import numpy as np
import pandas as pd
import random
from gensim.models.doc2vec import Doc2Vec
from dash import Dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output

def get_random_sentence(dataframe):
    random.seed(42)
    return random.sample(dataframe, 10)

# STYLES
text_st = {'width': '60%'}
dropdown_st = {'padding-left': '50px', 'padding-right': '50px', }
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

# LOADING FILES
clustered_data = pd.read_csv('https://raw.githubusercontent.com/KvindtEva/Course/main/DASH/clustered%20data.csv')
df = clustered_data.drop_duplicates(subset='desc')
desc = list(clustered_data.desc)
tags = list(pd.read_csv('https://raw.githubusercontent.com/KvindtEva/Course/main/DASH/tags.csv')
            .tags)
freq_nouns = pd.read_csv('https://raw.githubusercontent.com/KvindtEva/Course/main/DASH/freq_dict_sorted.csv')
model = Doc2Vec.load('doc2vec with parametrs')

# MAKING GRAPHS
freq_nouns = freq_nouns[(freq_nouns.freq < 1000) & (freq_nouns.freq >= 10)]
fig_freq = px.bar(freq_nouns, x="token", y="freq", title="Частота слов (>10)",
                  labels={
                     "token": "words (tokens)",
                     "freq": "frequencies"},)
fig_clustered = px.scatter(clustered_data,
                           x="tsne_x", y="tsne_y", color="labels_affinity_prop",
                           labels={"tsne_x": "TSNE first coordinate",
                                   "tsne_y": "TSNE second coordinate",
                                   "labels_affinity_prop": "Категории"},
                           title="Кластеризация Affinity propagation", height=600)


# DASHBOARD
text_st = {'width': '60%'}
dropdown_st = {'padding-left': '50px', 'padding-right': '50px', }

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.Div(children=[

        html.H1(children='Данные'),
        html.Div(children='''
            В нашем распоряжении оказались описания киноплощадок, собранные с различных ресурсов по их поиску.
            Далее представлены примеры описаний и их лемматизированная форма (использование MyStem) для работы с 
            эмбеддингом. Примеры:
        ''', style=text_st),
        html.Br(),
        html.Div(children=[
            html.Table([
                html.Tr([html.Td(get_random_sentence(list(df.desc))[i]),
                         html.Td(get_random_sentence(list(df.desc_clear))[i])
                         ]) for i in range(10)
            ]),
        ], style=dropdown_st),
        html.Br(),
        html.Br(),
        html.Br(),

        html.H1(children='ChipSelect'),
        html.Div(children='''
            Рекомендательная система по типу chip select предлагает пользователю выбрать один из существующих тегов, 
            который его интересует. Например: лофт, квартира, гараж, аэропорт и прочее. Генерация тегов происходила
            в частности на основе наиболее часто встречающихся слов.
        ''', style=text_st),
        dcc.Graph(id="freq_graph", figure=fig_freq),
        html.Br(),
        html.Div(children='''
            Функционал MyStem позволяет определять часть речи, благодаря чему из всех токенов были выбраны наиболее
            частые существительные. Они и существительные после слов "...категория:" в описании некоторых площадок
            использованы в качестве тегов для chip select. Некоторые специфические слова (цао, москва, замоскворечный)
            вручную убраны из списка.
        ''', style=text_st),
        html.Br(),
        html.Div(children=[
            dcc.Dropdown(
                id='tags',
                options=[{'label': i, 'value': i} for i in tags],
                value=tags[1]
            ),
            html.Br(),
            html.Div(id='chip_select_recomendations', style={'whiteSpace': 'pre-line'}),
        ], style=dropdown_st),
        html.Br(),
        html.Br(),
        html.Br(),

                html.H1(children='Рекомендации на основе Doc2Vec'),
                html.Div(children='''
                    После построения эмбеддинга описаний, можно рекомендовать площадки на основе их векторной близости в
                    Doc2Vec модели. *объяснить, почему выбрали D2V и как ее обучали*
                ''', style=text_st),
                html.Br(),
                html.Div(children=[
                    dcc.Dropdown(
                                id='desc_doc2vec_recomendations',
                                options=[{'label': df.desc[i], 'value': df.desc[i]} for i in df.index],
                                value = df.desc[0]
                            ),
                    html.Br(),
                    html.Div(id='doc2vec_recomendations', style={'whiteSpace': 'pre-line'}),
                ], style=dropdown_st),
                html.Br(),
                html.Br(),
                html.Br(),

        html.H1(children='Рекомендации на основе кластеризации'),
        html.Div(children='''
            После применения эмбеддинга пространство признаков было уменьшено до 2х и кластеризировано. Так 
            получились кластеры описаний.
        ''', style=text_st),
        html.Br(),
        dcc.Graph(id="clustered_graph", figure=fig_clustered),
        html.Br(),
        html.Div(children=[
            dcc.Dropdown(
                id='desc_cluster_recomendations',
                options=[{'label': df.desc[i], 'value': df.desc[i]} for i in df.index],
                value=df.desc[0]
            ),
            html.Br(),
            html.Div(id='cluster_recomendations', style={'whiteSpace': 'pre-line'}),
        ], style=dropdown_st),
        html.Br()
    ], style={'padding': '50px'})

])


@app.callback(
    Output('chip_select_recomendations', 'children'),
    Input('tags', 'value'))
def update_output(value):
    recomendations = ''
    count = 0
    for i in df.index:
        if value in df.desc_clear[i].split():
            recomendations += df.desc[i] + '\n'
            count += 1
        if count == 5:
            break
    return recomendations


@app.callback(
    Output('doc2vec_recomendations', 'children'),
    Input('desc_doc2vec_recomendations', 'value'))
def update_output(value):
    i = df[df.desc == value].index[0]
    ind = model.dv.most_similar([model.infer_vector(df.desc_clear[i].split())], topn = 5)
    most_similar = np.unique([desc[j[0]] for j in ind])
    recomendations = ''
    for j in most_similar:
        recomendations += j + '\n'
    return recomendations

@app.callback(
    Output('cluster_recomendations', 'children'),
    Input('desc_cluster_recomendations', 'value'))
def update_output(value):
    cluster = df[df.desc == value].labels_kmeans.iloc[0]
    most_similar = df[df.labels_kmeans == cluster].desc.unique()
    recomendations = ''
    for j in most_similar[:5]:
        recomendations += j + '\n'
    return recomendations



# START
if __name__ == '__main__':
    app.run_server(debug=True)