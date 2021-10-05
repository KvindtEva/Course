import numpy as np
import pandas as pd
import random
from gensim.models.doc2vec import Doc2Vec
from dash import Dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from pymystem3 import Mystem
from nltk.corpus import stopwords
from string import punctuation
import dash

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
desc = list(pd.read_csv('https://raw.githubusercontent.com/KvindtEva/Course/main/DASH/desc_all.csv').desc)
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
                           x="tsne_x", y="tsne_y", color="labels_kmeans",
                           labels={"tsne_x": "TSNE first coordinate",
                                   "tsne_y": "TSNE second coordinate",
                                   "labels_kmeans": "Категории"},
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
                value=tags[0]
            ),
            html.Br(),
            html.Div(id='chip_select_recomendations', style={'whiteSpace': 'pre-line'}),
        ], style=dropdown_st),
        html.Br(),
        # html.Br(),
        # html.Br(),

        # html.H1(children='Предложение тегов'),
        html.Div(children='''
            Вы можете ввести свое описание искомой площадки и система порекомендует существующие теги для
            данного запроса
        ''', style=text_st),
        html.Br(),
        html.Div(children=[
            dcc.Textarea(
                    id='new_query_input',
                    value='Введите новый запрос',
                    style={'width': '60%'},
            ),
            html.Br(),
            html.Button('Получить теги', id='submit_tags', n_clicks=0),
            html.Br(),
            html.Div(id='tags_rec', style={'whiteSpace': 'pre-line'}),
        ], style=dropdown_st),
        html.Br(),
        html.Div(children='''
            Или выберите понравившееся описание и система предложит теги, по которым можно осуществить поиск
        ''', style=text_st),
        html.Br(),
        html.Div(children=[
            dcc.Dropdown(
                id='choose_desc_find_tags',
                options=[{'label': df.desc[i], 'value': df.desc[i]} for i in df.index],
                value=df.desc[0]
            ),
            html.Br(),
            html.Div(id='found_tags', style={'whiteSpace': 'pre-line'}),
        ], style=dropdown_st),
        html.Br(),
        html.Div(children='''
            Можно сменить принадлежность тегов к описаниям. Для этого выберите описание, тег и что хотите
            с ним сделать: удалить или присвоить к данному описанию
        ''', style=text_st),
        html.Br(),
        html.Div(children=[
            dcc.Dropdown(
                id='choose_desc_change_tags',
                options=[{'label': df.desc[i], 'value': df.desc[i]} for i in df.index],
                value=df.desc[0]
            ),html.Br(),
            dcc.Dropdown(
                id='choose_tag_change_tags',
                options=[{'label': i, 'value': i} for i in tags],
                value=tags[0]
            ),html.Br(),
            html.Button('Удалить тег', id='delete_tag', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Добавить тег', id='insert_tag', n_clicks=0),
            html.Br(),
            html.Div(id='change_tag_result', style={'whiteSpace': 'pre-line'}),
        ], style=dropdown_st),
        html.Br(),
        html.Br(),
        html.Br(),

        html.H1(children='Рекомендации на основе Doc2Vec'),
        html.Div(children='''
            После построения эмбеддинга описаний, можно рекомендовать площадки на основе их векторной близости в
            Doc2Vec модели.
        ''', style=text_st),
        html.Br(),
        html.Div(children=[
            dcc.Dropdown(
                        id='desc_doc2vec_recomendations',
                        options=[{'label': df.desc[i], 'value': df.desc[i]} for i in df.index],
                        value=df.desc[0]),
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
        html.Br(),
        html.Br(),
        html.Br(),


        html.H1(children='Смена кластера'),
        html.Div(children='''
            Для начала посмотрим на содержимое каждого кластера
        ''', style=text_st),
        html.Br(),
        html.Div(children=[
            dcc.Dropdown(
                id='clusters',
                options=[{'label': i,
                          'value': i} for i in np.sort(df.labels_kmeans.unique())],
                value=np.sort(df.labels_kmeans.unique())[0]
            ),
            html.Br(),
            html.Div(id='clusters_consist', style={'whiteSpace': 'pre-line'}),
        ], style=dropdown_st),
        html.Br(),
        html.Div(children='''
            Если вы хотите поменять кластер для какого-то из описаний, введите его номер, 
            а затем номер кластера, в который хотите переместить описание
        ''', style=text_st),
        html.Br(),
        dcc.Textarea(
                id='desc_num',
                value='Номер описания',
                style={'width': '20%', 'height': '10px', 'margin-right': '10px'},
        ),
        dcc.Textarea(
                id='cluster_num',
                value='Номер нового кластера \n(от 0 до 99)',
                style={'width': '20%', 'height': '10px'},
        ),
        html.Br(),
        html.Button('Сменить кластер', id='submit_changes', n_clicks=0),
        html.Br(),html.Br(),
        html.Div(id='done', style={'whiteSpace': 'pre-line'}),
        html.Br(),
        html.Br(),
        html.Br()

    ], style={'padding': '50px'})

])


@app.callback(
    Output('chip_select_recomendations', 'children'),
    Input('tags', 'value'))
def update_output(value):
    recomendations = []
    for i in df.index:
        if df[value][i]:
            recomendations.append(df['desc'][i])
    random_recs = random.sample(recomendations, min(len(recomendations), 5))
    recomendations = ''
    for i in random_recs:
        recomendations += i + '\n'
    return recomendations

@app.callback(
    Output('tags_rec', 'children'),
    Input('submit_tags', 'n_clicks'),
    State('new_query_input', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        mystem = Mystem()
        russian_stopwords = stopwords.words("russian")
        tokens = mystem.lemmatize(value.lower())
        value = " ".join(token for token in tokens if token not in russian_stopwords
                         and token != " " and token.strip() not in punctuation and token != 't')
        value = ' ' + value + ' '
        ans = ''
        for tag in tags:
            if tag+' ' in value:
                ans += tag + ', '
        if ans == '':
            return 'Подходящих тегов не найдено, попробуйте ввести другой запрос'
        return 'Попробуйте поискать по тегам: ' + ans

@app.callback(
    Output('found_tags', 'children'),
    Input('choose_desc_find_tags', 'value'))
def update_output(value):
    i = df[df.desc == value].index[0]
    recomendations = 'Попробуйте поискать по тегам: '
    for tag in tags:
        if df[tag][i]:
            recomendations += tag + ', '
    return recomendations

@app.callback(
    Output('change_tag_result', 'children'),
    [Input('delete_tag', 'n_clicks'),
     Input('insert_tag', 'n_clicks')],
    [State('choose_desc_change_tags', 'value'),
     State('choose_tag_change_tags', 'value')],
)
def update_output(n_clicks_del, n_clicks_ins, description, tag):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    i = df[df.desc == description].index[0]
    if 'delete_tag' in changed_id:
        df[tag][i] = 0
    elif 'insert_tag' in changed_id:
        df[tag][i] = 1
    return ''

@app.callback(
    Output('doc2vec_recomendations', 'children'),
    Input('desc_doc2vec_recomendations', 'value'))
def update_output(value):
    i = df[df.desc == value].index[0]
    ind = model.dv.most_similar([model.infer_vector(df.desc_clear[i].split())], topn=5)
    most_similar = []
    for j in ind:
        try:
            most_similar.append(desc[j[0]])
        except IndexError:
            continue
    most_similar = np.unique(most_similar)
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

@app.callback(
    Output('clusters_consist', 'children'),
    Input('clusters', 'value'))
def update_output(value):
    descriptions = df[df.labels_kmeans == value].desc
    ind = df[df.labels_kmeans == value].desc.index
    recomendations = ''
    for i in range(len(descriptions)):
        recomendations += str(ind[i]) + ' ' + descriptions.iloc[i] + '\n'
    return recomendations

@app.callback(
    Output('done', 'children'),
    Input('submit_changes', 'n_clicks'),
    [State('desc_num', 'value'),
     State('cluster_num', 'value')])
def update_output(n_clicks, desc_num, cluster_num):
    recomendations = ''
    # changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # if 'submit_changes' in changed_id:
    if n_clicks > 0:
        df['labels_kmeans'][int(desc_num)] = int(cluster_num)
        for desc in df[df.labels_kmeans == int(cluster_num)].desc:
            recomendations += desc + '\n'
    # return recomendations
    return 'Cluster is changed'


# START
if __name__ == '__main__':
    app.run_server(debug=True)