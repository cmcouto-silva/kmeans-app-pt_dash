import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import kmeans
from sklearn.datasets import make_blobs


authorship = dcc.Markdown("""
**Autor:** Cainã Max Couto da Silva  
**LinkedIn:** [cmcouto-silva](https://www.linkedin.com/in/cmcouto-silva/)

---
""")

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# collapse_button = html.Div(
#     [
#         dbc.Button(
#             "Open collapse",
#             id="collapse-button_toggle",
#             className="mb-3",
#             color="primary",
#             n_clicks=0,
#         ),
#         dbc.Collapse(
#             dbc.Card(dbc.CardBody("This content is hidden in the collapse")),
#             id="collapse_button",
#             is_open=False,
#         ),
#     ]
# )

explanation = dcc.Markdown("""
K-means é um modelo de machine learning não supervisionado que visa identificar grupos (*i.e.* clusters) no conjunto de dados. Assim como todo modelo não supervisionado, seu objetivo é **identificar padrões nos dados** e interpretá-los, **não sendo adequado para predição**.

As variáveis utilizadas neste modelo são, **necessariamente**, numéricas. Como resultado, identificamos cada amostra à um grupo. A quantidade de grupos (K) identificados é determinada _a priori_, ou seja, antes de rodar o modelo. Contudo, existem métodos, como o método do cotovelo e da silhueta (Elbow e Silhouette, respectivamente), que nos auxiliam a identificar a melhor quantidade de grupos para nossos dados.

As variáveis categóricas podem ser utilizas após a descrição e interpretação das amostras presentes nestes grupos. **Não é recomendado o uso de ponderação arbitrária** para transformar variáveis categóricas em variáveis numéricas. Ao transformar variáveis categóricas em variáveis numéricas sem respaldo da literatura, assume-se que a distância entre as categorias são as mesmas, que por sua vez pode introduzir um viés na análise.

O modelo K-means é essencialmente baseado no cálculo da distância - normalmente a [distância euclidiana](https://pt.wikipedia.org/wiki/Dist%C3%A2ncia_euclidiana) - entre os pontos da amostra e os centroides, que são os pontos inicialmente aleatórios e representam o centro de cada grupo (cluster). Assim, a quantidade de centroides é igual ao número de grupos (estes representados pela letra "k").

Cabe ressaltar que, uma vez que K-means utiliza a distância como métrica do modelo, é necessário normalizar os dados caso as variáveis estejam em escalas diferentes.

**Em resumo, a técnica funciona assim:**

1. Definir um número de clusters (k)
2. Inicializar os k centroides
3. Categorizar cada ponto ao seu centroide mais próximo
4. Mover os centroides para o centro (média) dos pontos em sua categoria
5. Repetir as etapas 3 e 4 até as posições dos centroides não modificaram ou atingir um número máximo de iterações.  


A inicialização dos centroides pode ser totalmente aleatória ou otimizada utilizando um algoritmo conhecido como *K-means++*.

Na inicialização aleatória, selecionamos k pontos pré-existentes como centroides, ou atribuímos k centroides dentro da dimensão dos pontos dos nossos dados. Na técnica K-means++ o objetivo é inicializar os centroides o mais distante possível um do outro, que por sua vez pode eliminar viés de inicialização de centroides (vide tópico "Armadilha do K-means") e tende a diminuir a quantidade de iterações necessárias para convergência da posição final dos centroides.

Adicionalmente, algumas aplicações também rodam mais de uma vez o algoritmo com diferentes pontos de inicialização dos centroides, fornecendo como output aquele com a menor soma das variâncias internas de cada grupo.

_**Observações:**_

---

Os dados desta aplicação são simulados com a função [`datasets.make_blobs()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) da biblioteca [`scikit-learn`](https://scikit-learn.org/stable/index.html). Aqui, limitei tanto a quantidade de observações (100) quanto de variáveis (2), para a aplicação não ficar pesada e ser possível visualizar com gráficos bidimensionais, respectivamente. 

Não foi utilizado mais de uma corrida por k, de modo que isso pode influenciar no método do cotovelo, visto que para determinados Ks a inicialização do centroide não foi a das melhores. 

Sugestões ou críticas? Contate-me via [LinkedIn](https://www.linkedin.com/in/cmcouto-silva/).

_**Disponibilização dos códigos:**_

---

Os scripts com a implementação passo a passo do K-means e produção deste aplicativo estão disponíveis no GitHub:

- Repositório do aplicativo
- Script K-means passo a passo (sem scikit-learn)

""")

# Explanation - K-means' trap
kmeans_trap1 = dcc.Markdown("""
---

## **Armadilha do K-means**
""")

kmeans_trap2= dcc.Markdown("""
A inicialização aleatória dos centroides podem criar grupos que não representam grupos reais.
No exemplo abaixo, percebe-se claramente a existência de quatro grupos (à esquerda), 
equanto na animação à direita é possível observar que a iniciando os centroides próximos um dos outros,
neste caso, levou a um agrupamento não fidedigno dos dados reais. Por este motivo normalmente se utiliza `kmeans++` e/ou
aplica algumas vezes o modelo, ficando com aquele com menos distorção. 

Por fim, é importante salientar que há outras estruturas de agrupamentos mais complexas
onde o modelo K-means não possui bom desempenho. Nestes casos, usa-se outros modelos de agrupamento.
Como exemplo, a primeira figura da [página de modelos de clusterização](https://scikit-learn.org/stable/modules/clustering.html) do scikit-learn 
ilustra bem como cada modelo se sai na identificação de grupos em dados com diferentes formatos (distribuições).

""")

# Specific biased data
data_seed,labels_seed = make_blobs(centers=4, random_state=3)
model_seed = kmeans.Kmeans(data_seed, 4, seed=2)
model_seed.fit()

# Raw figure - seed
raw_fig_seed = go.Figure(
    data=go.Scatter(x=data_seed[:,0], y=data_seed[:,1], mode='markers', marker=dict(color=labels_seed)),
    layout=dict(title_text="<b>Grupos de referência</b>",
        template="simple_white", title_font=dict(size=32), height=510, title_x=0.15, title_font_size=21))
raw_fig_seed = raw_fig_seed.update_layout(autosize=True)

# Animation figure - seed
fig_seed = kmeans.plot(model_seed)
fig_seed = fig_seed.update_layout(autosize=False,
    title_text="<b>Visualizando viés de inicialização dos centroides</b>", title_font=dict(size=24))

# Show bias plots
bias_plots = dbc.Row([
    dbc.Col(width=5, children=dcc.Graph(figure=raw_fig_seed)),
    dbc.Col(width=5, children=dcc.Graph(figure=fig_seed))
    ])

# Button for main explanation
collapse_accordion = dbc.Card([
    dbc.CardHeader(
        html.H2(
            dbc.Button(
                "Ler explicação do método",
                color="black",
                id='collapse-accordion_toggle',
                n_clicks=0
            )
        )
    ),
    dbc.Collapse(
        dbc.CardBody(explanation),
        id="collapse_accordion",
        is_open=False
    )
])

sidebar_intro = dcc.Markdown("""
Para fins didáticos, os dados deste aplicativo são simulados utilizando a função 
[`datasets.make_blobs`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
da biblioteca [scikit-learn](https://scikit-learn.org/stable/index.html), conforme os parâmetros abaixo.
""")

# App Layout
app.layout = dbc.Container(
    fluid=True,
    style={"height": "100vh"},
    children=[
        html.H1('Visualizando o algorimo K-means etapa por etapa com Python', style={"padding-top":"10px"}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    width=3,
                    children=dbc.Card(
                        [
                            dbc.CardHeader("Parâmetros", style={"font-weight":"bold", "text-align":"left"}),
                            dbc.CardBody([
                                sidebar_intro,
                                html.Br(),
                                html.P('Número de grupos (clusters) simulados:'),
                                dcc.Slider(
                                    id='k',
                                    marks={i: str(i) for i in range(2,11)},
                                    min=2,
                                    max=10,
                                    step=1,
                                    value=2
                                ),
                                html.Br(),
                                html.Div('Desvio padrão dos dados simulados:'),
                                dcc.Slider(
                                    id='std',
                                    marks={i: f'{i:.1f}' for i in [0.1]+list(np.arange(0, 5.1, 0.5))},
                                    min=0.10, max=5.0, step=0.1,
                                    tooltip=dict(always_visible=False),
                                    value=1.0
                                ),
                                html.Br(),
                                html.Br(),
                                html.P('Modo de inicialização dos centroides:'),
                                dcc.Dropdown(
                                    id = 'mode',
                                    options = [{'label': mode, 'value': mode} for mode in ['random', 'kmeans++']],
                                    value='random'
                                ),
                            ])
                        ]
                    ),
                ),
                dbc.Col(
                    width=9,
                    children=dbc.Card(
                        [
                            dbc.CardHeader("Explicação e visualização do modelo K-means"),
                            dbc.CardBody([
                                authorship,
                                # collapse_button,
                                collapse_accordion,
                                dbc.Row([
                                    dbc.Col(width=6, children=dcc.Graph(id='raw', figure={})),
                                    dbc.Col(width=6, children=dcc.Graph(id='elbow', figure={}))
                                    ]),
                                # dbc.Row([
                                    # dbc.Col(width=2),
                                    # dbc.Col(
                                        # width=10,
                                        dcc.Graph(
                                            id='animated_plot',
                                            figure={},
                                            className='container', style={'maxWidth': '800px', "textAlign":"center"}
                                            ),
                                #     ),
                                # ], style={"textAlign":"center"}),
                                html.Br(),
                                kmeans_trap1,
                                html.Br(),
                                kmeans_trap2,
                                bias_plots
                            ])
                        ],
                        style={"height": "222vh"},
                    ),
                ),
            ],
        ),
    ],
)

# Explanation expander (accordion)
@app.callback(
    Output("collapse_accordion", "is_open"),
    [Input("collapse-accordion_toggle", "n_clicks")],
    [State("collapse_accordion", "is_open")]
)
def toggle_accordion(n, is_open):
    if n:
        return not is_open
    return is_open

# # Explanation expander (button)
# @app.callback(
#     Output("collapse_button", "is_open"),
#     [Input("collapse-button_toggle", "n_clicks")],
#     [State("collapse_button", "is_open")],
# )
# def toggle_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open

# Plots
@app.callback(
    [
        Output('raw', 'figure'),
        Output('elbow', 'figure'),
        Output('animated_plot', 'figure'),
        ],
    [
        Input('k', 'value'),
        Input('std', 'value'),
        Input('mode', 'value')
        ],
    prevent_initial_call=False # True : Não ativa o callback quando a pagina é atualizada
)
def make_plot(k, std, mode):
    data = make_blobs(centers=k, cluster_std=std)
    df = pd.DataFrame(data[0], columns=['x','y']).assign(label=data[1])
    model, wss = kmeans.calculate_WSS(data[0], k, 10, mode=mode)
    animated_plot = kmeans.plot(model)
    animated_plot = animated_plot.update_layout(
        title_text="""<b>Visualizando as etapas do K-means</b>""",
        title_font=dict(size=24), title_x=0.5,
        width=800, height=600
        )


    # Raw Figure
    raw_fig = go.Figure(
    data=animated_plot.data[0],
    layout=dict(
        template='seaborn', title='<b>Pontos sem agrupamento</b>',
        xaxis=dict({'title':'x'}), yaxis=dict({'title':'y'})
        )
    )

    # Elbow Figure
    elbow_fig = go.Figure(
	data=go.Scatter(x=list(range(1,11)), y=wss),
        layout=dict(
            template='seaborn', title='<b>Método do cotovelo</b>',
            xaxis=dict({'title':'k'}), yaxis=dict({'title':'wss'})
            )
	)

    
    return raw_fig, elbow_fig, animated_plot








if __name__ == "__main__":
    app.run_server(debug=True)

