import os
from som import SOM
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# ファイル名をアプリ名として起動。その際に外部CSSを指定できる。
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server



iris = load_iris()
X = iris.data

n_dim_latent = 2
n_epoch = 20

random_state = check_random_state(8)
init = random_state.rand(X.shape[0], n_dim_latent) * 2.0 - 1.0

som = SOM(X=X, latent_dim=n_dim_latent, resolution=30,
          init=init, sigma_max=1.0, sigma_min=0.1, tau=20)

som.fit(nb_epoch=n_epoch)

color_sequence = np.array(px.colors.qualitative.Set2)
width_fig = None
height_fig = None


# この`layout`にアプリの外観部分を指定していく。
# `dash_html_components`がHTMLタグを提供し、React.jsライブラリを使って実際の要素が生成される。
# HTMLの開発と同じ感覚で外観を決めることが可能

# fig = px.scatter(x=som.Z[:, 0], y=som.Z[:, 1])
fig_ls = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text='Latent space'),
        xaxis={'range': [som.Z[:, 0].min()-0.05, som.Z[:, 0].max()+0.05]
               },
        yaxis={
            'range': [som.Z[:, 1].min()-0.05, som.Z[:, 1].max()+0.05],
            'scaleanchor': 'x',
            'scaleratio': 1.0
        },
        width=width_fig,
        height=width_fig,
        showlegend=False
    )
)
# draw contour of mapping
fig_ls.add_trace(go.Contour(x=som.Zeta[:, 0], y=som.Zeta[:, 1],
                            z=som.Y[:, 0], colorscale='GnBu_r',
                            line_smoothing=0.85,
                            contours_coloring='heatmap', name='cp'
                            )
                 )
# draw invisible grids to click
fig_ls.add_trace(
    go.Scatter(x=som.Zeta[:, 0], y=som.Zeta[:, 1], mode='markers',
               visible=True,
               marker=dict(symbol='square', size=10, opacity=0.0,color='black'),
               name='latent space')
)
index_grids = 1

# draw latent variables
fig_ls.add_trace(
    go.Scatter(
        x=som.Z[:, 0], y=som.Z[:, 1],
        mode='markers', name='latent variable',
        marker=dict(
            size=10,
            color=color_sequence[iris.target],
            line=dict(
                width=2,
                color="dimgrey"
            )
        ),
        text=iris.target_names[iris.target]
    )
)
index_z = 2
# draw click point initialized by visible=False
fig_ls.add_trace(
    go.Scatter(
        x=np.array(0.0), y=np.array(0.0),
        visible=False,
        marker=dict(
            size=10,
            symbol='x',
            color=color_sequence[len(np.unique(iris.target))],
            line=dict(
                width=1,
                color="white"
            )
        ),
        name='clicked_point'
    )
)
fig_bar = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text='Feature bars'),
        yaxis={'range': [0, X.max()]},
        width=width_fig,
        height=height_fig
    )
)
fig_bar.add_trace(
    go.Bar(x=iris.feature_names, y=np.zeros(som.X.shape[1]),
           marker=dict(color=color_sequence[len(np.unique(iris.target))])
           )
)

bar_data_store = dcc.Store(
    id='bar-figure-store',
    data=fig_bar
)
config = {'displayModeBar': False}
app.layout = html.Div(children=[
    # `dash_html_components`が提供するクラスは`childlen`属性を有している。
    # `childlen`属性を慣例的に最初の属性にしている。
    html.H1(children='Visualization iris dataset by SOM'),
    # html.Div(children='by component plance of SOM.'),
    # `dash_core_components`が`plotly`に従う機能を提供する。
    # HTMLではSVG要素として表現される。
    bar_data_store,
    html.Div(
        [
            dcc.Graph(
                id='left-graph',
                figure=fig_ls,
                config=config
            ),
            html.P('Feature as contour'),
            dcc.Dropdown(
                id='feature_dropdown',
                options=[{"value": i, "label": x}
                         for i, x in enumerate(iris.feature_names)],
                value=0
            )
        ],
        style={'display': 'inline-block', 'width': '49%'}
    ),
    html.Div(
        [dcc.Graph(
            id='right-graph',
            # figure=fig_bar,
            # config=config
        ),
        html.Details([
            html.Summary('Contents of figure storage'),
            dcc.Markdown(
                id='clientside-figure-json'
            )
        ])
        ],
        style={'display': 'inline-block', 'width': '49%'}
    )
])
app.clientside_callback(
    """
    function(data){
        return data
    }
    """,
    Output('right-graph', 'figure'),
    Input('bar-figure-store', 'data')
)
@app.callback(
    Output('bar-figure-store', 'data'),
    Input('left-graph', component_property='clickData')
)
def update_bar(clickData):
    print(clickData)
    if clickData is not None:
        index = clickData['points'][0]['pointIndex']
        # print('index={}'.format(index))
        if clickData['points'][0]['curveNumber'] == index_z:
            # print('clicked latent variable')
            # if latent variable is clicked
            # data[0]['y'] = som.X[index]
            fig_bar.update_traces(y=som.X[index])
            print(fig_bar)
            # fig_ls.update_traces(visible=False, selector=dict(name='clicked_point'))
            return fig_bar
        elif clickData['points'][0]['curveNumber'] == index_grids:
            # print('clicked map')
            # if contour is clicked
            fig_bar.update_traces(y=som.Y[index])
            # data[0]['y'] = som.X[index]
            return fig_bar
        else:
            return fig_bar
    else:
        return fig_bar
# @app.callback(
#     Output('clientside-figure-json', 'children'),
#     Input('bar-figure-store', 'data')
# )
# def generated_figure_json(data):
#     return '```\n'+json.dumps(data, indent=2)+'\n```'

# Define callback function when data is clicked by normal callback
# @app.callback(
#     Output(component_id='right-graph', component_property='figure'),
#     Input(component_id='left-graph', component_property='clickData')
# )
# def update_bar(clickData):
#     # print(clickData)
#     if clickData is not None:
#         index = clickData['points'][0]['pointIndex']
#         # print('index={}'.format(index))
#         if clickData['points'][0]['curveNumber'] == index_z:
#             # print('clicked latent variable')
#             # if latent variable is clicked
#             fig_bar.update_traces(y=som.X[index])
#             fig_ls.update_traces(visible=False, selector=dict(name='clicked_point'))
#         elif clickData['points'][0]['curveNumber'] == index_grids:
#             # print('clicked map')
#             # if contour is clicked
#             fig_bar.update_traces(y=som.Y[index])
#         # elif clickData['points'][0]['curveNumber'] == 0:
#         #     print('clicked heatmap')
#         return fig_bar
#     else:
#         return dash.no_update

@app.callback(
    Output(component_id='left-graph', component_property='figure'),
    [Input(component_id='feature_dropdown', component_property='value'),
     Input(component_id='left-graph', component_property='clickData')]
)
def update_ls(index_selected_feature, clickData):
    # print(clickData)
    print(index_selected_feature, clickData)
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        return dash.no_update
    else:
        clicked_id_text = ctx.triggered[0]['prop_id'].split('.')[0]
        # print(clicked_id_text)
        if clicked_id_text == 'feature_dropdown':
            # print(index_selected_feature)
            fig_ls.update_traces(z=som.Y[:, index_selected_feature],
                                 selector=dict(type='contour', name='cp'))
            return fig_ls
        elif clicked_id_text == 'left-graph':
            if clickData['points'][0]['curveNumber'] == index_grids:
                # if contour is clicked
                # print('clicked map')
                fig_ls.update_traces(
                    x=np.array(clickData['points'][0]['x']),
                    y=np.array(clickData['points'][0]['y']),
                    visible=True,
                    marker=dict(
                        symbol='x'
                    ),
                    selector=dict(name='clicked_point', type='scatter')
                )
            elif clickData['points'][0]['curveNumber'] == index_z:
                # print('clicked latent variable')
                fig_ls.update_traces(
                    x=np.array(clickData['points'][0]['x']),
                    y=np.array(clickData['points'][0]['y']),
                    visible=True,
                    marker=dict(
                        symbol='circle'
                    ),
                    selector=dict(name='clicked_point', type='scatter')
                )
                # if latent variable is clicked
                # fig_ls.update_traces(visible=False, selector=dict(name='clicked_point'))

            fig_ls.update_traces(z=som.Y[:, index_selected_feature],
                                 selector=dict(type='contour', name='cp'))
            return fig_ls
        else:
            return dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)


