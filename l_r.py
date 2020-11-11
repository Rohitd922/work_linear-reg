import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
import base64
import datetime
import io
import dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
from numpy import random
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.SOLAR])
server = app.server

card_upload = dbc.Card([
    dbc.CardBody(
    [
        dcc.Upload(
            id = 'upload-data',
            children = html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style = {
                'color' : 'white',
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple = True
        ),
        html.Div(id = 'output-data-upload')

    ])
    ],
    color = "warning",  # https://bootswatch.com/default/ for more card colors
    inverse = False,  # change color of text (black or white)
    outline = False,  # True = remove the block colors from the background and header,
)

card_data = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Data Selection", className = "card-title",
                        style = {'color': 'white', 'text-align': 'center', 'marginBottom': 5,
                                 'border': '1px black solid', 'borderRadius': 2, 'marginTop': 5}
                        ),
                html.Hr(),
                html.H4(
                    "Create Training Data",
                    className = "card-text",
                    style = {'text-align': 'left'}
                ),
                html.Hr(),
                html.P(
                    "Select the underlying True distribution for the Data"
                ),
                dcc.Dropdown(
                    id = 'datatype-dropdown',
                    options = [
                        {'label': 'Sine', 'value': 'sin'},
                        {'label': 'Cosine', 'value': 'cos'},
                        {'label': 'Logarithm', 'value': 'log'},
                        {'label': 'Exponential', 'value': 'exp'}
                    ],
                    style = {'color': 'black'},
                    value = 'sin'
                ),
                html.Hr(),
                html.P(
                    "Noise"
                ),
                dcc.Slider(
                    id = 'noise-slider', min = 0, max = 1, step = 0.001, value = 0.01,
                    marks = {
                        0: {'label': '0', 'style': {'color': '#77b0b1'}},
                        1: {'label': '1', 'style': {'color': '#77b0b1'}}}
                ),
                html.Div(id = 'slider-output-noise'),
                html.Hr(),
                html.P(
                    "Number of training samples"
                ),
                dbc.Input(id = 'train-number-input', value = 10, type = "number", min = 0, max = 500, step = 1),
                html.Div(id = 'train-number-output'),
                html.Hr(),
                html.P(
                    "Number of test samples"
                ),
                dbc.Input(id = 'test-number-input', value = 10, type = "number", min = 0, max = 500, step = 1),
                html.Div(id = 'test-number-output')
            ]
        ),
    ],
    color = "primary",  # https://bootswatch.com/default/ for more card colors
    inverse = True,  # change color of text (black or white)
    outline = False,  # True = remove the block colors from the background and header,
)

card_train = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H3("Train your model", className = "card-title",
                        style = {'color': 'white', 'text-align': 'center', 'marginBottom': 5,
                                 'border': '1px black solid', 'borderRadius': 2, 'marginTop': 5}
                        ),
                html.Hr(),
                html.H4(
                    "Hyper-parameter Selection",
                    className = "card-text",
                    style = {'text-align': 'left'}
                ),
                html.Hr(),
                html.P(
                    "Select Model"
                ),
                dcc.Dropdown(
                    id = 'model-dropdown',
                    options = [
                        {'label': 'Linear Regression', 'value': 'linear'},
                        {'label': 'Lasso Regression', 'value': 'lasso'},
                        {'label': 'Ridge Regression', 'value': 'ridge'},
                    ],
                    style = {'color': 'black'},
                    value = 'linear',
                    searchable = False,
                    clearable = True,
                ),
                html.Hr(),
                html.P(
                    "Select Polynomial Degree"
                ),
                dcc.Slider(
                    id = 'slider-polynomial-degree',
                    min = 1,
                    max = 10,
                    step = 1,
                    value = 1,
                    marks = {i: i for i in range(1, 11)}
                ),
                html.Hr(),
                html.P(
                    "Select Regularization Co-efficient"
                ),
                dcc.Slider(
                    id = "slider-alpha",
                    min = -3,
                    max = 3,
                    value = 1,
                    marks = {i: '{}'.format(10 ** i) for i in range(-4, 4)},
                ),
            ],
        ),
    ],
    color = "warning",  # https://bootswatch.com/default/ for more card colors
    inverse = True,  # change color of text (black or white)
    outline = False,  # True = remove the block colors from the background and header,
)

card_display = dbc.Card(
    [
        dbc.CardBody([
            html.H3("Data Visualization", className = "card-title",
                    style = {'color': 'white', 'text-align': 'center', 'marginBottom': 5,
                             'border': '1px black solid', 'borderRadius': 2, 'marginTop': 5}
                    ),
            html.Div(dcc.Graph(id = 'graph-display'))
        ]),
    ],
    color = "info",  # https://bootswatch.com/default/ for more card colors
    inverse = True,  # change color of text (black or white)
    outline = False,  # True = remove the block colors from the background and header,
)

app.layout = html.Div([
    html.H2("Linear Regression", style = {'text-align': 'center', 'marginBottom': 25, 'border': '2px white solid',
                                          'borderRadius': 3, 'marginTop': 25}),
    html.Div([
        dbc.CardGroup(dbc.Col(card_upload, width=3)),
        dbc.CardGroup((dbc.Col(card_data, width = 3),
                 dbc.Col(card_train, width = 3),
                 dbc.Col(card_display, width = 6)
                 )
                )
    ])

])


@app.callback(
    dash.dependencies.Output('slider-output-noise', 'children'),
    [dash.dependencies.Input('noise-slider', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('train-number-output', 'children'),
    [dash.dependencies.Input('train-number-input', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('test-number-output', 'children'),
    [dash.dependencies.Input('test-number-input', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(Output('slider-alpha', 'disabled'),
              [Input('model-dropdown', 'value')])
def disable_slider_alpha(model):
    return model in ['linear']


def format_coefs(coefs):
    equation_list = [f"{coef}x^{i}" for i, coef in enumerate(coefs)]
    equation = "$" + " + ".join(equation_list) + "$"

    replace_map = {"x^0": "", "x^1": "x", '+ -': '- '}
    for old, new in replace_map.items():
        equation = equation.replace(old, new)

    return equation


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df=df.head()
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df=df.head()
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children




@app.callback(dash.dependencies.Output(component_id = 'graph-display', component_property = 'figure'),
              [dash.dependencies.Input('datatype-dropdown', 'value'),
               dash.dependencies.Input('noise-slider', 'value'),
               dash.dependencies.Input('train-number-input', 'value'),
               dash.dependencies.Input('test-number-input', 'value'),
               dash.dependencies.Input('model-dropdown', 'value'),
               dash.dependencies.Input('slider-polynomial-degree', 'value'),
               dash.dependencies.Input('slider-alpha', 'value')]
              )
def update_graph(dt, var, ntr, nte, mod, deg, alp):
    np.random.seed(100)
    x_train, y_train, x_test, y_test = 0, 0, 0, 0

    if dt in ['sin', 'cos', 'exp', 'log']:
        if dt == 'sin':
            x_data = random.uniform(0, 1 ,ntr)
            x_train = x_data.reshape(-1, 1)  # create a matrix from the list
            y_train = np.sin(2 * np.pi * x_data) + random.normal(0, var, x_data.shape)
            x_data_test = random.uniform(0,1,nte)
            x_test = x_data_test.reshape(-1, 1)
            y_test = np.sin(2 * np.pi * x_data_test) + random.normal(0, var, x_data_test.shape)
        if dt == 'cos':
            x_data = random.uniform(0,1,ntr)
            x_train = x_data.reshape(-1, 1)  # create a matrix from the list
            y_train = np.cos(2 * np.pi * x_data) + random.normal(0, var, x_data.shape)
            x_data_test = random.uniform(0,1,nte)
            x_test = x_data_test.reshape(-1, 1)
            y_test = np.cos(2 * np.pi * x_data_test) + random.normal(0, var, x_data_test.shape)
        if dt == 'log':
            x_data = random.uniform(0,1,ntr)
            x_train = x_data.reshape(-1, 1)  # create a matrix from the list
            y_train = np.log(x_data) + random.normal(0, var, x_data.shape)
            x_data_test = random.uniform(0,1,nte)
            x_test = x_data_test.reshape(-1, 1)
            y_test = np.log(x_data_test) + random.normal(0, var, x_data_test.shape)
        if dt == 'exp':
            x_data = random.uniform(0,1,ntr)
            x_train = x_data.reshape(-1, 1)  # create a matrix from the list
            y_train = np.exp(x_data) + random.normal(0, var, x_data.shape)
            x_data_test = random.uniform(0,1,nte)
            x_test = x_data_test.reshape(-1, 1)
            y_test = np.exp(x_data_test) + random.normal(0, var, x_data_test.shape)

    X_train = PolynomialFeatures(degree = deg, include_bias = True).fit_transform(x_train)
    X_test = PolynomialFeatures(degree = deg, include_bias = True).fit_transform(x_test)

    if mod == 'lasso':
        model = Lasso(alpha = alp, normalize = True)
    elif mod == 'ridge':
        model = Ridge(alpha = alp, normalize = True)
    else:
        model = LinearRegression(normalize = True)

    model.fit(X_train, y_train)

    Xmax = X_train
    if X_train.max() < X_test.max():
        Xmax = X_test

    Xmin = X_train
    if X_train.min() > X_test.min():
        Xmin = X_test

    x_range = np.linspace(Xmin.min(), Xmax.max(), 1000, endpoint=True).reshape(-1, 1)
    X_range_poly = PolynomialFeatures(degree = deg, include_bias = True).fit_transform(x_range)

    y_range = model.predict(X_range_poly)

    test_score = model.score(X_test, y_test)
    test_error = mean_squared_error(y_test, model.predict(X_test))
    train_score = model.score(X_train, y_train)
    train_error = mean_squared_error(y_train, model.predict(X_train))

    eq = format_coefs(model.coef_.round(2))

    fig = go.Figure([
        go.Scatter(x = x_train.squeeze(), y = y_train, name = 'train', mode = 'markers'),
        go.Scatter(x = x_test.squeeze(), y = y_test, name = 'test', mode = 'markers'),
        go.Scatter(x = x_range.squeeze(), y = y_range, name = eq),
    ])
    fig.update_layout(
        # title= str(dataset) + "  Linear Regression Plot",
        xaxis_title = 'X',
        yaxis_title = "Dependent Variable",
        legend_title = "Legend",
        title = f"test_MSE: {test_error:.3f}, train_MSE: {train_error:.3f}",
        legend=dict(orientation='h', ),
        margin = dict(l = 25, r = 15, t = 50, b = 50),
        hovermode='closest',
        font = dict(
            family = "Courier New, monospace",
            size = 12,
            color = "RebeccaPurple"

        ))

    return fig


if __name__ == '__main__':
    app.run_server(host = '127.0.0.1', port = "5010", debug = False)
