import dash
import dash_bootstrap_components as dbc
from dash import dcc, ctx
from dash import html
from dash.dependencies import Input, Output, State

import random

# Initialize Dash app with external stylesheets
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY], title="MS2LDA")
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="MS2LDA")
server = app.server  # To deploy on a server like Heroku

# Layout of the Dash app using Bootstrap components
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.Img(src="assets/MS2LDA_LOGO_white.jpg", alt="MS2LDA Logo", height="250px"),
                dcc.Markdown("""
                Developed by [Jonas Dietrich](https://github.com/j-a-dietrich) and [Rosina Torres Ortega] (https://github.com/rtlortega).
                """)
            ], width=True),
        ], align="end"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dcc.Markdown("##### Settings"),
                # Add more buttons and collapses here for other functionalities
                dbc.Button(
                    "LDA settings",
                    id="lda_button"
                ),
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Slider(
                                id='alpha_slider_input_1',
                                min=2,
                                max=400,
                                step=1,
                                value=40,
                                #marks={i: str(i) for i in range(11)},
                                #tooltip={'placement': 'bottom', 'always_visible': True}
                            ),
                            html.Label(id='alpha_slider_output_1', children="Angle of Attack: 5"),  # Display slider value dynamically
                            ],
                        ),
                        dbc.CardBody([
                            dcc.Slider(
                                id='alpha_slider_input',
                                min=2,
                                max=400,
                                step=1,
                                value=40,
                                #marks={i: str(i) for i in range(11)},
                                #tooltip={'placement': 'bottom', 'always_visible': True}
                            ),
                            html.Label(id='alpha_slider_output', children="Angle of Attack: 5"),  # Display slider value dynamically
                            ],
                        ),
                        dbc.CardBody([
                            dcc.Slider(
                                id='alpha_slider_input_2',
                                min=2,
                                max=400,
                                step=1,
                                value=40,
                                #marks={i: str(i) for i in range(11)},
                                #tooltip={'placement': 'bottom', 'always_visible': True}
                            ),
                            html.Label(id='alpha_slider_output_2', children="Angle of Attack: 5"),  # Display slider value dynamically
                            ],
                        ),
                        dbc.CardBody([
                            dcc.Slider(
                                id='alpha_slider_input_3',
                                min=2,
                                max=400,
                                step=1,
                                value=40,
                                #marks={i: str(i) for i in range(11)},
                                #tooltip={'placement': 'bottom', 'always_visible': True}
                            ),
                            html.Label(id='alpha_slider_output_3', children="Angle of Attack: 5"),  # Display slider value dynamically
                            ],
                        ),
                        dbc.CardBody([
                            dcc.Slider(
                                id='alpha_slider_input_4',
                                min=2,
                                max=400,
                                step=1,
                                value=40,
                                #marks={i: str(i) for i in range(11)},
                                #tooltip={'placement': 'bottom', 'always_visible': True}
                            ),
                            html.Label(id='alpha_slider_output_4', children="Angle of Attack: 5"),  # Display slider value dynamically
                            ],
                        ),
                        ]
                    ),
                    id="lda_collapse",
                    is_open=False
                ),
                html.Hr(),
                # Add more buttons and collapses here for other functionalities
                # Add more buttons and collapses here for other functionalities
                dbc.Button(
                    "Motif settings",
                    id="operating_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            # Placeholder for operating slider components
                            "Operating Slider Components"
                        )
                    ),
                    id="operating_collapse",
                    is_open=False
                ),
                html.Hr(),
                # Add more buttons and collapses here for other functionalities
                # Add more buttons and collapses here for other functionalities
                dbc.Button(
                    "Display settings",
                    id="display_button"
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(
                            # Placeholder for operating slider components
                            "Operating Slider Components"
                        )
                    ),
                    id="display_collapse",
                    is_open=False
                ),
        html.Hr(),
        dcc.Markdown("##### Running"),
        # Upload Button
                dcc.Upload(
                    id='upload_data',
                    children=html.Button('Upload Data', className="btn btn-success"),
                    multiple=False
                ),
                html.Hr(),
        dcc.Markdown("##### Analysis"),
        html.Hr(),
        dcc.Markdown("##### Screening"),
        dbc.Button(
                "Start",
                id="btn-nclicks-1",
                color="success",
                outline=True
                ),
        html.Hr(),
                # Add more buttons and collapses here for other functionalities
            ], width=3),
            dbc.Col([
                dcc.Graph(id='display', style={'height': '90vh'}),
            ], width=9, align="start")
        ]),
        html.Hr(),
        
        dcc.Markdown("""
        This project is supported by the Computational Metabolomics group at Wageningen University and the Pesticide Group at Wageningen Food Safety Research Center.
        """),
    ],
    fluid=True
)

#-------------------------------#
# button
@app.callback(
    Output('btn-nclicks-1', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
)
def update_output(btn1):
    msg = "no button clicked"
    if "btn-nclicks-1" == ctx.triggered_id:
        msg = "Button clicked"
    print(msg)
    return html.Div(msg)



# Callbacks to toggle collapses
@app.callback(
    Output("display_collapse", "is_open"),
    [Input("display_button", "n_clicks")],
    [State("display_collapse", "is_open")]
)
def toggle_display_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Callbacks to toggle collapses
@app.callback(
    Output("lda_collapse", "is_open"),
    [Input("lda_button", "n_clicks")],
    [State("lda_collapse", "is_open")]
)
def toggle_lda_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Callbacks to toggle collapses
@app.callback(
    Output("operating_collapse", "is_open"),
    [Input("operating_button", "n_clicks")],
    [State("operating_collapse", "is_open")]
)
def toggle_operating_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback to display slider values
@app.callback(
    Output("alpha_slider_output", "children"),
    [Input("alpha_slider_input", "drag_value")]
)
def display_alpha_slider(drag_value):
    return f"N of Motifs: {drag_value}"


# Callback to display slider values
@app.callback(
    Output("alpha_slider_output_1", "children"),
    [Input("alpha_slider_input_1", "drag_value")]
)
def display_alpha_slider_1(drag_value):
    return f"N of Iterations: {drag_value}"

# Callback to display slider values
@app.callback(
    Output("alpha_slider_output_2", "children"),
    [Input("alpha_slider_input_2", "drag_value")]
)
def display_alpha_slider_1(drag_value):
    return f"Min Intensity MS1: {drag_value}"


# Callback to display slider values
@app.callback(
    Output("alpha_slider_output_3", "children"),
    [Input("alpha_slider_input_3", "drag_value")]
)
def display_alpha_slider_1(drag_value):
    return f"Min Intensity MS2: {drag_value}"


# Callback to display slider values
@app.callback(
    Output("alpha_slider_output_4", "children"),
    [Input("alpha_slider_input_4", "drag_value")]
)
def display_alpha_slider_1(drag_value):
    return f"Mass Tolerance: {drag_value}"
#--------------------------------------------------------#
# Callback to handle file upload
@app.callback(
    Output('display', 'figure'),  # Update the figure or content based on uploaded data
    [Input('upload_data', 'contents')],
    [State('upload_data', 'filename')]
)
def update_uploaded_data(contents, filename):
    if contents is not None:
        # Here you can process the uploaded data (e.g., read file contents, parse data, etc.)
        # Example: Read and process the uploaded file (you can replace this with your data processing logic)
        ###content_type, content_string = contents.split(',')
        ###decoded = base64.b64decode(content_string)
        # Process the data as needed...
        # Example: Plot the processed data and return the updated figure
        y = [random.randint(1, 20) for __ in range(3)]
        figure = {
            'data': [{'x': [1, 2, 3], 'y': y, 'type': 'bar', 'name': 'SF'}],
            'layout': {'title': f'Uploaded File: {filename}'}
        }
        return figure
    else:
        return {}





# Main function to run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
    
    #import subprocess
    #url = "http://127.0.0.1:8050/"
    #subprocess.Popen(['open', url]) doesn't work

