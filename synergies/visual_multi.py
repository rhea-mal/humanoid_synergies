import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx, MATCH, ALL
import pandas as pd
import numpy as np
import base64
import io
import uuid
import redis
import threading
import time

# --- Redis Setup ---
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

ROBOT_QI_KEY = "robot_qi"
ROBOT_QF_KEY = "robot_qf"
ROBOT_DQ_KEY = "robot_dq"

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Motion Synergy Generator"

# Global publishing threads
publishing_threads = {}

# --- Layout ---
app.layout = html.Div([
    html.H1("SynSculptor", style={'textAlign': 'center', 'marginBottom': '30px'}),

    # --- top controls row as a flex container ---
    html.Div(
        [
            # group the upload + add buttons
            html.Div(
                [
                    dcc.Upload(
                        id="upload-multi",
                        children=html.Button(
                            "+ Upload Synergy Library",
                            className="btn btn-outline-secondary"
                        ),
                        accept=".csv",
                        multiple=True,
                        style={"display": "inline-block", "cursor": "pointer"}
                    ),
                    html.Button(
                        "+ Add Synergy",
                        id="add-file-btn",
                        n_clicks=0,
                        className="btn btn-outline-secondary"
                    ),
                ],
                style={
                    'display': 'flex',
                    'gap': '10px',
                }
            ),

            # spacer
            html.Div(style={'flex': '1'}),

            # generate button and transition choice
            html.Div(
                [
                    html.Button(
                        "Generate",
                        id="generate-btn",
                        n_clicks=0,
                        className="btn btn-primary"
                    ),
                    html.Span("with", style={'margin': '0 10px', 'fontSize': '1.1em'}),
                    dcc.Dropdown(
                        id="transition-dropdown",
                        options=[
                            {"label": "Jump", "value": "jump"},
                            {"label": "Linear (LERP)", "value": "lerp"},
                            {"label": "Cosine", "value": "cosine"},
                            {"label": "B2B", "value": "nojump"}
                        ],
                        value="lerp",
                        clearable=False,
                        style={'width': '180px'}
                    )
                ],
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '10px'
                }
            )
        ],
        style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '0 40px',
            'marginBottom': '40px'
        }
    ),

    html.Div(id="file-slider-container", children=[]),
    dcc.Store(id="file-data-store", data={})
], style={'padding': '40px'})

# app.layout = html.Div([
#     html.H1("SynSculptor", style={'textAlign': 'center', 'marginBottom': '30px'}),

#     html.Div([
#         dcc.Upload(
#             id="upload-multi",
#             children=html.Button(
#                 "+ Upload Synergy Library", 
#                 className="btn btn-outline-secondary",
#                 style={"marginRight": "20px"}
#             ),
#             accept=".csv",
#             multiple=True,
#             style={
#                 # make the upload container shrink‐wrap around the button
#                 "display": "inline-block",
#                 "cursor": "pointer"
#             }
#         ),
#         html.Button("+ Add Synergy", id="add-file-btn", n_clicks=0, className="btn btn-outline-secondary", style={"marginRight": "20px"}),
#         html.Button("Generate", id="generate-btn", n_clicks=0, className="btn btn-primary", style={"marginRight": "20px"}),
#         dcc.Dropdown(
#             id="transition-dropdown",
#             options=[
#                 {"label": "Jump", "value": "jump"},
#                 {"label": "Linear (LERP)", "value": "lerp"},
#                 {"label": "Cosine", "value": "cosine"},
#                 {"label": "B2B", "value": "nojump"}
#             ],
#             value="lerp",
#             clearable=False,
#             style={"width": "180px", "display": "inline-block", "verticalAlign": "middle", "marginRight": "20px"}
#         ),

#     ], style={'textAlign': 'center', 'marginBottom': '40px'}),

#     html.Div(id="file-slider-container", children=[]),

#     dcc.Store(id="file-data-store", data={})
# ], style={'padding': '40px'})

def array_to_redis_string(array):
    return '[' + ','.join(f"{x:.6f}" for x in array) + ']'

@app.callback(
    Output({"type": "upload", "index": MATCH}, "children"),
    Input({"type": "upload", "index": MATCH}, "filename"),
    prevent_initial_call=True
)
def update_upload_text(filename):
    if filename is not None:
        # Only update the upload box text, NOT the Move title
        return html.Div([f"Uploaded: {filename}"])
    return dash.no_update


# --- Upload Block Creation ---
# def add_upload_block(n_clicks, children, content=None):
#     if n_clicks == 0:
#         return children

#     uid = str(uuid.uuid4())
#     index = len(children)

#     new_block = html.Div([
#         html.H5(f"Move {index+1}", style={'marginBottom': '10px', 'fontWeight': 'bold'}),

#         html.Div([
#             html.Div(id={"type": "arrow", "index": uid}, children="▶️", style={
#                 "fontSize": "24px", "cursor": "pointer", "marginRight": "10px", "marginTop": "8px"
#             }),
#             dcc.Upload( ## if content is not none, auto upload what we pass through
#                 id={"type": "upload", "index": uid},
#                 children=html.Div(["Drag or click to upload a synergy .csv"]),
#                 accept=".csv",
#                 multiple=False,
#                 style={
#                     "width": "100%",
#                     "padding": "10px",
#                     "border": "1px dashed #ccc",
#                     "borderRadius": "5px",
#                     "textAlign": "center"
#                 }
#             ),
#         ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "10px"}),


#         dbc.Collapse(
#             id={"type": "collapse", "index": uid},
#             is_open=False,
#             children=[
#                 html.Label("PC 1", style={"fontWeight": "bold", "fontSize": "20px", "color": "black"}),
#                 dcc.Slider(
#                     id={"type": "pc0-slider", "index": uid},
#                     min=-1, max=1, step=0.01, value=0.5,
#                     marks={-1: '-1', 0: '0', 1: '1'},
#                     tooltip={"placement": "bottom", "always_visible": True}
#                 ),
#                 html.Label("PC 2", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "20px", "color": "black"}),
#                 dcc.Slider(
#                     id={"type": "pc1-slider", "index": uid},
#                     min=-1, max=1, step=0.01, value=0,
#                     marks={-1: '-1', 0: '0', 1: '1'},
#                     tooltip={"placement": "bottom", "always_visible": True}
#                 ),
#                 html.Label("PC 3", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "20px", "color": "black"}),
#                 dcc.Slider(
#                     id={"type": "pc2-slider", "index": uid},
#                     min=-1, max=1, step=0.01, value=0,
#                     marks={-1: '-1', 0: '0', 1: '1'},
#                     tooltip={"placement": "bottom", "always_visible": True}
#                 ),
#                 html.Div([
#                     html.Div([
#                         html.Label("Step Size", style={"fontWeight": "bold", "marginBottom": "5px"}),
#                         dcc.Slider(
#                             id={"type": "stepsize-slider", "index": uid},
#                             min=-0.1, max=0.1, step=0.01, value=0.01,
#                             marks={-0.1: '-0.1', 0: '0', 0.1: '0.1'},
#                             tooltip={"placement": "bottom", "always_visible": True}
#                         ),
#                     ], style={'flex': '1', 'paddingRight': '10px'}),
#                     html.Div([
#                         html.Label("Max Steps", style={"fontWeight": "bold", "marginBottom": "5px"}),
#                         dcc.Slider(
#                             id={"type": "maxsteps-slider", "index": uid},
#                             min=10, max=500, step=10, value=100,
#                             marks={10: '10', 250: '250', 500: '500'},
#                             tooltip={"placement": "bottom", "always_visible": True}
#                         ),
#                     ], style={'flex': '1', 'paddingLeft': '10px'}),
#                 ], style={'display': 'flex', 'flexDirection': 'row', 'marginTop': '30px'}),
#             ]
#         ),

#         html.Hr()
#     ], style={'marginBottom': '30px'})

#     children.append(new_block)
#     return children


# — helper to run a single move synchronously —
def run_move(uid, a, b, c, stepsize, maxsteps, stored_data, transition, prev_pose=None):
    df = pd.DataFrame.from_dict(stored_data[uid])
    pc0 = np.array(df[df.iloc[:,0]=="PC_0"].iloc[0,1:].tolist(), float)
    pc1 = np.array(df[df.iloc[:,0]=="PC_1"].iloc[0,1:].tolist(), float)
    pc2 = np.array(df[df.iloc[:,0]=="PC_2"].iloc[0,1:].tolist(), float)
    blended_dq = a*pc0 + b*pc1 + c*pc2

    robot_qi = np.array(
        df[df.iloc[:,0]=="robot_qi"].iloc[0,1:].tolist(), 
        dtype=float
    )
    ## Re-init keys
    if transition == "nojump" and prev_pose is not None:
        robot_qi = prev_pose.copy()

    # write qi to Redis
    redis_client.set(ROBOT_QI_KEY, array_to_redis_string(robot_qi))

    dq_accumulator = np.zeros_like(blended_dq)
    redis_client.set(ROBOT_DQ_KEY, array_to_redis_string(dq_accumulator))
    
    if prev_pose is not None and transition in ["lerp", "cosine"]:
        # select blending scheme
        blend_steps = 10000
        for i in range(blend_steps):
            a = i / (blend_steps - 1)
            if transition == "lerp": 
                q_interp = a*robot_qi + (1-a)*prev_pose
                redis_client.set(ROBOT_QI_KEY, array_to_redis_string(q_interp))
            elif transition == "cosine":
                weight = 0.5 * (1 - np.cos(np.pi * a))
                q_interp = (1 - weight) * prev_pose + weight * robot_qi
                redis_client.set(ROBOT_QI_KEY, array_to_redis_string(q_interp))
    else:
        redis_client.set(ROBOT_DQ_KEY, array_to_redis_string(robot_qi))

    if not (transition == "nojump" and prev_pose is not None):
        redis_client.set(ROBOT_QI_KEY, array_to_redis_string(robot_qi))
    # 3) blend & normalize
    blended_dq = a*pc0 + b*pc1 + c*pc2
    norm = np.linalg.norm(blended_dq)
    if norm > 1e-6:
        blended_dq /= norm

    # 4) reset-and-run accumulator
    dq_accumulator = np.zeros_like(blended_dq)
    interval = 1.0/100.0
    for _ in range(maxsteps):
        dq_accumulator += stepsize * blended_dq
        # publish full position: qi + accumulated Δq
        redis_client.set(
            ROBOT_DQ_KEY, 
            array_to_redis_string(dq_accumulator)
        )
        time.sleep(interval)
    return dq_accumulator+robot_qi


@app.callback(
    Output("generate-btn","n_clicks", allow_duplicate=True),
    Input("generate-btn","n_clicks"),
    State({"type":"pc0-slider","index":ALL},"value"),
    State({"type":"pc1-slider","index":ALL},"value"),
    State({"type":"pc2-slider","index":ALL},"value"),
    State({"type":"stepsize-slider","index":ALL},"value"),
    State({"type":"maxsteps-slider","index":ALL},"value"),
    State({"type":"upload","index":ALL},"id"),
    State("file-data-store","data"),
    State("transition-dropdown", "value"), 
    prevent_initial_call=True
)
def generate_and_play(n_clicks,
                      pc0_vals, pc1_vals, pc2_vals,
                      stepsize_vals, maxsteps_vals,
                      upload_ids, stored_data, transition):

    if not stored_data:
        return dash.no_update
    
    prev_pose=None
    for idx, id_dict in enumerate(upload_ids):
        uid     = id_dict["index"]
        a, b, c = pc0_vals[idx], pc1_vals[idx], pc2_vals[idx]
        stepsz  = stepsize_vals[idx]
        msteps  = int(maxsteps_vals[idx])

        prev_pose = run_move(uid, a, b, c, stepsz, msteps, stored_data, transition, prev_pose)
    return 0

# --- Expand/Collapse Callback ---
@app.callback(
    Output({"type": "collapse", "index": MATCH}, "is_open"),
    Output({"type": "arrow", "index": MATCH}, "children"),
    Input({"type": "arrow", "index": MATCH}, "n_clicks"),
    State({"type": "collapse", "index": MATCH}, "is_open"),
    prevent_initial_call=True
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks:
        new_open = not is_open
        return new_open, "🔽" if new_open else "▶️"
    return is_open, "▶️"


def add_upload_block(n_clicks, children, default_filename=None):
    if n_clicks == 0:
        return children

    uid = str(uuid.uuid4())
    index = len(children)

    # if we have a default file, show its name; else show the placeholder text
    upload_children = (
        html.Div([f"Uploaded: {default_filename}"])
        if default_filename else
        html.Div(["Drag or click to upload a synergy .csv"])
    )

    new_block = html.Div(
        [
            html.H5(
                f"Move {index+1}",
                style={"marginBottom": "10px", "fontWeight": "bold"},
            ),
            html.Div(
                [
                    html.Div(
                        id={"type": "arrow", "index": uid},
                        children="▶️",
                        style={
                            "fontSize": "24px",
                            "cursor": "pointer",
                            "marginRight": "10px",
                            "marginTop": "8px",
                        },
                    ),
                    dcc.Upload(
                        id={"type": "upload", "index": uid},
                        children=upload_children,
                        accept=".csv",
                        multiple=False,
                        style={
                            "width": "100%",
                            "padding": "10px",
                            "border": "1px dashed #ccc",
                            "borderRadius": "5px",
                            "textAlign": "center",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "marginBottom": "10px",
                },
            ),
            dbc.Collapse(
                id={"type": "collapse", "index": uid},
                is_open=False,
                children=[
                    html.Label("PC 1", style={"fontWeight": "bold", "fontSize": "20px"}),
                    dcc.Slider(
                        id={"type": "pc0-slider", "index": uid},
                        min=-1,
                        max=1,
                        step=0.01,
                        value=0.5,
                        marks={-1: "-1", 0: "0", 1: "1"},
                        tooltip={"always_visible": True},
                    ),
                    html.Label(
                        "PC 2", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "20px"}
                    ),
                    dcc.Slider(
                        id={"type": "pc1-slider", "index": uid},
                        min=-1,
                        max=1,
                        step=0.01,
                        value=0,
                        marks={-1: "-1", 0: "0", 1: "1"},
                        tooltip={"always_visible": True},
                    ),
                    html.Label(
                        "PC 3", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "20px"}
                    ),
                    dcc.Slider(
                        id={"type": "pc2-slider", "index": uid},
                        min=-1,
                        max=1,
                        step=0.01,
                        value=0,
                        marks={-1: "-1", 0: "0", 1: "1"},
                        tooltip={"always_visible": True},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Step Size",
                                        style={"fontWeight": "bold", "marginBottom": "5px"},
                                    ),
                                    dcc.Slider(
                                        id={"type": "stepsize-slider", "index": uid},
                                        min=-0.1,
                                        max=0.1,
                                        step=0.01,
                                        value=0.01,
                                        marks={-0.1: "-0.1", 0: "0", 0.1: "0.1"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                style={"flex": "1", "paddingRight": "10px"},
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Max Steps",
                                        style={"fontWeight": "bold", "marginBottom": "5px"},
                                    ),
                                    dcc.Slider(
                                        id={"type": "maxsteps-slider", "index": uid},
                                        min=10,
                                        max=500,
                                        step=10,
                                        value=100,
                                        marks={10: "10", 250: "250", 500: "500"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                style={"flex": "1", "paddingLeft": "10px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "marginTop": "30px",
                        },
                    ),
                ],
            ),
            html.Hr(),
        ],
        style={"marginBottom": "30px"},
    )

    children.append(new_block)
    return children



@app.callback(
    Output("file-slider-container","children"),
    Output("file-data-store","data"),
    Input("add-file-btn","n_clicks"),
    Input("upload-multi","contents"),
    State("upload-multi","filename"),
    State("file-slider-container","children"),
    State("file-data-store","data"),
    prevent_initial_call=True
)
def add_blocks(add_clicks, multi_contents, multi_fnames, children, stored):
    new_children = list(children)
    new_store    = dict(stored)
    trigger = ctx.triggered_id

    if trigger == "add-file-btn":
        # manual single‐block
        new_children = add_upload_block(add_clicks, new_children)

    elif trigger == "upload-multi" and multi_contents:
        for content, fname in zip(multi_contents, multi_fnames):
            # 1) decode & read CSV
            header, b64 = content.split(",",1)
            decoded     = base64.b64decode(b64)
            df          = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            # 2) store under a new UID
            uid = str(uuid.uuid4())
            new_store[uid] = df.to_dict()

            # 3) add a block pre‐populated with that default file
            new_children = add_upload_block(
                add_clicks,
                new_children,
                default_filename=fname
            )

    return new_children, new_store



# --- Parse Uploaded Files ---
@app.callback(
    Output("file-data-store", "data", allow_duplicate=True),
    Input({"type": "upload", "index": ALL}, "contents"),
    State({"type": "upload", "index": ALL}, "id"),
    State("file-data-store", "data"),
    prevent_initial_call="initial_duplicate"
)
def parse_uploaded_files(contents_list, ids, stored_data):
    if not contents_list:
        return dash.no_update

    for content, id_dict in zip(contents_list, ids):
        if content:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            stored_data[id_dict['index']] = df.to_dict()

            robot_qi = np.array(list(df[df.iloc[:,0] == "robot_qi"].iloc[0, 1:]))
            robot_qf = np.array(list(df[df.iloc[:,0] == "robot_qf"].iloc[0, 1:]))

            redis_client.set(ROBOT_QI_KEY, array_to_redis_string(robot_qi))
            redis_client.set(ROBOT_QF_KEY, array_to_redis_string(robot_qf))

    return stored_data

# --- Handle Slider Change and Start Publishing ---
@app.callback(
    Output("add-file-btn", "n_clicks", allow_duplicate=True),
    Input({"type": "pc0-slider", "index": ALL}, "value"),
    Input({"type": "pc1-slider", "index": ALL}, "value"),
    Input({"type": "pc2-slider", "index": ALL}, "value"),
    Input({"type": "stepsize-slider", "index": ALL}, "value"),
    Input({"type": "maxsteps-slider", "index": ALL}, "value"),
    State({"type": "pc0-slider", "index": ALL}, "id"),
    State("file-data-store", "data"),
    prevent_initial_call="initial_duplicate"
)
def update_and_publish(pc0_vals, pc1_vals, pc2_vals, stepsize_vals, maxsteps_vals, ids, stored_data):
    if not stored_data:
        return dash.no_update

    triggered_idx = None
    for idx, id_dict in enumerate(ids):
        if ctx.triggered_id and ctx.triggered_id['index'] == id_dict['index']:
            triggered_idx = id_dict['index']
            break

    if triggered_idx is None:
        return dash.no_update

    df_dict = stored_data[triggered_idx]
    df = pd.DataFrame.from_dict(df_dict)

    robot_pc0 = np.array(list(df[df.iloc[:,0] == "PC_0"].iloc[0, 1:]))
    robot_pc1 = np.array(list(df[df.iloc[:,0] == "PC_1"].iloc[0, 1:]))
    robot_pc2 = np.array(list(df[df.iloc[:,0] == "PC_2"].iloc[0, 1:]))

    idx_slider = [i for i, id_dict in enumerate(ids) if id_dict['index'] == triggered_idx][0]

    a = pc0_vals[idx_slider]
    b = pc1_vals[idx_slider]
    c = pc2_vals[idx_slider]
    stepsize = stepsize_vals[idx_slider]
    maxsteps = int(maxsteps_vals[idx_slider])

    blended_dq = a * robot_pc0 + b * robot_pc1 + c * robot_pc2
    norm = np.linalg.norm(blended_dq)
    if norm > 1e-6:
        blended_dq = blended_dq / norm

    if triggered_idx in publishing_threads:
        publishing_threads[triggered_idx].do_run = False  # stop previous thread

    def publisher(blended_dq, stepsize, maxsteps):
        t = threading.current_thread()
        dq_accumulator = np.zeros_like(blended_dq)
        interval = 1.0 / 100.0  # 100 Hz

        for _ in range(maxsteps):
            if not getattr(t, "do_run", True):
                break
            dq_accumulator += stepsize * blended_dq
            redis_client.set(ROBOT_DQ_KEY, array_to_redis_string(dq_accumulator))
            time.sleep(interval)

        print("✅ Finished publishing all steps.")

    pub_thread = threading.Thread(target=publisher, args=(blended_dq, stepsize, maxsteps))
    pub_thread.start()
    pub_thread.do_run = True
    publishing_threads[triggered_idx] = pub_thread

    return dash.no_update

if __name__ == '__main__':
    app.run(debug=True)
