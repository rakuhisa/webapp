import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import dash_table
import base64
import io
import plotly.graph_objs as go

app = dash.Dash(__name__)

common_style= {'font-family':'Comic Sans MS','textAlign':'center','color':'#E6E6E6','background-color':'#3C3C3C'}

#dataset = pd.read_csv('D:/python_data_analysis_ohmsha-master/iris_without_species.csv',index_col=0)

models = {'2':2,
          '3':3,
          '4':4,
          '5':5,
          '6':6,
          '7':7,
          '8':8,
          '9':9,}

app.layout = html.Div([
    html.H1(children='K-means Clustering Analysis  Application'),
        
    html.H3('Only .csv, .xlsx'),
    
    html.H3('index: column1, column: row1 in excel sheet'),
     
    dcc.Upload(id='upload-data',
               children=html.Div([
                   'Drag and Drop or ',
                   html.A('Select Files')
               ]),
               style={
                   'width': '60%',
                   'height': '60px',
                   'lineHeight': '60px',
                   'borderWidth': '1px',
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   'margin': '0 auto'
               },
                multiple=True
              ),
    html.Br(),
    
    html.Div(
        dcc.Loading(
            id='loading-1',
            children=[
                dash_table.DataTable(
                    id='output-data-upload',
                    column_selectable='multi',
                    fixed_rows={'headers':True,'data':0},
                    style_table={
                        'overflowX':'scroll',
                        'overflowY':'scroll',
                        'maxHeight':'250px',
                        'color':'#E6E6E6',
                        'background-color':'#3C3C3C'
                    },
                    style_header={
                        'fontWeight':'bold',
                        'textAlign':'center'},
                    style_cell={
                        'color':'#E6E6E6',
                        'background-color':'#3C3C3C',
                        'maxWidth': 0,
                    },
                )
            ],
            type='cube'
        ),
        style={
                'height': '300px'
            }),
    html.H3('number of clusters',
            style={'textAlign':'start',
                   'text-indent':'10px'
                  }
           ),
    
    
    dcc.Dropdown(id='model-dropdown',
                 options=[{'label': k, 'value': k} for k in models.keys()],
                 style={'color':'#3C3C3C',
                        'background-color':'#E6E6E6',
                        'textAlign': 'start'
                       },
                 placeholder='Select number of clusters.'),
              
    html.Br(),
    
    html.Div(
        dcc.Loading(
            id='loading-2',
            children=[
                html.H3(id='explained_variance_ratio_'),
                
                dcc.Graph(id='cluster-plot')
            ],
            type='cube'
        ),
    ),
    
    html.Br(),
    
    html.Div(
        dcc.Loading(
            id='loading-3',
            children=[
                dash_table.DataTable(
                    id='final-data',
                    export_format='csv',
                    column_selectable='multi',
                    fixed_rows={'headers':True,'data':0},
                    style_table={
                        'overflowX':'scroll',
                        'overflowY':'scroll',
                        'maxHeight':'250px',
                        'color':'#E6E6E6',
                        'background-color':'#3C3C3C'
                    },
                    style_header={
                        'fontWeight':'bold',
                        'textAlign':'center'},
                    style_cell={
                        'color':'#E6E6E6',
                        'background-color':'#3C3C3C',
                        'maxWidth': 0,
                        
                    },
                )
            ],
            type='cube'
        ),
        style={
                'height': '300px'
            }),
                        
],
    style=common_style
)

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),index_col=0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel((io.BytesIO(decoded)),index_col=0)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    data_ = df.to_dict('records')
    columns_ = [{'name': i, 'id': i} for i in df.columns]

    return [data_, columns_]

@app.callback([Output('output-data-upload', 'data'),
               Output('output-data-upload', 'columns')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    # ファイルがない時の自動コールバックを防ぐ
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    contents = [parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]

    return [contents[0][0], contents[0][1]]

@app.callback([
    Output(component_id='explained_variance_ratio_', component_property='children'),
    Output(component_id='cluster-plot', component_property='figure'),
    Output(component_id='final-data',component_property='data'),
    Output(component_id='final-data',component_property='columns')],
    [Input(component_id='model-dropdown', component_property='value'),
    Input(component_id='output-data-upload', component_property='data')]
     )

def update_result(number_of_clusters,dict_data):
    
    if dict_data is None or number_of_clusters is None:
        raise dash.exceptions.PreventUpdate
    
    dataset=pd.DataFrame(data=dict_data)
    
    number=models[number_of_clusters]
    
    pca = PCA(n_components=2)

    scaler=StandardScaler()

    feature_std=scaler.fit_transform(dataset)
    features_pca=pca.fit_transform(feature_std)

    score=pd.DataFrame(pca.transform(feature_std),columns=["pc1","pc2"])
    score.index=dataset.index

    cluster=KMeans(n_clusters=number,random_state=0,n_jobs=-1)
    model=cluster.fit(feature_std)

    cluster_str=list(map(str,model.labels_))

    label_=pd.DataFrame(cluster_str,index=dataset.index,columns=["cluster_label"])

    df_concat = pd.concat([score, label_],axis=1)

    contribution_ratios=pd.DataFrame(pca.explained_variance_ratio_)
    
    fig=px.scatter(df_concat, x="pc1",y="pc2",template="plotly_dark",color="cluster_label")
    
    data_1 = df_concat.to_dict('records')
    columns_1 = [{'name': i, 'id': i} for i in df_concat.columns]

    
    return[
        f' Percentage of variance {pca.explained_variance_ratio_}',
        fig,
        data_1, 
        columns_1
    ]
        
if __name__ == '__main__':
    app.run_server(debug=True)
