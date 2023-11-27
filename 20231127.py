# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:46:54 2023

@author: matoda
"""
import importlib
import subprocess

packages_to_check = ["pandas", "networkx", "plotly", "dash", "re"]

def install_if_not_exists(package_name):
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is already installed.")
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call(["pip", "install", package_name])
        except Exception as e:
            print(f"Failed to install {package_name}. Error: {e}")
        else:
            print(f"{package_name} has been successfully installed.")

# Check and install each package
for package in packages_to_check:
    install_if_not_exists(package)

import re
import pandas as pd
import networkx as nx
import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


# Load the data
data = pd.read_excel('G:/내 드라이브/StarSeed/Starseed 스킬네트워크 작업.xlsx', sheet_name='스킬효과 작업 내역')
data = data.dropna(subset = ['skill'])
data['char_name_kr'] = '[캐릭터]' + data['char_name_kr']
data.loc[(data.skill == '액티브1'), 'skill'] = '액티브'
data.loc[(data.skill == '액티브2'), 'skill'] = '액티브'
data['frist'] = '[우선대상]['+ data['target'] + ']' + data['우선순위']
data['skill'] = '['+ data['skill'] + ']' + data['스킬명']
data['target'] = '['+ data['target'] + ']' +'['+ data['range(자신/단일/전체/범위)'] + ']'
data['target_effect'] = '['+ data['target'] + ']' +data['effect']
data['trigger'] = '[조건]['+ data['if_1'] + ']'
data.loc[(data['if_2'].notnull()), 'trigger'] = data['trigger'] + '['+ data['if_2'] + ']'
data = data[['char_name_kr', 'skill', 'frist', 'trigger', 'target_effect']]


# Data transformation
# 캐릭터 스킬(순수하게 캐릭터-스킬)
dataA = data[data['frist'].isnull()]
dataA = dataA[dataA['trigger'].isnull()]
dataA = dataA[['char_name_kr', 'target_effect']].drop_duplicates().reset_index(drop=True)
dataA.columns = ['name', 'effect']

# 캐릭터-우선순위-스킬효과
dataB = data.dropna(subset = ['frist'])
dataB = dataB[['char_name_kr', 'frist', 'target_effect']].dropna().drop_duplicates().reset_index(drop=True)
dataB.columns = ['name', 'effect','trigger_effect']

# 캐릭터-조건-스킬효과
dataC = data.dropna(subset = ['trigger'])
dataC = dataC[['char_name_kr', 'trigger', 'target_effect']].dropna().drop_duplicates().reset_index(drop=True)
dataC.columns = ['name', 'effect','trigger_effect']

# 기존 'source-target' 엣지와 'trigger' 엣지를 병합
effect_combined = pd.concat([dataA, dataB, dataC])



G_final_combined = nx.Graph()
# effect_combined 데이터를 사용하여 연결 구성
for _, row in effect_combined.iterrows():
    G_final_combined.add_node(row['name'], type='character')
    G_final_combined.add_node(row['effect'], type='effect')
    G_final_combined.add_edge(row['name'], row['effect'], type='name_effect')

    if 'trigger_effect' in row and pd.notna(row['trigger_effect']):
        G_final_combined.add_node(row['trigger_effect'], type='trigger_effect')
        G_final_combined.add_edge(row['effect'], row['trigger_effect'], type='effect_trigger_effect')

# Plotly를 사용한 네트워크 그래프 시각화 준비
pos_final_combined = nx.spring_layout(G_final_combined, seed=42)


# 그룹별 색상 지정
group_colors = {1: 'blue', 2: 'green', 3: 'cyan', 4: 'magenta'}

# 그룹 데이터 생성
group1 = data[['char_name_kr']].drop_duplicates().reset_index(drop=True)
group1.columns = ['name']
group1['group'] = 1

group2 = data[['frist']].dropna().drop_duplicates().reset_index(drop=True)
group2.columns = ['name']
group2['group'] = 2

group3 = data[['trigger']].dropna().drop_duplicates().reset_index(drop=True)
group3.columns = ['name']
group3['group'] = 3

group4 = data[['target_effect']].dropna().drop_duplicates().reset_index(drop=True)
group4.columns = ['name']
group4['group'] = 4

# 그룹 병합 및 중복 제거
effect_num = pd.concat([group1, group2, group3, group4]).drop_duplicates().reset_index(drop=True)

# 각 노드의 색상 할당
node_color = []
for node in G_final_combined.nodes():
    group_df = effect_num[effect_num['name'] == node]
    if not group_df.empty:
        group = group_df['group'].values[0]
        color = group_colors.get(group, 'grey')
        print(f"Node: {node}, Group: {group}, Color: {color}")  # 디버깅 출력
        node_color.append(color)
    else:
        print(f"Node: {node} not found in effect_num")  # 노드가 effect_num에 없는 경우
        node_color.append('grey')
        
# Dash 앱 초기화
app = dash.Dash(__name__)

# 그래프 시각화 함수
def generate_network_graph(selected_node):
    edge_x = []
    edge_y = []
    for edge in G_final_combined.edges():
        x0, y0 = pos_final_combined[edge[0]]
        x1, y1 = pos_final_combined[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos_final_combined[node][0] for node in G_final_combined.nodes()]
    node_y = [pos_final_combined[node][1] for node in G_final_combined.nodes()]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_size = [len(list(G_final_combined.neighbors(node))) for node in G_final_combined.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            #size=node_size,
            size=node_size,
            color=node_color,
            symbol=['star' if node in group1['name'].values else 'circle' for node in G_final_combined.nodes()],
            line_width=2))

    node_trace.text = [node for node in G_final_combined.nodes()]

    selected_edge_x = []
    selected_edge_y = []
    selected_node_x = []
    selected_node_y = []

    if selected_node:
        for edge in G_final_combined.edges():
            if edge[0] == selected_node or edge[1] == selected_node:
                x0, y0 = pos_final_combined[edge[0]]
                x1, y1 = pos_final_combined[edge[1]]
                selected_edge_x.extend([x0, x1, None])
                selected_edge_y.extend([y0, y1, None])

        for edge in G_final_combined.edges():
            if edge[0] == selected_node:
                x, y = pos_final_combined[edge[1]]
                selected_node_x.append(x)
                selected_node_y.append(y)
            elif edge[1] == selected_node:
                x, y = pos_final_combined[edge[0]]
                selected_node_x.append(x)
                selected_node_y.append(y)

    selected_edge_trace = go.Scatter(
        x=selected_edge_x, y=selected_edge_y,
        line=dict(width=2, color='red'),
        hoverinfo='none',
        mode='lines',
        name='highlighted_edges')
    # 노드에 연결된 엣지의 개수에 따라 크기 동적 설정
    selected_node_trace = go.Scatter(
        x=selected_node_x, y=selected_node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            #size=node_size,
            size=10,#이 사이즈는 고정해야함/노드사이즈로하면 오류남
            color='red',  # 선택된 노드의 색상을 빨간색으로 지정
            line_width=4))

    selected_node_trace.text = [selected_node]

    return [edge_trace, node_trace, selected_edge_trace, selected_node_trace]

# Dash 앱 레이아웃
app.layout = html.Div([    
    dcc.Graph(id='network-graph', config={'displayModeBar': False}, style={'width': '1200px', 'height': '1000px'}),
    html.Div(id='node-info', style={'margin-top': '20px', 'font-size': '16px'}),
    # 테이블 컴포넌트를 포함하는 div에 스타일을 적용
    html.Div(
        dash_table.DataTable(
            id='node-table',
            columns=[],  # 초기 컬럼 정의
            data=[]      # 초기 데이터
        ),
        style={'margin-top': '20px'}  # 여기에 스타일 적용
    )
])


def find_related_characters(selected_node, data):
    # 데이터셋에서 선택된 노드의 값이 포함된 모든 행을 반환
    selected_node_escaped = re.escape(selected_node)
    mask = data.apply(lambda col: col.str.contains(selected_node_escaped)).any(axis=1)
    related_data = data[mask]
    return related_data

@app.callback(
    [Output('network-graph', 'figure'), 
     Output('node-info', 'children'),
     Output('node-table', 'data'),
     Output('node-table', 'columns')],
    [Input('network-graph', 'clickData')]
)
def update_graph(clickData):
    selected_node = None
    node_info_content = 'Select a node to see its details.'
    table_data = []
    table_columns = []

    if clickData and 'points' in clickData and len(clickData['points']) > 0:
        if 'text' in clickData['points'][0]:
            selected_node = clickData['points'][0]['text']
            node_info_content = f'Selected Node: {selected_node}'

            # 모든 컬럼에서 선택된 노드의 값을 포함하는 모든 행을 가져옴
            related_data = find_related_characters(selected_node, data)
            table_data = related_data.to_dict('records')
            table_columns = [{"name": col, "id": col} for col in related_data.columns]

    edge_trace, node_trace, selected_edge_trace, selected_node_trace = generate_network_graph(selected_node)

    return (
        {
            'data': [edge_trace, node_trace, selected_edge_trace, selected_node_trace],
            'layout': go.Layout(
                title='<br>Network graph with Dash',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        },
        node_info_content,
        table_data,
        table_columns
    )
# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
