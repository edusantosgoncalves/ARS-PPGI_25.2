import pandas as pd
import networkx as nx
import numpy as np
from metricas import plot_degree_distribution_complementar

# Importar bibliotecas opcionais com fallback
try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_plot

    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    plotly_plot = None
    PLOTLY_AVAILABLE = False

# Matplotlib como fallback quando plotly não estiver disponível
import matplotlib.pyplot as plt

MATPLOTLIB_AVAILABLE = True

# Ler o dataset CSV (ajuste o caminho se necessário)
edges = pd.read_csv("./tool_mitre_bipartite_edges_by_id.csv")
nodes = pd.read_csv("./tool_mitre_bipartite_nodes.csv")

# Criar o grafo a partir dos dados
G = nx.Graph()

# . Inserindo todas as arestas...
for _, row in edges.iterrows():
    src = str(row["source"])
    src_type = src.split("::")[0]
    bipartite_src = 0 if src_type == "TOOL" else 1
    tgt = str(row["target"])
    tgt_type = tgt.split("::")[0]
    bipartite_tgt = 0 if tgt_type == "TOOL" else 1
    attack_id = row["id"]
    category = row["category"]

    if not G.has_node(src):
        G.add_node(src, label=src, kind=src_type, bipartite=bipartite_src)
    if not G.has_node(tgt):
        G.add_node(tgt, label=tgt, kind=tgt_type, bipartite=bipartite_tgt)

    # Se já existe uma aresta entre src e tgt, agregar os dados
    existing_edges = G.get_edge_data(src, tgt, default=None)
    if existing_edges:
        # Pode ser Graph (um dict de atributos) ou MultiGraph (dict de chaves -> dict de atributos).
        # Detectar qual formato e extrair o dict de atributos correto.
        multiedge = False
        first_key = None
        if isinstance(existing_edges, dict) and any(
            isinstance(v, dict) for v in existing_edges.values()
        ):
            # MultiGraph style: escolher a primeira aresta (ou poderíamos iterar/mesclar todas)
            first_key = list(existing_edges.keys())[0]
            edge_data = existing_edges[first_key]
            multiedge = True
        else:
            # Graph style: existing_edges já é o dict de atributos da aresta
            edge_data = existing_edges

        # Garantir que temos um dict de atributos para atualizar
        if not isinstance(edge_data, dict):
            edge_data = {"weight": 0, "attack_ids": [], "attack_categories": []}
            if multiedge:
                G[src][tgt][first_key] = edge_data
            else:
                G[src][tgt] = edge_data

        # Atualizar contadores/coleções de forma segura
        edge_data["weight"] = edge_data.get("weight", 0) + 1

        if "attack_ids" not in edge_data or not isinstance(
            edge_data["attack_ids"], list
        ):
            edge_data["attack_ids"] = []
        edge_data["attack_ids"].append(str(attack_id))

        if "attack_categories" not in edge_data or not isinstance(
            edge_data["attack_categories"], list
        ):
            edge_data["attack_categories"] = []
        if category not in edge_data["attack_categories"]:
            edge_data["attack_categories"].append(category)

        continue
    else:
        G.add_edge(
            src,
            tgt,
            weight=1,
            attack_ids=[str(attack_id)],
            attack_categories=[category],
        )


# . Inserindo todos os nós...
for _, row in nodes.iterrows():
    # Se o nó já existe, pular
    if G.has_node(row["node"]):
        continue
    bipartite_value = 0 if row["kind"] == "TOOL" else 1
    G.add_node(
        row["node"], label=row["label"], kind=row["kind"], bipartite=bipartite_value
    )

# * Obtendo métricas da rede
num_nos = G.number_of_nodes()
num_tools_nos = len([n for n, d in G.nodes(data=True) if d.get("kind") == "TOOL"])
num_mitre_nos = len([n for n, d in G.nodes(data=True) if d.get("kind") == "MITRE"])
print(f"Nós totais: {num_nos} | Tools: {num_tools_nos} | MITRE: {num_mitre_nos}")
num_arestas = G.number_of_edges()
print(f"Arestas: {num_arestas}")

# * Obtendo densidade
densidade = nx.density(G)
print(f"Densidade: {densidade}")

# * Grau medio (número médio de coocorrências por ataque)
graus = []
for node in G.nodes:
    graus.append(nx.degree(G, node))

grau_medio = np.mean(graus)

print(f"Grau médio: {grau_medio}")

plot_degree_distribution_complementar(
    G,
    titles=["Tool-MITRE Bipartido"],
    output_path="./",
    output_file_name="tool_bipartite_degree_distribution.png",
    show=False,
)


# Estilização (seaborn quando disponível)
if sns is not None:
    sns.set_style("whitegrid")

# Detectar se o grafo é bipartido e calcular layout apropriado
is_bipartite = False
try:
    is_bipartite = nx.is_bipartite(nx.Graph(G))
except Exception:
    is_bipartite = False

if is_bipartite:
    H = nx.Graph(G)  # usar versão simples para análise bipartida
    try:
        top_nodes, bottom_nodes = nx.bipartite.sets(H)
        # garantir listas (ordenadas para layout reprodutível)
        top_nodes = list(top_nodes)
        bottom_nodes = list(bottom_nodes)
    except Exception:
        # fallback caso sets falhe (raro)
        all_nodes = list(H.nodes())
        top_nodes = all_nodes[::2]
        bottom_nodes = [n for n in all_nodes if n not in top_nodes]

    # layout bipartido: um conjunto à esquerda, outro à direita
    pos = nx.bipartite_layout(H, top_nodes)
else:
    # não bipartido: fallback para spring layout
    print("Grafo não detectado como bipartido. Usando spring layout.")
    pos = nx.spring_layout(G, seed=42)
    top_nodes = None
    bottom_nodes = None


# Exporte uma imagem da rede (agora com suporte específico para bipartido)
def export_network_image(graph, positions, filename="network_graph.html"):
    # decidir se usará bipartido (tem as partições definidas)
    use_bip = top_nodes is not None and bottom_nodes is not None

    if PLOTLY_AVAILABLE:
        edge_x = []
        edge_y = []
        for u, v, *_ in graph.edges(data=True):
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # separar nós por partição (se disponível) para cores/tamanhos distintos
        def node_trace_for(nodes_list, color, size):
            node_x = []
            node_y = []
            texts = []
            for n in nodes_list:
                x, y = positions[n]
                node_x.append(x)
                node_y.append(y)
                texts.append(graph.nodes[n].get("label", str(n)))
            return go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=texts,
                textposition="top center",
                hoverinfo="text",
                marker=dict(showscale=False, color=color, size=size, line_width=1),
            )

        traces = [edge_trace]

        if use_bip:
            traces.append(node_trace_for(top_nodes, "#1f77b4", 16))  # azul (ex.: tools)
            traces.append(
                node_trace_for(bottom_nodes, "#ff7f0e", 12)
            )  # laranja (ex.: MITRE)
        else:
            # todos em uma única camada
            traces.append(
                node_trace_for(list(graph.nodes()), "#FF5733", 10),
            )

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # garantindo extensão .html para saída interativa
        out = filename if filename.lower().endswith(".html") else filename + ".html"
        plotly_plot(fig, filename=out, auto_open=False)
        print(f"Network graph exported to {out} using Plotly.")
    elif MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 8))
        # desenhando arestas
        nx.draw_networkx_edges(graph, positions, alpha=0.5)

        # desenhando nós por partição
        if use_bip:
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=top_nodes,
                node_size=300,
                node_color="#1f77b4",
                label="Partição A",
            )
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=bottom_nodes,
                node_size=150,
                node_color="#ff7f0e",
                label="Partição B",
            )
        else:
            nx.draw_networkx_nodes(
                graph,
                positions,
                node_size=100,
                node_color="#FF5733",
            )

        # rótulos mais enxutos para não poluir o grafo
        labels = {n: graph.nodes[n].get("label", str(n)) for n in graph.nodes()}
        nx.draw_networkx_labels(
            positions=positions, G=graph, labels=labels, font_size=8
        )

        plt.axis("off")
        # salvar em PNG por padrão para matplotlib
        out = filename if filename.lower().endswith(".png") else filename + ".png"
        plt.savefig(out, format="PNG", bbox_inches="tight")
        plt.close()
        print(f"Network graph exported to {out} using Matplotlib.")
    else:
        print("No plotting library available to export the network graph.")


# exportando...
export_network_image(G, pos, filename="bipartite_tool_network_graph.html")
