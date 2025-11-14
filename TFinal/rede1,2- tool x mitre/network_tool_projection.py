import pandas as pd
import networkx as nx
import json
import numpy as np
from metricas import plot_degree_distribution_complementar

# Matplotlib como fallback quando plotly não estiver disponível
import matplotlib.pyplot as plt

# Ler o dataset CSV (ajuste o caminho se necessário)
edges = pd.read_csv("./tool_projection_edges.csv")
nodes = pd.read_csv("./tool_projection_nodes.csv")

# Criar o grafo a partir dos dados
# . Inserindo todos os nós...
G = nx.Graph()

# . Inserindo todas as arestas...
for _, row in edges.iterrows():
    src = str(row["source"])
    tgt = str(row["target"])
    weight = int(row["weight"]) if not pd.isna(row["weight"]) else 1

    try:
        mitres = json.loads(row["mitres"]) if not pd.isna(row["mitres"]) else []
    except Exception:
        mitres = []

    try:
        attack_ids = (
            json.loads(row["attack_ids"]) if not pd.isna(row["attack_ids"]) else []
        )
    except Exception:
        attack_ids = []

    try:
        attack_categories = (
            json.loads(row["attack_categories"])
            if not pd.isna(row["attack_categories"])
            else []
        )
    except Exception:
        attack_categories = []

    if not G.has_node(src):
        G.add_node(src, label=src)
    if not G.has_node(tgt):
        G.add_node(tgt, label=tgt)

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
        edge_data["weight"] = edge_data.get("weight", 0) + weight

        if "attack_ids" not in edge_data or not isinstance(
            edge_data["attack_ids"], list
        ):
            edge_data["attack_ids"] = []
        edge_data["attack_ids"].extend([str(aid) for aid in attack_ids])

        if "attack_categories" not in edge_data or not isinstance(
            edge_data["attack_categories"], list
        ):
            edge_data["attack_categories"] = []
        for category in attack_categories:
            if category not in edge_data["attack_categories"]:
                edge_data["attack_categories"].append(category)

        continue
    else:
        G.add_edge(
            src,
            tgt,
            weight=weight,
            attack_ids=[str(aid) for aid in attack_ids],
            attack_categories=attack_categories,
        )


# . Inserindo todos os nós...
for _, row in nodes.iterrows():
    # Se o nó já existe, pular
    if G.has_node(row["node"]):
        continue
    G.add_node(row["node"], label=row["label"])


# * Obtendo métricas da rede
num_nos = G.number_of_nodes()
num_arestas = G.number_of_edges()
print(f"Nós: {num_nos}, Arestas: {num_arestas}")

# * Obtendo densidade
densidade = nx.density(G)
print(f"Densidade: {densidade}")

# * Obter coef. clusterização global e local (o quanto de "panelinha" tem, se é num todo ou em pequenas panelinhas)
coef_clusterizacao_local = nx.average_clustering(G)
coef_clusterizacao_global = nx.transitivity(G)

# * Grau medio (número médio de coocorrências por ataque)
graus = []
for node in G.nodes:
    graus.append(nx.degree(G, node))

grau_medio = np.mean(graus)

print(f"Grau médio: {grau_medio}")
print(f"Cluster global: {coef_clusterizacao_global}")
print(f"Cluster médio: {coef_clusterizacao_local}")

plot_degree_distribution_complementar(
    G,
    titles=["Tool-Tool Projection"],
    output_path="./",
    output_file_name="tool_projection_degree_distribution.png",
    show=False,
)


# Calcular posições dos nós (layout) e plotar
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(12, 12))
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_size=50, node_color="#FF5733")
nx.draw_networkx_labels(
    G,
    pos,
    labels={node: G.nodes[node]["label"] for node in G.nodes()},
    font_size=8,
)
plt.axis("off")
plt.savefig("projected_tool_network_graph.png", format="PNG")
plt.close()
print(f"Grafo exportado usando Matplotlib: projected_tool_network_graph.png.")
