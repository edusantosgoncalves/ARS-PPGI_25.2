# Dependencias (Py)
import re
import json
import os

# Dependencias (Bibliotecas)
import pandas as pd
import networkx as nx

# Visualização
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from pyvis.network import Network

    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

from collections import defaultdict
from itertools import combinations
import numpy as np

# Caminho do dataset
CSV_PATH = os.path.join(os.path.dirname(__file__), "../Attack_Dataset.csv")

# Lendo dataset e selecionando colunas de interesse
df = pd.read_csv(CSV_PATH)

required_cols = {"ID", "Category", "Tools Used", "MITRE Technique"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Colunas ausentes no dataset: {missing}")

# Determinando padrões de extração
MITRE_REGEX = re.compile(r"(T\d{4}(?:\.\d{1,3})?)", flags=re.IGNORECASE)
SPLIT_REGEX = re.compile(r"[;,/|\\\n]+|\s-\s|\s—\s")


# Funções auxiliares
def extract_mitre_codes(text: str) -> list:
    if pd.isna(text):
        return []
    raw = str(text)
    matches = MITRE_REGEX.findall(raw)
    seen = set()
    codes = []
    for match in matches:
        base = match.upper().split(".", 1)[0]
        if base and base not in seen:
            seen.add(base)
            codes.append(base)
    return codes


def split_tools(text: str) -> list:
    if pd.isna(text):
        return []
    raw = str(text)
    # remover emojis e selectors de variação comuns
    emoji_re = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # símbolos & pictogramas
        "\U0001f680-\U0001f6ff"  # transportes & mapas
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002700-\U000027bf"  # dingbats
        "\U00002600-\U000026ff"  # misc symbols
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U0001fa70-\U0001faff"  # symbols & pictographs extended-A
        "\ufe0f"  # variation selector
        "]+",
        flags=re.UNICODE,
    )
    cleaned = emoji_re.sub("", raw)
    parts = [p.strip() for p in SPLIT_REGEX.split(cleaned) if p and p.strip()]
    parts = [" ".join(p.lower().split()) for p in parts]
    parts = [p.strip() for p in parts if p]
    return sorted(set(parts))


# Aplicando extrações
df["mitre_codes"] = df["MITRE Technique"].apply(extract_mitre_codes)
df["tools"] = df["Tools Used"].apply(split_tools)

# -----------------------------------------------------------------------------
# Construir rede bipartida: Tools Used x MITRE Technique
# Cada aresta representa a relação (tool, mitre) e agrega os IDs dos incidentes
# -----------------------------------------------------------------------------
# key: (tool, mitre) -> {'attack_ids': set, 'categories': set}
tool_mitre = {}

for idx, row in df.iterrows():
    incident_id = row["ID"]
    if pd.isna(incident_id):
        continue
    id_str = str(incident_id)
    tools = row.get("tools") or []
    mitres = row.get("mitre_codes") or []
    category = row.get("Category")
    for t in tools:
        for m in mitres:
            key = (t, m)
            if key not in tool_mitre:
                tool_mitre[key] = {"attack_ids": set(), "categories": set()}
            tool_mitre[key]["attack_ids"].add(id_str)
            if pd.notna(category):
                tool_mitre[key]["categories"].add(str(category))

if not tool_mitre:
    raise ValueError(
        "Nenhuma relação Tool Used x MITRE Technique encontrada após a normalização."
    )

# Construindo grafo bipartido Tool <-> MITRE (MultiGraph: uma aresta por ID)
# Criar mapeamento id -> category (para preencher atributo 'category' por aresta)
id_to_category = {}
for idx, row in df.iterrows():
    incident_id = row["ID"]
    if pd.isna(incident_id):
        continue
    id_to_category[str(incident_id)] = (
        str(row["Category"]) if pd.notna(row.get("Category")) else None
    )

# T: MultiGraph onde cada aresta corresponde a um ID (uma ocorrência)
T = nx.MultiGraph()
for (tool, mitre), attrs in tool_mitre.items():
    tool_node = f"TOOL::{tool}"
    mitre_node = f"MITRE::{mitre}"
    if not T.has_node(tool_node):
        T.add_node(tool_node, bipartite=0, label=tool, kind="TOOL")
    if not T.has_node(mitre_node):
        T.add_node(mitre_node, bipartite=1, label=mitre, kind="MITRE")
    # adicionar uma aresta por attack id
    for aid in sorted(attrs["attack_ids"]):
        cat = id_to_category.get(aid)
        T.add_edge(tool_node, mitre_node, id=aid, category=cat)

print(
    "Grafo bipartido (TOOL - MITRE) MultiGraph criado | nós:",
    T.number_of_nodes(),
    "| arestas (multiedges):",
    T.number_of_edges(),
)

# Exportando edgelist e CSVs do grafo TOOL-MITRE
# write_edgelist não preserva múltiplas arestas com atributos individuais; exportaremos CSV com cada aresta separada

# Exportando arestas com atributos completos (uma linha por aresta / ID)
edges_data = []
for u, v, key, d in T.edges(keys=True, data=True):
    edges_data.append(
        {
            "source": u,
            "target": v,
            "id": str(d.get("id")),
            "category": d.get("category"),
        }
    )

edges_out_df = pd.DataFrame(edges_data)
edges_out_df.to_csv(
    "./tool_mitre_bipartite_edges_by_id.csv", index=False, encoding="utf-8"
)
print(
    "Exportado (uma linha por ID):",
    "./tool_mitre_bipartite_edges_by_id.csv",
    "| linhas:",
    len(edges_out_df),
)

# Também exportar um edgelist agregada (peso = número de IDs por par)
agg_rows = []
for (tool, mitre), attrs in tool_mitre.items():
    agg_rows.append(
        {
            "Tool": tool,
            "MITRE": mitre,
            "weight": len(attrs["attack_ids"]),
            "attack_ids": json.dumps(sorted(attrs["attack_ids"]), ensure_ascii=False),
            "categories": json.dumps(sorted(attrs["categories"]), ensure_ascii=False),
        }
    )

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv("./tool_mitre_bipartite_edges_agg.csv", index=False, encoding="utf-8")
print(
    "Exportado (agregado por par):",
    "./tool_mitre_bipartite_edges_agg.csv",
    "| linhas:",
    len(agg_df),
)

# Exportando nós
nodes_data = []
for n, attrs in T.nodes(data=True):
    nodes_data.append(
        {
            "node": n,
            "label": attrs.get("label"),
            "bipartite": attrs.get("bipartite"),
            "kind": attrs.get("kind"),
        }
    )

nodes_df = pd.DataFrame(nodes_data)
nodes_df.to_csv("./tool_mitre_bipartite_nodes.csv", index=False, encoding="utf-8")
print("Exportado:", "./tool_mitre_bipartite_nodes.csv", "| nós:", len(nodes_df))


# ---------------------------------------------------
# Projetando a rede bipartida Tool–MITRE para uma rede Tool–Tool
# - Dois tools estão conectados se aparecem juntos em um mesmo MITRE em pelo menos um incidente (ID);
# - Para cada par:
#     * mitres: lista de MITREs em que ambos aparecem
#     * attack_ids: lista de incident IDs em que ambos aparecem (interseção por MITRE)
#     * attack_categories: categorias dos incidentes (a partir de id_to_category)
#     * weight: número total de incident IDs compartilhados (soma das interseções)
# ---------------------------------------------------

# edge_acc acumula dados por par de tools
edge_acc = defaultdict(
    lambda: {
        "mitres": set(),
        "attack_ids": set(),
        "attack_categories": set(),
        "weight": 0,
    }
)

# Para cada MITRE, obtenha a lista de tools e seus attack_ids
mitre_tools = defaultdict(list)  # mitre -> list of (tool, set(attack_ids))
for (tool, mitre), attrs in tool_mitre.items():
    mitre_tools[mitre].append(
        (tool, set(attrs["attack_ids"]), set(attrs["categories"]))
    )

# Para cada mitre, calcule pares de tools
for mitre, tool_list in mitre_tools.items():
    if len(tool_list) < 2:
        continue
    # combina pares de tools
    for (t1, ids1, cats1), (t2, ids2, cats2) in combinations(tool_list, 2):
        # incidentos em que ambos os tools aparecem para este mitre = interseção
        shared_ids = ids1.intersection(ids2)
        if not shared_ids:
            # Ignorando pares sem incidentes compartilhados
            continue
        key = tuple(sorted((f"TOOL::{t1}", f"TOOL::{t2}")))
        ed = edge_acc[key]
        ed["mitres"].add(mitre)
        ed["attack_ids"].update(shared_ids)
        # categorias para os incident ids (a partir do mapeamento id_to_category)
        for aid in shared_ids:
            cat = id_to_category.get(aid)
            if cat is not None:
                ed["attack_categories"].add(cat)
        ed["weight"] += len(shared_ids)

# Construindo o grafo projetado Tools–Tools
G_tools = nx.Graph()
# adicionar nós (somente tools presentes)
all_tools = set([n for n in T.nodes() if n.startswith("TOOL::")])
for tnode in all_tools:
    G_tools.add_node(tnode, bipartite=0, label=tnode.replace("TOOL::", ""), kind="TOOL")

for (u, v), attrs in edge_acc.items():
    attack_ids_list = sorted(attrs["attack_ids"])
    cats_list = sorted(attrs["attack_categories"])
    mitres_list = sorted(attrs["mitres"])
    G_tools.add_edge(
        u,
        v,
        weight=int(attrs["weight"]),
        mitres=mitres_list,
        attack_ids=attack_ids_list,
        attack_categories=cats_list,
    )

print(
    "Projeção TOOL-TOOL criada | nós:",
    G_tools.number_of_nodes(),
    "| arestas:",
    G_tools.number_of_edges(),
)


# helper para converter tipos numpy -> nativos
def _pyify(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return x


# Exportar arestas da projeção (uma linha por par de tools)
edges_data = []
for u, v, d in G_tools.edges(data=True):
    edges_data.append(
        {
            "source": _pyify(u),
            "target": _pyify(v),
            "weight": _pyify(d.get("weight", 1)),
            "mitres": json.dumps(d.get("mitres", []), ensure_ascii=False),
            "attack_ids": json.dumps(
                [str(_pyify(i)) for i in d.get("attack_ids", [])], ensure_ascii=False
            ),
            "attack_categories": json.dumps(
                [str(_pyify(c)) for c in d.get("attack_categories", [])],
                ensure_ascii=False,
            ),
        }
    )

edges_df = pd.DataFrame(edges_data)
edges_df.to_csv("./tool_projection_edges.csv", index=False, encoding="utf-8")
print("Exportado:", "./tool_projection_edges.csv", "| linhas:", len(edges_df))

# Exportar nós da projeção
nodes_data = []
for n, attrs in G_tools.nodes(data=True):
    nodes_data.append(
        {"node": _pyify(n), "label": attrs.get("label"), "kind": attrs.get("kind")}
    )

nodes_df = pd.DataFrame(nodes_data)
nodes_df.to_csv("./tool_projection_nodes.csv", index=False, encoding="utf-8")
print("Exportado:", "./tool_projection_nodes.csv", "| nós:", len(nodes_df))
