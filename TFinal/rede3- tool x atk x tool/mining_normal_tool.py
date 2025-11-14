# Dependencias (Py)
import re
import os

# Dependencias (Bibliotecas)
import pandas as pd
import networkx as nx

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
# Construir rede: Tools Used x Tools
# Cada aresta representa a relação (tool, tool) por realização no mesmo id
# -----------------------------------------------------------------------------
# key: (tool, tool) -> {'attack_ids': set, 'categories': set}
tool_tool = {}

for idx, row in df.iterrows():
    incident_id = row["ID"]
    if pd.isna(incident_id):
        continue
    id_str = str(incident_id)
    tools = row.get("tools") or []
    mitres = row.get("mitre_codes") or []
    category = row.get("Category")
    for t in tools:
        for m in tools:
            if t == m:
                continue

            key = (t, m)
            if key not in tool_tool:
                tool_tool[key] = {"attack_ids": set(), "categories": set()}
            tool_tool[key]["attack_ids"].add(id_str)
            if pd.notna(category):
                tool_tool[key]["categories"].add(str(category))

if not tool_tool:
    raise ValueError("Nenhuma relação Tool x Tool encontrada após a normalização.")

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

# T: Graph onde cada aresta corresponde a um ID (uma ocorrência)
T = nx.Graph()
for (tool, tool2), attrs in tool_tool.items():
    tool_node = f"TOOL::{tool}"
    tool2_node = f"TOOL::{tool2}"

    # . Adicionar nós se não existirem
    if not T.has_node(tool_node):
        T.add_node(tool_node, bipartite=0, label=tool)
    if not T.has_node(tool2_node):
        T.add_node(tool2_node, bipartite=0, label=tool2)

    # . Adicionar aresta agregada se houver attack_ids
    if attrs["attack_ids"] is None or len(attrs["attack_ids"]) == 0:
        continue
    else:
        categories = attrs.get("categories")
        T.add_edge(
            tool_node,
            tool2_node,
            weight=len(attrs["attack_ids"]),
            attack_ids=attrs["attack_ids"],
            category=categories,
        )

print(
    "Grafo (TOOL - TOOL) criado | nós:",
    T.number_of_nodes(),
    "| arestas (multiedges):",
    T.number_of_edges(),
)

# Exportando nos e arestas
nodes_data = []
for n, d in T.nodes(data=True):
    nodes_data.append({"id": n, "label": d.get("label")})

edges_data = []
for u, v, d in T.edges(data=True):
    edges_data.append(
        {
            "source": u,
            "target": v,
            "weight": d.get("weight"),
            "attack_ids": d.get("attack_ids"),
            "category": d.get("category"),
        }
    )


edges_df = pd.DataFrame(edges_data)
edges_df.to_csv("./tool_attacks_edges.csv", index=False, encoding="utf-8")
print("Exportado:", "./tool_attacks_edges.csv", "| linhas:", len(edges_df))


nodes_df = pd.DataFrame(nodes_data)
nodes_df.to_csv("./tool_attacks_nodes.csv", index=False, encoding="utf-8")
print("Exportado:", "./tool_attacks_nodes.csv", "| nós:", len(nodes_df))
