import pandas as pd
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import defaultdict
from networkx.algorithms import bipartite
from networkx.algorithms.community import louvain_communities
from metricas import plot_degree_distribution_complementar

# Caminho do dataset (obtido de: https://www.kaggle.com/datasets/tannubarot/cybersecurity-attack-and-defence-dataset/data)
CSV_PATH = os.path.join(os.path.dirname(__file__), "./Attack_Dataset.csv")

# Determinando padrões de extração e funções auxiliares
MITRE_REGEX = re.compile(r"(T\d{4}(?:\.\d{1,3})?)", flags=re.IGNORECASE)
SPLIT_REGEX = re.compile(r"[;,/|\\\n]+|\s-\s|\s—\s")


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
    # Removendo emojis e caracteres especiais
    emoji_re = re.compile(r"[^\w\s,\.\-\(\)]", flags=re.UNICODE)
    cleaned = emoji_re.sub("", raw)

    parts = [p.strip() for p in SPLIT_REGEX.split(cleaned) if p and p.strip()]
    parts = [" ".join(p.lower().split()) for p in parts]
    parts = [p.strip() for p in parts if p]
    return sorted(set(parts))


# Funções auxiliares para análise de similaridade de Jaccard
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2)) if s1 or s2 else 0


def generate_jaccard_matrix(comm_list_b, comm_list_gcc):
    print("\n" + "=" * 50)
    print("Jaccard Similarity Matrix (Top 5 B vs Top 5 GCC)")
    print("=" * 50)

    top_b = comm_list_b[:5]
    top_gcc = comm_list_gcc[:5]

    matrix = []

    for cb in top_b:
        row = []
        for cgcc in top_gcc:
            row.append(jaccard_similarity(cb, cgcc))
        matrix.append(row)

    df = pd.DataFrame(
        matrix,
        index=[f"B_G{i+1}" for i in range(len(top_b))],
        columns=[f"GCC_G{i+1}" for i in range(len(top_gcc))],
    )
    print(df)


def main():
    print("Lendo dataset...")
    if not os.path.exists(CSV_PATH):
        print(f"Erro: Dataset não encontrado em {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # Mineração
    print("Minerando...")
    df["mitre_codes"] = df["MITRE Technique"].apply(extract_mitre_codes)
    df["tools"] = df["Tools Used"].apply(split_tools)

    # Construindo lista de arestas bipartidas e estruturas auxiliares
    tool_mitre_weights = (
        Counter()
    )  # Contador para quantas vezes uma ferramenta resolve uma técnica

    for _, row in df.iterrows():
        tools = row.get("tools") or []
        mitres = row.get("mitre_codes") or []

        for t in tools:
            for m in mitres:
                tool_mitre_weights[(t, m)] += 1

    # Inicializando rede
    B = nx.Graph()  # Bipartido: Tool <-> Technique

    print("Construindo rede...")
    for (t, m), count in tool_mitre_weights.items():
        B.add_node(t, bipartite=0)
        B.add_node(m, bipartite=1)
        B.add_edge(t, m, weight=count)

    tools_nodes = []
    mitre_nodes = []

    for node in B.nodes():
        if B.nodes[node].get("bipartite") == 0:
            tools_nodes.append(node)
        elif B.nodes[node].get("bipartite") == 1:
            mitre_nodes.append(node)

    # Métricas essenciais
    print("* Rede Bipartida - Métricas Básicas: *")
    print(f"Nós: {B.number_of_nodes()}")
    print(f"Arestas: {B.number_of_edges()}")
    print(f"Nós de ferramentas: {len(tools_nodes)}")
    print(f"Nós de técnicas MITRE: {len(mitre_nodes)}")

    # * Density
    density = nx.density(B)
    density_tools = bipartite.density(B, tools_nodes)
    density_mitre = bipartite.density(B, mitre_nodes)
    print(f"Densidade: {density:.6f}")
    print(f"Densidade bipartida (ferramentas): {density_tools:.6f}")
    print(f"Densidade bipartida (MITRE): {density_mitre:.6f}")

    # * Grau médio
    degrees = []
    degrees_tools = []
    degrees_mitre = []
    for node in B.nodes():
        degrees.append(nx.degree(B, node))
        if B.nodes[node].get("bipartite") == 0:
            degrees_tools.append(nx.degree(B, node))
        elif B.nodes[node].get("bipartite") == 1:
            degrees_mitre.append(nx.degree(B, node))

    avg_degree = np.mean(degrees)
    avg_degree_tools = np.mean(degrees_tools)
    avg_degree_mitre = np.mean(degrees_mitre)

    print(f"Grau médio: {avg_degree:.6f}")
    print(f"Grau médio ferramentas: {avg_degree_tools:.6f}")
    print(f"Grau médio mitre: {avg_degree_mitre:.6f}")

    plot_degree_distribution_complementar(
        B,
        titles=["Tool-MITRE Bipartido (Bipartite)"],
        output_path="./",
        output_file_name="tool_bipartite_degree_distribution_bi.png",
        show=False,
    )

    """# * Plotando rede
    plt.figure(figsize=(12, 10))

    # Definindo layout "spring"
    pos = nx.spring_layout(B, seed=42, weight="weight")

    # Separando nós por partição
    tools = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    mitre = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]

    # Desenhando arestas
    nx.draw_networkx_edges(B, pos, alpha=0.15, width=0.5)

    # Desenhando nós (cores diferentes por partição)
    nx.draw_networkx_nodes(
        B, pos, nodelist=tools, node_size=20, node_color="tab:blue", alpha=0.8
    )

    nx.draw_networkx_nodes(
        B, pos, nodelist=mitre, node_size=20, node_color="tab:orange", alpha=0.8
    )

    # Definindo não aparição de labels
    plt.axis("off")

    # Salvando figura
    plt.tight_layout()
    plt.savefig("bipartite_graph_no_labels.png", dpi=300)
    plt.close()"""

    # -------------------------------------------------------------------------
    # Quais coocorências de ferramentas e técnicas que são mais utilizadas?
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais coocorências de ferramentas e técnicas que são mais utilizadas?")
    print("=" * 50)
    # Ordenando arestas por peso decrescente
    sorted_b_edges = sorted(
        B.edges(data=True), key=lambda x: x[2]["weight"], reverse=True
    )
    for u, v, data in sorted_b_edges[:10]:
        n1_type = B.nodes[u].get("bipartite")
        tool = u if n1_type == 0 else v
        mitre = v if n1_type == 0 else u
        print(
            f"Ferramenta: {tool:<30} | MITRE: {mitre:<10} | Incidentes: {data['weight']}"
        )

    # -------------------------------------------------------------------------
    # Quais são as ferramentas que mais atacam? (Peso das arestas dos nós de ferramentas)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais são as ferramentas que mais atacam?")
    print("=" * 50)
    tool_attack_counts = defaultdict(int)
    for u, v, data in B.edges(data=True):
        if B.nodes[u].get("bipartite") == 0:
            tool_attack_counts[u] += data["weight"]
        elif B.nodes[v].get("bipartite") == 0:
            tool_attack_counts[v] += data["weight"]
    sorted_tool_attacks = sorted(
        tool_attack_counts.items(), key=lambda x: x[1], reverse=True
    )
    for tool, total_attacks in sorted_tool_attacks[:10]:
        print(f"Ferramenta: {tool:<30} | Total Incidentes: {total_attacks}")

    # -------------------------------------------------------------------------
    # Quais são as ferramentas mais generalistas em relação a ataques? (Centralidade de grau)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais são as ferramentas mais generalistas em relação a ataques?")
    print("=" * 50)

    # Centralidade de grau
    degree_dict = nx.degree_centrality(B)

    print("\n--- Top 10 de Ferramentas (Centralidade de Grau) ---")
    count_tools = 0
    for node in sorted(degree_dict, key=degree_dict.get, reverse=True):
        if count_tools == 10:
            break
        if B.nodes[node].get("bipartite") == 0:
            print(f"Ferramenta: {node} | Centralidade: {degree_dict[node]:.6f}")
            count_tools += 1

    # -------------------------------------------------------------------------
    # Quais são os grupos de ferramentas e técnicas que mais coocorrem juntas? (Detecção de comunidades)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais são os grupos de ferramentas e técnicas que mais coocorrem juntas?")
    print("=" * 50)
    # Utilizando Louvain para detecção de comunidades
    communities = louvain_communities(B, weight="weight")

    # Ordenando comunidades por tamanho decrescente
    comm_list_B = sorted(communities, key=len, reverse=True)

    # Quantas comunidades foram encontradas
    print(f"\nTotal de comunidades encontradas em B: {len(comm_list_B)}")

    # Top 5 maiores comunidades
    i = 0
    for comm in comm_list_B:
        if i >= 5:
            break

        subB = B.subgraph(comm)
        degrees = dict(subB.degree(weight="weight"))

        tools = []
        mitres = []
        tool_count = 0
        mitre_count = 0
        for node in comm:
            if B.nodes[node].get("bipartite") == 0:
                tools.append((node, degrees[node]))
                tool_count += 1
            if B.nodes[node].get("bipartite") == 1:
                mitres.append((node, degrees[node]))
                mitre_count += 1

        print(
            f"\nComunidade {i+1} | Ferramentas: {tool_count} | MITRE: {mitre_count} | Tamanho: {len(comm)}"
        )

        # Mostrar as top 10 ferramentas e técnicas MITRE nesta comunidade por grau local

        # Ordenando decrescentemente por grau
        top_tools = sorted(tools, key=lambda x: x[1], reverse=True)[:10]
        top_mitres = sorted(mitres, key=lambda x: x[1], reverse=True)[:10]

        if top_tools:
            print("  Top 10 Ferramentas:")
            for node, deg in top_tools:
                print(f"    {node}: {deg:.2f}")

        if top_mitres:
            print("  Top 10 Técnicas MITRE:")
            for node, deg in top_mitres:
                print(f"    {node}: {deg:.2f}")

        i += 1

    # Obtendo a GCC para análises adicionais
    print("\n!" + "=" * 50)
    print("Análises sobre a GCC (Giant Connected Component)")
    print("=" * 50 + "!")
    gcc_graph = B.subgraph(max(nx.connected_components(B), key=len)).copy()

    tools_nodes_gcc = []
    mitre_nodes_gcc = []

    for node in gcc_graph.nodes():
        if B.nodes[node].get("bipartite") == 0:
            tools_nodes_gcc.append(node)
        elif B.nodes[node].get("bipartite") == 1:
            mitre_nodes_gcc.append(node)

    # Basic stats
    print(f"Nós na GCC: {gcc_graph.number_of_nodes()}")
    print(f"Nós ferramentas na GCC: {len(tools_nodes_gcc)}")
    print(f"Nós técnicas MITRE na GCC: {len(mitre_nodes_gcc)}")
    print(f"Arestas na GCC: {gcc_graph.number_of_edges()}")
    print(
        f"Porcentagem de nós da GCC presentes em B: {(gcc_graph.number_of_nodes() / B.number_of_nodes()) * 100:.2f}%"
    )

    # * Densidade
    density = nx.density(gcc_graph)
    density_tools = bipartite.density(gcc_graph, tools_nodes_gcc)
    density_mitre = bipartite.density(gcc_graph, mitre_nodes_gcc)
    print(f"Densidade: {density:.6f}")
    print(f"Densidade bipartida (ferramentas): {density_tools:.6f}")
    print(f"Densidade bipartida (MITRE): {density_mitre:.6f}")

    # * Grau médio
    degrees = []
    degrees_tools = []
    degrees_mitre = []
    for node in gcc_graph.nodes():
        degrees.append(nx.degree(gcc_graph, node))
        if gcc_graph.nodes[node].get("bipartite") == 0:
            degrees_tools.append(nx.degree(gcc_graph, node))
        elif gcc_graph.nodes[node].get("bipartite") == 1:
            degrees_mitre.append(nx.degree(gcc_graph, node))

    avg_degree = np.mean(degrees)
    avg_degree_tools = np.mean(degrees_tools)
    avg_degree_mitre = np.mean(degrees_mitre)

    print(f"Grau médio: {avg_degree:.6f}")
    print(f"Grau médio ferramentas: {avg_degree_tools:.6f}")
    print(f"Grau médio MITRE: {avg_degree_mitre:.6f}")

    plot_degree_distribution_complementar(
        B,
        titles=["Tool-MITRE Bipartido (Bipartite) - GCC"],
        output_path="./",
        output_file_name="tool_bipartite_degree_distribution_bi_gcc.png",
        show=False,
    )

    # -------------------------------------------------------------------------
    # Quais coocorências de ferramentas e técnicas que são mais utilizadas?
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais coocorências de ferramentas e técnicas que são mais utilizadas?")
    print("=" * 50)

    sorted_gcc_edges = sorted(
        gcc_graph.edges(data=True), key=lambda x: x[2]["weight"], reverse=True
    )
    for u, v, data in sorted_gcc_edges[:10]:
        n1_type = gcc_graph.nodes[u].get("bipartite")
        tool = u if n1_type == 0 else v
        mitre = v if n1_type == 0 else u
        print(
            f"Ferramenta: {tool:<30} | MITRE: {mitre:<10} | Incidentes: {data['weight']}"
        )

    # -------------------------------------------------------------------------
    # Quais são as ferramentas que mais atacam? (Peso das arestas dos nós de ferramentas)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais são as ferramentas que mais atacam?")
    print("=" * 50)
    tool_attack_counts_gcc = defaultdict(int)
    for u, v, data in gcc_graph.edges(data=True):
        if gcc_graph.nodes[u].get("bipartite") == 0:
            tool_attack_counts_gcc[u] += data["weight"]
        elif gcc_graph.nodes[v].get("bipartite") == 0:
            tool_attack_counts_gcc[v] += data["weight"]
    sorted_tool_attacks_gcc = sorted(
        tool_attack_counts_gcc.items(), key=lambda x: x[1], reverse=True
    )
    for tool, total_attacks in sorted_tool_attacks_gcc[:10]:
        print(f"Ferramenta: {tool:<30} | Total Incidentes: {total_attacks}")

    # -------------------------------------------------------------------------
    # Quais são as ferramentas mais generalistas em relação a ataques? (Centralidade de grau)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais são as ferramentas mais generalistas em relação a ataques?")
    print("=" * 50)

    degree_dict = nx.degree_centrality(gcc_graph)

    print("\n--- Top 10 de Ferramentas (Centralidade de Grau) ---")
    count_tools = 0
    for node in sorted(degree_dict, key=degree_dict.get, reverse=True):
        if count_tools == 10:
            break
        if gcc_graph.nodes[node].get("bipartite") == 0:
            print(f"Ferramenta: {node} | Centralidade: {degree_dict[node]:.6f}")
            count_tools += 1

    # -------------------------------------------------------------------------
    # Quais são os grupos de ferramentas e técnicas que mais coocorrem juntas? (Detecção de comunidades)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Quais são os grupos de ferramentas e técnicas que mais coocorrem juntas?")
    print("=" * 50)

    communities = louvain_communities(gcc_graph, weight="weight")

    comm_list_GCC = sorted(communities, key=len, reverse=True)

    print(f"\nTotal de comunidades encontradas na GCC: {len(comm_list_GCC)}")

    i = 0
    for comm in comm_list_GCC:
        if i >= 5:
            break

        subGCC = gcc_graph.subgraph(comm)
        degrees_comm_gcc = dict(subGCC.degree(weight="weight"))

        tools_gcc = []
        mitres_gcc = []
        tool_count_gcc = 0
        mitre_count_gcc = 0
        for node in comm:
            if B.nodes[node].get("bipartite") == 0:
                tools_gcc.append((node, degrees_comm_gcc[node]))
                tool_count_gcc += 1
            if B.nodes[node].get("bipartite") == 1:
                mitres_gcc.append((node, degrees_comm_gcc[node]))
                mitre_count_gcc += 1

        print(
            f"\nComunidade {i+1} | Ferramentas: {len(tools_gcc)} | MITRE: {len(mitres_gcc)} | Tamanho: {len(comm)}"
        )

        # Mostrar as top 10 ferramentas e técnicas MITRE nesta comunidade por grau local

        # Ordenando decrescentemente por grau
        top_tools_gcc = sorted(tools_gcc, key=lambda x: x[1], reverse=True)[:10]
        top_mitres_gcc = sorted(mitres_gcc, key=lambda x: x[1], reverse=True)[:10]

        if top_tools_gcc:
            print("  Top 10 Ferramentas:")
            for node, deg in top_tools_gcc:
                print(f"    {node}: {deg:.2f}")

        if top_mitres_gcc:
            print("  Top 10 Técnicas MITRE:")
            for node, deg in top_mitres_gcc:
                print(f"    {node}: {deg:.2f}")

        i += 1

    # Gerando matriz de similaridade de Jaccard para as top 5 comunidades B e comunidades GCC
    generate_jaccard_matrix(comm_list_B, comm_list_GCC)


if __name__ == "__main__":
    main()
