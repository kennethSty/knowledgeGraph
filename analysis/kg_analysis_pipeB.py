from utils.kg_utils import KnowledgeGraph
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

kg = KnowledgeGraph()

#Only outcomment this when you run the script the first time
#kg.query("""
#CALL gds.graph.project(
#  'my_graph',
#  '*',
#  '*')
#  """)

## BETWEENNESS CENTRALITY
#gds.util.asNode(nodeId).section_title, gds.util.asNode(nodeId).id, gds.util.asNode(nodeId).name .. page_title
btw_central_list_section = kg.query("""
CALL gds.betweenness.stream('my_graph')
YIELD nodeId, score
RETURN coalesce(gds.util.asNode(nodeId).section_title) AS node, score
ORDER BY score DESC;
""")

btw_central_list_node = kg.query("""
CALL gds.betweenness.stream('my_graph')
YIELD nodeId, score
RETURN coalesce(gds.util.asNode(nodeId).id) AS node, score
ORDER BY score DESC;
""")

btw_central_list_page = kg.query("""
CALL gds.betweenness.stream('my_graph')
YIELD nodeId, score
RETURN coalesce(gds.util.asNode(nodeId).page_title) AS node, score
ORDER BY score DESC;
""")

#metrics calc
#section
section_nodes = [node['node'] for node in btw_central_list_section if node['node']!=None]
section_scores = [node['score'] for node in btw_central_list_section if node['node']!=None]
section_short_nodes = section_nodes[0:10]
section_short_scores = section_scores[0:10]

#page
page_nodes = [node['node'] for node in btw_central_list_page if node['node']!=None]
page_scores = [node['score'] for node in btw_central_list_page if node['node']!=None]
page_short_nodes = page_nodes[0:10]
page_short_scores = page_scores[0:10]

#node
node_nodes = [node['node'] for node in btw_central_list_node if node['node']!=None]
node_scores = [node['score'] for node in btw_central_list_node if node['node']!=None]
node_short_nodes = node_nodes[0:10]
node_short_scores = node_scores[0:10]


#Plotting
sns.set(style="whitegrid", palette="viridis")
colors = sns.color_palette("viridis", 4)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))

axes.bar(node_short_nodes, node_short_scores, color = colors[2])
axes.set_ylabel('Betweenness Centrality', fontsize=18, color='white')
axes.set_xlabel('Nodes', fontsize=18, color='white')
axes.grid(alpha=0.6, linestyle='--')
axes.set_title("Betweenness Centrality of auto. generated nodes", color='white', fontsize=20)
axes.set_xticklabels(node_short_nodes, color='white', fontsize=20)
axes.set_yticklabels([round(node) for node in node_short_scores[::-1]], color='white', fontsize=20)
axes.tick_params(axis='x', rotation = 45)

plt.tight_layout()
plt.savefig('nodes_btw_centrality_pipeA.png', dpi=300, transparent=True)

fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
not_mesh_percentages = [0, 30, 0]
graphs = ["Graph A", "Graph B", "Graph C"]
axes2.bar(graphs, not_mesh_percentages, color = colors[2])
axes2.set_ylabel('% unmatched', fontsize=18, color='white')
axes2.set_xlabel('Nodes', fontsize=18, color='white')
axes2.grid(alpha=0.6, linestyle='--')
axes2.set_title("Unmatched nodes among nodes with highest btw. centrality score", color='white', fontsize=20)
axes2.set_xticklabels(graphs, color='white', fontsize=20)
axes2.tick_params(axis='x', rotation = 45)

plt.tight_layout()
plt.savefig('unmatched.png', dpi=300, transparent=True)

## AVG SHORTEST PATH LENGTH
avg_shortest_path_len = kg.query("""
CALL gds.allShortestPaths.stream('my_graph')
YIELD sourceNodeId, targetNodeId, distance
RETURN avg(distance) AS avgPathLength;
""")

max_shortest_path_len = kg.query("""
CALL gds.allShortestPaths.stream('my_graph')
YIELD sourceNodeId, targetNodeId, distance
RETURN max(distance) AS avgPathLength;
""")

shortest_paths = kg.query("""
CALL gds.allShortestPaths.stream('my_graph')
YIELD sourceNodeId, targetNodeId, distance
RETURN coalesce(gds.util.asNode(sourceNodeId).id, gds.util.asNode(sourceNodeId).section_title, gds.util.asNode(sourceNodeId).page_title) as node, coalesce(gds.util.asNode(targetNodeId).id, gds.util.asNode(targetNodeId).section_title, gds.util.asNode(targetNodeId).page_title) as target_node, distance
ORDER BY distance DESC
""")

shortest_paths_pages = kg.query("""
MATCH (p1:Page), (p2:Page)
WHERE id(p1) < id(p2)
MATCH path = shortestPath((p1)-[*]-(p2))
RETURN p1.page_title as node, p2.page_title as target_node, length(path) AS distance
""")

shortest_paths_nodes = kg.query("""
MATCH (n1), (n2)
WHERE NOT (n1:Page OR n1:Section OR n1:Category)
AND NOT (n2:Page OR n2:Section OR n2:Category)
AND id(n1) < id(n2)
MATCH path = shortestPath((n1)-[*]-(n2))
RETURN n1.id as node, n2.id as target_node, length(path) AS distance
""")

shortest_paths_sections = kg.query("""
MATCH (s1:Section), (s2:Section)
WHERE id(s1) < id(s2)
MATCH path = shortestPath((s1)-[*]-(s2))
RETURN s1.section_title as node, s2.section_title as target_node, length(path) AS distance
""")


#Calc shortest paths
node_pairs = [str(node['node'])+" ---- "+str(node['target_node'])  for node in shortest_paths if node['node']!=None and node['target_node']!=None]
distances = [node['distance'] for node in shortest_paths if node['node']!=None and node['target_node']!=None]
node_pairs_nodes = [str(node['node'])+" ---- "+str(node['target_node'])  for node in shortest_paths_nodes if node['node']!=None and node['target_node']!=None]
total_value_counts = Counter(distances)
distances_nodes = [node['distance'] for node in shortest_paths_nodes if node['node']!=None and node['target_node']!=None]
node_value_counts = Counter(distances_nodes)
node_pairs_sections = [str(node['node'])+" ---- "+str(node['target_node'])  for node in shortest_paths_sections if node['node']!=None and node['target_node']!=None]
distances_sections = [node['distance'] for node in shortest_paths_sections if node['node']!=None and node['target_node']!=None]
section_value_counts = Counter(distances_sections)
node_pairs_pages = [str(node['node'])+" ---- "+str(node['target_node'])  for node in shortest_paths_pages if node['node']!=None and node['target_node']!=None]
distances_pages = [node['distance'] for node in shortest_paths_pages if node['node']!=None and node['target_node']!=None]
page_value_counts = Counter(distances_pages)

#Plotting
colors = sns.color_palette("viridis", 4)
fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

axes[0][0].bar(list(total_value_counts.keys()), list(total_value_counts.values()), color = colors[0])
axes[0][0].set_ylabel('Frequency', fontsize=14, color='white')  # Adjust ylabel to 'Frequency'
axes[0][0].set_xlabel('Shortest path lengths btw. all nodes', fontsize=14, color='white')
axes[0][0].grid(alpha=0.6, linestyle='--')
axes[0][0].tick_params(axis='x', colors='white')
axes[0][0].tick_params(axis='y', colors='white')

axes[1][0].bar(list(node_value_counts.keys()), list(node_value_counts.values()), color = colors[1])
axes[1][0].set_ylabel('Frequency', fontsize=14, color='white')  # Adjust ylabel to 'Frequency'
axes[1][0].set_xlabel('Shortest path lengths btw. auto. gen. nodes', fontsize=18, color='white')
axes[1][0].grid(alpha=0.6, linestyle='--')
axes[1][0].tick_params(axis='x', colors='white')
axes[1][0].tick_params(axis='y', colors='white')

axes[1][1].bar(list(section_value_counts.keys()), list(section_value_counts.values()), color = colors[2])
axes[1][1].set_ylabel('Frequency', fontsize=14, color='white')  # Adjust ylabel to 'Frequency'
axes[1][1].set_xlabel('Shortest path lengths btw. section nodes', fontsize=18, color='white')
axes[1][1].grid(alpha=0.6, linestyle='--')
axes[1][1].tick_params(axis='x', colors='white')
axes[1][1].tick_params(axis='y', colors='white')

axes[0][1].barh(node_pairs_pages, distances_pages, color = colors)
axes[0][1].set_ylabel('Page Pairs', fontsize=14, color='white')  # Adjust ylabel to 'Frequency'
axes[0][1].set_xlabel('Shortest path lengths btw. page nodes', fontsize=18, color='white')
axes[0][1].grid(alpha=0.6, linestyle='--')
axes[0][1].tick_params(axis='x', colors='white')
axes[0][1].tick_params(axis='y', colors='white')

plt.tight_layout()
plt.savefig('shortest_paths_pipeA.png', dpi=300, transparent=True)


m = kg.query("""
    MATCH ()-[r]->()
    RETURN count(r) AS numberOfEdges""")

n = kg.query("""
    MATCH (n)
    RETURN count(n) AS value""")

m_auto_only = kg.query("""
    MATCH (a)-[r]->(b)
    WHERE NOT a:Page AND NOT a:Section AND NOT a:Category
      AND NOT b:Page AND NOT b:Section AND NOT b:Category
    RETURN count(r) AS numberOfEdges
""")

print(f"Avg. shortest path length total: {avg_shortest_path_len}")
print(f"Avg. shortest path length auto nodes: {sum(distances_nodes)/len(distances_nodes)}")
print(f"Avg. shortest path length sections: {sum(distances_sections)/len(distances_sections)}")
print(f"Max shortest path length total: {max_shortest_path_len}")
print(f"Number edges:{m}")
print(f"Number nodes:{n}")
print(f"Number of edges between auto generated nodes: {m_auto_only}")
