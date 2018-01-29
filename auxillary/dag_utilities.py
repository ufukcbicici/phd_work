import networkx as nx


class Dag:
    def __init__(self):
        self.dag_object = nx.DiGraph()

    def parents(self, node):
        parents = list(self.dag_object.predecessors(node))
        return parents

    def children(self, node):
        children = list(self.dag_object.successors(node))
        return children

    def ancestors(self, node):
        ancestors = list(nx.ancestors(self.dag_object, node))
        return ancestors

    def descendants(self, node):
        descendants = list(nx.descendants(self.dag_object, node))
        return descendants

    def get_leaves(self):
        leaf_nodes = [node for node in self.dag_object.nodes() if
                      self.dag_object.in_degree(node) != 0 and self.dag_object.out_degree(node) == 0]
        return leaf_nodes

    def remove_node(self, node):
        self.dag_object.remove_node(node)

    def add_node(self, node):
        self.dag_object.add_node(node)

    def add_edge(self, parent, child):
        self.dag_object.add_edge(parent, child)

    def get_topological_sort(self):
        return list(nx.topological_sort(self.dag_object))

    def get_shortest_path(self, source, dest):
        return nx.shortest_path(self.dag_object, source, dest)

    def get_shortest_path_length(self, source, dest):
        return len(nx.shortest_path(self.dag_object, source, dest)) - 1
