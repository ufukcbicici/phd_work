import networkx as nx


class Dag:
    def __init__(self):
        self.dagObject = nx.DiGraph()

    def parents(self, node):
        parents = list(self.dagObject.predecessors(node))
        return parents

    def children(self, node):
        children = list(self.dagObject.successors(node))
        return children

    def ancestors(self, node):
        ancestors = list(nx.ancestors(self.dagObject, node))
        return ancestors

    def descendants(self, node):
        descendants = list(nx.descendants(self.dagObject, node))
        return descendants

    def get_leaves(self):
        leaf_nodes = [node for node in self.dagObject.nodes() if
                      self.dagObject.in_degree(node) != 0 and self.dagObject.out_degree(node) == 0]
        return leaf_nodes

    def remove_node(self, node):
        self.dagObject.remove_node(node)

    def add_node(self, node):
        self.dagObject.add_node(node)

    def add_edge(self, parent, child):
        self.dagObject.add_edge(parent, child)

    def get_topological_sort(self):
        return list(nx.topological_sort(self.dagObject))

    def get_shortest_path(self, source, dest):
        return nx.shortest_path(self.dagObject, source, dest)

    def get_shortest_path_length(self, source, dest):
        return len(nx.shortest_path(self.dagObject, source, dest)) - 1
