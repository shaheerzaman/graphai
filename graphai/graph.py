class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.start_node = None
        self.end_nodes = []

    def add_node(self, node):
        self.nodes.append(node)
        if node.is_start:
            if self.start_node is not None:
                raise Exception("Multiple start nodes are not allowed.")
            self.start_node = node
        if node.is_end:
            self.end_nodes.append(node)

    def add_edge(self, source, destination):
        # TODO add logic to check that source and destination are nodes
        # and they exist in the graph object already
        edge = Edge(source, destination)
        self.edges.append(edge)

    def set_start_node(self, node):
        self.start_node = node

    def set_end_node(self, node):
        self.end_node = node

    def compile(self):
        if not self.start_node:
            raise Exception("Start node not defined.")
        if not self.end_nodes:
            raise Exception("No end nodes defined.")
        if not self._is_valid():
            raise Exception("Graph is not valid.")
        print("Graph compiled successfully.")

    def _is_valid(self):
        # Implement validation logic, e.g., checking for cycles, disconnected components, etc.
        return True

    def execute(self, input):
        current_node = self.start_node
        while current_node not in self.end_nodes:
            output = current_node.invoke(input=input)
            current_node = self._get_next_node(current_node, output)
            if current_node.is_end:
                break
        return output

    def _get_next_node(self, current_node, output):
        for edge in self.edges:
            if edge.source == current_node:
                return edge.destination
        raise Exception("No outgoing edge found for current node.")


class Edge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination