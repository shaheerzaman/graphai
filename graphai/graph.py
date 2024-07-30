from typing import Callable, List
from graphai.nodes.base import _Node


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
                raise Exception(
                    "Multiple start nodes are not allowed. Start node "
                    f"'{self.start_node.name}' already exists, so new start "
                    f"node '{node.name}' can not be added to the graph."
                )
            self.start_node = node
        if node.is_end:
            self.end_nodes.append(node)

    def add_edge(self, source: _Node, destination: _Node):
        # TODO add logic to check that source and destination are nodes
        # and they exist in the graph object already
        edge = Edge(source, destination)
        self.edges.append(edge)

    def add_router(self, source: _Node, router: _Node, destinations: List[_Node]):
        if not router.is_router:
            raise TypeError("A router object must be passed to the router parameter.")
        self.add_edge(source, router)
        for destination in destinations:
            self.add_edge(router, destination)

    def set_start_node(self, node: _Node):
        self.start_node = node

    def set_end_node(self, node: _Node):
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
        state = input
        while current_node not in self.end_nodes:
            state = current_node.invoke(input=state)
            if current_node.is_router:
                # if we have a router node we let the router decide the next node
                next_node_name = str(state["choice"])
                del state["choice"]
                current_node = self._get_node_by_name(next_node_name)
            else:
                # otherwise, we have linear path
                current_node = self._get_next_node(current_node, state)
            if current_node.is_end:
                break
        return state

    def _get_node_by_name(self, node_name: str) -> _Node:
        for node in self.nodes:
            if node.name == node_name:
                return node
        raise Exception(f"Node with name {node_name} not found.")

    def _get_next_node(self, current_node, output):
        for edge in self.edges:
            if edge.source == current_node:
                return edge.destination
        raise Exception(
            f"No outgoing edge found for current node '{current_node.name}'."
        )


class Edge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination