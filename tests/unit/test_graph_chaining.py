"""
Test method chaining capabilities in Graph class.
"""

import pytest
from graphai import Graph
from graphai.nodes import node


class TestGraphChaining:
    """Test cases for Graph method chaining."""

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes for testing."""
        @node(start=True, name="start")
        async def start_node(input: str):
            """Start node"""
            return {"output": f"Started with: {input}"}

        @node(name="process")
        async def process_node(output: str):
            """Process node"""
            return {"result": f"Processed: {output}"}

        @node(end=True, name="end")
        async def end_node(result: str):
            """End node"""
            return {"final": f"Completed: {result}"}

        return start_node, process_node, end_node

    def test_basic_chaining(self, sample_nodes):
        """Test basic method chaining for add_node and add_edge."""
        start_node, process_node, end_node = sample_nodes
        
        graph = (
            Graph()
            .add_node(start_node)
            .add_node(process_node)
            .add_node(end_node)
            .add_edge(start_node, process_node)
            .add_edge(process_node, end_node)
        )
        
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.start_node == start_node
        assert end_node in graph.end_nodes

    def test_chaining_with_string_references(self, sample_nodes):
        """Test chaining with string node references in add_edge."""
        start_node, process_node, end_node = sample_nodes
        
        graph = (
            Graph()
            .add_node(start_node)
            .add_node(process_node)
            .add_node(end_node)
            .add_edge("start", "process")
            .add_edge("process", "end")
        )
        
        assert len(graph.edges) == 2
        assert graph.edges[0].source.name == "start"
        assert graph.edges[0].destination.name == "process"
        assert graph.edges[1].source.name == "process"
        assert graph.edges[1].destination.name == "end"

    def test_state_management_chaining(self):
        """Test chaining of state management methods."""
        initial_state = {"counter": 0}
        
        graph = (
            Graph(initial_state=initial_state)
            .update_state({"counter": 1, "name": "test"})
            .update_state({"counter": 2})
        )
        
        assert graph.get_state() == {"counter": 2, "name": "test"}

    def test_set_state_chaining(self):
        """Test set_state method chaining."""
        graph = (
            Graph(initial_state={"old": "data"})
            .set_state({"new": "data"})
            .update_state({"extra": "info"})
        )
        
        assert graph.get_state() == {"new": "data", "extra": "info"}

    def test_reset_state_chaining(self):
        """Test reset_state method chaining."""
        graph = (
            Graph(initial_state={"some": "data"})
            .update_state({"more": "data"})
            .reset_state()
            .update_state({"fresh": "start"})
        )
        
        assert graph.get_state() == {"fresh": "start"}

    def test_compile_chaining(self, sample_nodes):
        """Test compile method chaining."""
        start_node, process_node, end_node = sample_nodes
        
        graph = (
            Graph()
            .add_node(start_node)
            .add_node(process_node)
            .add_node(end_node)
            .add_edge(start_node, process_node)
            .add_edge(process_node, end_node)
            .compile()
        )
        
        # If we get here without exception, compile worked
        assert graph.start_node == start_node
        assert end_node in graph.end_nodes

    def test_set_start_end_node_chaining(self, sample_nodes):
        """Test set_start_node and set_end_node chaining."""
        start_node, process_node, end_node = sample_nodes
        
        # Add nodes without start/end flags
        @node(name="alt_start")
        async def alt_start():
            return {}
        
        @node(name="alt_end")
        async def alt_end():
            return {}
        
        graph = (
            Graph()
            .add_node(alt_start)
            .add_node(alt_end)
            .set_start_node(alt_start)
            .set_end_node(alt_end)
        )
        
        assert graph.start_node == alt_start
        assert graph.end_node == alt_end

    def test_complex_chaining_scenario(self, sample_nodes):
        """Test a complex chaining scenario using multiple methods."""
        start_node, process_node, end_node = sample_nodes
        
        graph = (
            Graph()
            .set_state({"initial": True})
            .add_node(start_node)
            .add_node(process_node)
            .add_node(end_node)
            .add_edge(start_node, process_node)
            .add_edge(process_node, end_node)
            .update_state({"step": 1})
            .compile()
            .update_state({"compiled": True})
        )
        
        assert graph.get_state() == {"initial": True, "step": 1, "compiled": True}
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_chaining_returns_same_instance(self, sample_nodes):
        """Test that chaining methods return the same Graph instance."""
        start_node, _, _ = sample_nodes
        
        graph = Graph()
        result1 = graph.add_node(start_node)
        result2 = result1.update_state({"test": True})
        result3 = result2.reset_state()
        
        # All results should be the same instance
        assert graph is result1
        assert result1 is result2
        assert result2 is result3

    def test_router_chaining(self, sample_nodes):
        """Test add_router method chaining."""
        start_node, _, end_node = sample_nodes
        
        from graphai.nodes import router
        
        @router(name="router")
        async def router_node(output: str):
            return {"route": "end"}
        
        @node(name="alt_end", end=True)
        async def alt_end_node(output: str):
            return {"final": "alt"}
        
        graph = (
            Graph()
            .add_node(start_node)
            .add_node(router_node)
            .add_node(end_node)
            .add_node(alt_end_node)
            .add_router(
                sources=[start_node],
                router=router_node,
                destinations=[end_node, alt_end_node]
            )
        )
        
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 3  # start->router, router->end, router->alt_end

    def test_chaining_with_exceptions(self, sample_nodes):
        """Test that exceptions don't break the chain unexpectedly."""
        start_node, _, _ = sample_nodes
        
        graph = Graph().add_node(start_node)
        
        # This should raise an exception (node already exists)
        with pytest.raises(Exception, match="already exists"):
            graph.add_node(start_node).update_state({"test": True})
        
        # Graph should still be usable after exception
        assert len(graph.nodes) == 1
        
    @pytest.mark.asyncio
    async def test_chained_graph_execution(self, sample_nodes):
        """Test that a chained graph can be executed successfully."""
        start_node, process_node, end_node = sample_nodes
        
        graph = (
            Graph()
            .add_node(start_node)
            .add_node(process_node)
            .add_node(end_node)
            .add_edge(start_node, process_node)
            .add_edge(process_node, end_node)
            .compile()
        )
        
        result = await graph.execute({"input": "test"})
        assert "final" in result
        assert "Completed" in result["final"]