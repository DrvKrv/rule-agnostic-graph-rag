import networkx as nx
from models import GraphPayload, TraversalInstruction

class GovernanceGraphEngine:
    def __init__(self):
        self.graph = nx.DiGraph()

    def load_graph_from_payload(self, payload: GraphPayload):
        self.graph.clear()
        
        for node in payload.nodes:
            self.graph.add_node(node.id, **node.model_dump())
        
        for edge in payload.edges:
            self.graph.add_edge(
                edge.source, 
                edge.target, 
                **edge.model_dump(exclude={"source", "target"})
            )

    def detect_cycles(self) -> Optional[list]:
        try:
            return nx.find_cycle(self.graph, orientation="original")
        except nx.NetworkXNoCycle:
            return None

    def execute_rule_agnostic_cascade(self, instructions: TraversalInstruction) -> dict:
        # Implement rule-agnostic recursive traversal math
        pass