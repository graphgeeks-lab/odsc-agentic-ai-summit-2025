"""
Utility functions for the Graph RAG pipeline.
"""

import kuzu

from baml_client import b

# --- Database ---


class KuzuDatabaseManager:
    """Manages Kuzu database connection and schema retrieval."""

    def __init__(self, db_path: str = "ex_kuzu_db"):
        self.db_path = db_path
        self.db = kuzu.Database(db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)
        self._setup_vector_extension()

    def _setup_vector_extension(self):
        """Install and load vector extension once."""
        self.conn.execute("INSTALL vector; LOAD vector;")

    def get_connection(self) -> kuzu.Connection:
        """Get the database connection."""
        return self.conn

    def close(self):
        """Close the database connection."""
        if hasattr(self, "conn"):
            self.conn.close()
        if hasattr(self, "db"):
            self.db.close()

    @property
    def get_schema_dict(self) -> dict[str, list[dict]]:
        # Get schema for LLM
        nodes = self.conn._get_node_table_names()
        relationships = self.conn._get_rel_table_names()

        schema = {"nodes": [], "edges": []}

        for node in nodes:
            node_schema = {"label": node, "properties": []}
            node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
            while node_properties.has_next():  # type: ignore
                row = node_properties.get_next()  # type: ignore
                node_schema["properties"].append({"name": row[1], "type": row[2]})
            schema["nodes"].append(node_schema)

        for rel in relationships:
            edge = {
                "label": rel["name"],
                "src": rel["src"],
                "dst": rel["dst"],
                "properties": [],
            }
            rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
            while rel_properties.has_next():  # type: ignore
                row = rel_properties.get_next()  # type: ignore
                edge["properties"].append({"name": row[1], "type": row[2]})
            schema["edges"].append(edge)

        return schema

    def get_schema_xml(self, schema: dict[str, list[dict]]) -> str:
        """Convert the JSON schema into XML format with structure, nodes, and relationships."""
        # Structure section: just relationship structure
        structure_lines = []
        for edge in schema["edges"]:
            structure_lines.append(
                f'  <rel label="{edge["label"]}" from="{edge["src"]}" to="{edge["dst"]}" />'
            )
        structure_xml = "<structure>\n" + "\n".join(structure_lines) + "\n</structure>"

        # Nodes section: node label and properties
        node_lines = []
        for node in schema["nodes"]:
            prop_lines = [
                f'    <property name="{prop["name"]}" type="{prop["type"]}" />'
                for prop in node["properties"]
            ]
            node_lines.append(
                f'  <node label="{node["label"]}">\n' + "\n".join(prop_lines) + "\n  </node>"
            )
        nodes_xml = "<nodes>\n" + "\n".join(node_lines) + "\n</nodes>"

        # Relationships section: edge label and properties (if any)
        rel_lines = []
        for edge in schema["edges"]:
            if edge.get("properties"):
                prop_lines = [
                    f'    <property name="{prop["name"]}" type="{prop["type"]}" />'
                    for prop in edge["properties"]
                ]
                rel_lines.append(
                    f'  <rel label="{edge["label"]}">\n' + "\n".join(prop_lines) + "\n  </rel>"
                )
            else:
                rel_lines.append(f'  <rel label="{edge["label"]}" />')
        relationships_xml = "<relationships>\n" + "\n".join(rel_lines) + "\n</relationships>"

        # Combine all sections
        return f"{structure_xml}\n{nodes_xml}\n{relationships_xml}"
