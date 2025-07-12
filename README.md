Agentic Workflows for Graph RAG
================

### ODSC Agentic AI Summit 2025 Workshop

By GraphGeeks

----

:spiral_calendar:  July 16 - 31, 2025

:world_map:        Virtual

:writing_hand:     [AI Summit Track](https://www.summit.ai/#Tracks)

----

## All you really need to know

- Python
- How to use the terminal
- Basics of Docker

We'll be covering the basics of each of these in the workshop, so no worries if you're not that familiar with them!

## Instructors

* Amy Hodler
* David Hughes
* Prashanth Rao
* Dennis Irorere

## Dataset

This workshop uses a dataset of 2,726 FHIR records of patients and their notes. The dataset is obtained
from this [Hugging Face dataset](https://huggingface.co/datasets/kishanbodybrain/test-fhir/tree/main/data)
and preprocessed using the script `create_dataset.py`.

To create the dataset locally, run the following command:

```bash
uv run create_dataset.py
```

This creates the following two JSON files:

- Raw data: 2,726 notes in unstructured text format, output to `data/note.json`
- Evaluation data: 2,726 FHIR JSON records, output to `data/fhir.json`

## Setup Python environment

It's recommended to [install uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the dependencies.

```bash
uv sync
```
Alternatively, you can install the dependencies manually via pip.

```bash
pip install -r requirements.txt
```

## Components

### Information extraction

Our first goal is to extract entities and relationships that can form a knowledge graph from
the raw data (patient notes) in the `data/note.json` file. The information extraction pipeline
is powered by [BAML](https://www.boundaryml.com/), a programming language for obtaining high-quality
structured outputs from LLMs.

```bash
cd src
uv run baml_extract.py
```

### Store the graph in Kuzu

Kuzu is an embedded (in-process) graph database, so there is no server setup required! It's already included as a dependency in the `pyproject.toml` file, installed via `uv sync`.

```bash
cd src
uv run build_graph.py
```
The Kuzu graph is stored in the `fhir_kuzu_db` directory, and can be visualized using the Kuzu Explorer tool (see [below](#graph-visualization)).

### Graph RAG pipeline

Once the graph is created, we will build a Graph RAG pipeline (also powered by BAML) that uses the
graph to answer questions about the patient data.

See the [Graph RAG script](src/graphrag.py).

```bash
cd src
uv run graphrag.py
```

This runs a vanilla Graph RAG pipeline that uses the graph to answer questions about the patient data.

### Agents & workflow orchestration

Our next goal is to add agentic components to the Graph RAG system to improve the quality of the
answers and to make the system more robust. We will use a variety of tools to build, monitor, and
evaluate the agents.

### End-to-end evaluation

Evaluation consists of two parts:

1. Evaluating the quality of the graph construction
2. Evaluating the quality of the Graph RAG and agent pipeline

#### Graph construction evaluation

The evaluation data in `data/fhir.json` is used to evaluate the
quality of results from the information extraction pipeline in BAML.

See the [evaluation script](src/baml_extract_eval.py).

```bash
cd src
uv run baml_extract_eval.py
```

#### Graph, vector and FTS-based (hybrid) RAG and agent pipeline evaluation

We evaluate the RAG system that consists of a combination of graph, vector and FTS-based RAG
using the [Opik](https://www.comet.com/site/products/opik/) observability tool.

## Graph visualization

Once the knowledge graph has been created in Kuzu, you can visualize it using the
[Kuzu Explorer](https://docs.kuzudb.com/visualization/kuzu-explorer/#what-is-kuzu-explorer)tool.

Use the provided `docker-compose.yml` configured with the relative path to your data, and
start the Docker container for Kuzu explorer as follows:

```bash
docker compose up
```

Once finished, spin down the Kuzu Explorer container:

```bash
docker compose down
```

You can view multi-hop paths in the graph as follows:
```cypher
MATCH (a)-[r*1..4]-(b) RETURN * LIMIT 500;
```

![](./assets/fhir-graph-paths.png)