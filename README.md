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

- Basics of Python
- Basic familiarity with vector search
- How to use the terminal and modern IDEs (like VS Code or Cursor)

### Tips to get started

After you've attended the sessions, we recommend cloning this repo and following our [tips to get started](./TIPS.md).

## Instructors

This 3-part workshop is led by a team of 4 instructors.

* [Amy Hodler](https://www.linkedin.com/in/amyhodler)
* [Dennis Irorere](https://www.linkedin.com/in/dennis-irorere)
* [Prashanth Rao](https://www.linkedin.com/in/prrao87)
* [David Hughes](https://www.linkedin.com/in/dahugh)

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

### 1. Information extraction

Our first goal is to extract entities and relationships that can form a knowledge graph from
the raw data (patient notes) in the `data/note.json` file. The information extraction pipeline
is powered by [BAML](https://www.boundaryml.com/), a programming language for obtaining high-quality
structured outputs from LLMs.

### 2. Store the graph in Kuzu

[Kuzu](https://kuzudb.com/) is an embedded (in-process) graph database that we use to persist the
graph and query it using Cypher.

### 3. Store embeddings of the notes in LanceDB

We'll use [LanceDB](https://lancedb.com/) to store the embeddings of the notes. A vector index
and a full-text search (FTS) index are created, so that we can run a hybrid search (vector + FTS)
search on the data.

## Evaluation

There are two parts to the evaluation on our system:

1. Evaluating the quality of the structured data extraction
2. Evaluating the quality of the Graph RAG and agent pipeline

#### 1. Structured data evaluation

The evaluation of the structured data in `data/fhir.json` is used to gauge the
quality of the structured information extraction from BAML.

See the [evaluation script](src/baml_extract_eval.py).

#### 2. Graph, vector and FTS-based (hybrid) RAG and agent pipeline evaluation

We evaluate the RAG system that consists of a combination of graph, vector and FTS-based RAG
using the [Opik](https://www.comet.com/site/products/opik/) observability tool. The code for
evaluating the RAG system instruments the BAML prompts with Opik so that we can trace the execution
of each stage and quantify the system's performance. Guardrails are also added to showcase how to
protect from sensitive data leaking out of the system.

See the following docs in this repo for more details:
- [BAML instrumentation](src/BAML_INSTRUMENTATION_README.md)
- [Guardrails](src/GUARDRAILS_README.md)

## Graph visualization

It can help to visualize the graph to understand the structure of the data. Two options are provided
below.

### Option 1: Kuzu Explorer (Requires Docker)
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

### Option 2: G.V()

Alternatively, you can use the [G.V()](https://gdotv.com/) tool to visualize the graph.
This is an all-in-one graph database client to write, debug, test and analyze results for your graph database of choice. G.V() comes with first-class support for Kuzu databases -- simply drag-drop
your Kuzu graph database file into the G.V() desktop app, and you're good to go!

The full graph as visualized in G.V() is shown below:

![](./assets/fhir-graph-gdotv.png)
