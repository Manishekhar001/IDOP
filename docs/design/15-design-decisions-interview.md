# 15-design-decisions-interview: Rationale & Narrative

This manual records the core design philosophy, technical rationales, architectural tradeoffs, and stakeholder Q&A narratives backing the IDOP engineering decisions.

---

## 1. Project Genesis & Vision

Enterprise data ecosystems are notoriously fragmented. Analysts and business decision-makers are bottlenecked by:
1.  **Ticket Latencies**: Requesting a simple data aggregation or customer count from the database engineering team often takes days or weeks.
2.  **Safety & Access Restrictions**: Direct database connections are locked down due to severe security policies, data leakage risks, and command injectability.
3.  **Unstructured Blindspots**: Crucial corporate policies, refunds procedures, and vendor terms are locked inside document repositories (PDF, Excel, CSV), remaining invisible to traditional SQL databases.

**IDOP resolves this by acting as a zero-trust, high-safety gateway.** It abstracts SQL generation, document-driven updates, and unstructured vector search into a natural language chat interface, wrapped in cryptographically signed approval gates and strict validation guardrails.

---

## 2. Deep-Dive Design Decisions

### Decision 2.1: LangGraph vs. Simple LangChain Chains
*   **Problem**: Simple sequential LangChain chains are linear and fragile. They lack self-correction, cannot route dynamically, and are unable to execute human-in-the-loop approvals without losing state.
*   **Rationale**: **LangGraph** models the platform as a stateful, cyclic directed graph. If the generated SQL fails validation, it loops back to the generation node. If the RAG answer is unsupported by evidence, it loops back to the re-generation node. Checkpoints are automatically persisted inside PostgreSQL checkpointers (`AsyncPostgresSaver`), enabling sessions to survive service restarts and scale horizontally.

### Decision 2.2: Voyage AI Rerank-2.5 vs. Raw Vector Scores
*   **Problem**: Dense vector scores represent raw semantic proximity but struggle with fine-grained context relevance. They often rank minor semantic hits above highly relevant paragraphs containing slightly different wording.
*   **Rationale**: **Voyage AI Rerank-2.5** acts as a cross-encoder, comparing the precise syntax of the query and documents simultaneously. This ensures the absolute highest relevance context is pushed to the first 500 tokens of the context window, drastically reducing LLM hallucinations.

### Decision 2.3: Dual-Vector Hybrid Search vs. Single Vector
*   **Problem**: Dense embeddings (e.g. Nomic `nomic-embed-text-v1.5`) excel at capturing high-level intent but perform poorly on exact keyword lookups (serial numbers, email patterns, SKU codes). Sparse algorithms (BM25) capture keywords but miss semantic meaning.
*   **Rationale**: **Qdrant Dual-Vector Search** implements both. High-dimensional dense vectors represent abstract intent, while sparse BM25 indices capture alphanumeric exactness. Results are fused using **Reciprocal Rank Fusion (RRF)**, offering the best of both worlds.

### Decision 2.4: Redis Upstash vs. Disk Storage Cache
*   **Problem**: Document caching (chunks, embeddings) belongs in persistent storage, but query-level caching requires extremely low latency (under 10ms). Disk caches are slow and difficult to scale across clustered instances.
*   **Rationale**: We implement a **Four-Tier Caching System**. Upstash Redis acts as a distributed cache for transient queries, SQL generations, and SQL results with dynamic TTLs. Disk/S3 acts as a static document chunk cache. If Redis is severed, the system gracefully degrades to a thread-safe LRU Local Cache, preserving uptime.

---

## 3. Engineering Interview & Defense Narrative

This section acts as a study guide and defense framework for technical architecture reviews and stakeholder presentations:

### Q1: Why did we select LangGraph for this application instead of simpler agent frameworks like CrewAI or AutoGen?
> **Answer**: IDOP is a deterministic corporate workflow tool, not an open-ended playground for autonomous agents. CrewAI and AutoGen utilize autonomous loops that can deviate, generate runaway token bills, or fail to terminate. LangGraph enforces a strict state shape (`CSRAGState`) and predefined transitions (routing, validating, auditing, self-correction loops). This allows us to guarantee safe routing paths, enforce hard constraints (such as Approval Gates), and keep execution predictable.

### Q2: Why is the STM (Short-Term Memory) checkpointer hosted in PostgreSQL instead of a fast in-memory store like Redis?
> **Answer**: Redis is an excellent caching layer, but it is not designed to act as a system-of-record. LangGraph checkpointers store the absolute state of multi-step reasoning runs. In corporate environments, if a user takes 2 hours to approve an Excel mutation, the container could restart or scale down. PostgreSQL checkpointers provide ACID compliance and persistence guarantees, ensuring that pending sessions and state snapshots are never lost.

### Q3: Explain how the SQLValidator prevents SQL injections. Isn't Vanna or GPT-4o smart enough to avoid them?
> **Answer**: Relying on an LLM to prevent security breaches is a major vulnerability (prompt injections can bypass system rules). The `SQLValidator` acts as a deterministic, programmatic firewall. It parses the generated SQL and checks for forbidden tokens (`DROP`, `TRUNCATE`, `ALTER`, etc.) as distinct words. Furthermore, when the user executes mutations, the queries are parameterized before reaching Supabase Postgres, ensuring raw user inputs never interact with the database engine directly.

### Q4: How does the system handle "hallucinations" during RAG retrieval?
> **Answer**: IDOP uses a two-tier **Self-Reflective RAG (SRAG)** guardrail:
> 1.  **verify_support**: GPT-4o-mini acts as a Natural Language Inference (NLI) evaluator, validating if every claim in the generated answer is directly backed by the retrieved document chunks. If any claim is unsupported, the answer is rejected and routed to a `revise_answer` node.
> 2.  **verify_usefulness**: Checks if the answer actually addresses the user's question. If not, the query is rewritten, and the system starts a fresh retrieval run.

### Q5: Tavily Web Search is listed as a fallback. Why and when is it triggered?
> **Answer**: If the `CRAGEvaluator` scores retrieved document chunks below a relevance threshold of **0.3** (`INCORRECT`), it indicates that the company's internal knowledge store does not contain the answer. Rather than hallucinating or failing, the graph routes execution to the `web_search` node. The node queries the internet via Tavily API and feeds the live results to the generator, providing a robust fallback.

### Q6: What happens if a bulk spreadsheet upload contains 100 rows, and row 99 fails business rules?
> **Answer**: The entire operation is rolled back immediately. In enterprise database systems, partial mutations create dirty states (e.g. inserting 98 employees but failing on the 99th, leaving the database out of sync). The `MutationExecutor` runs the entire batch inside an isolated transaction block (`async with db_session.begin():`). If a single row fails a business rule or database constraint, an exception is thrown, the transaction is rolled back, and the database remains untouched.

### Q7: Why are approval sessions stored in-memory rather than database tables?
> **Answer**: In-memory caching for pending sessions (via a thread-safe dict or Redis with short TTLs) minimizes database pollution. Pending queries and mutations are transient—they are either approved within minutes or expire. Storing them in a fast cache ensures extremely low lookup times when the `/approve` endpoint is called.

---

## 4. Performance & Operational Standards

```
+------------------------------------+-----------------------------+-------------------+
| Operation                          | Latency Target              | Cache Status      |
+------------------------------------+-----------------------------+-------------------+
| /chat (Simple Memory Chat)         | < 800ms                     | Bypassed          |
| /chat (Cached RAG Hit)             | < 50ms                      | Tier 2 Hit        |
| /chat (Full CSRAG + Web Fallback)  | 5.0s - 8.0s (Reflection)    | Cache Miss        |
| /sql/generate (NL-to-SQL)          | 1.2s - 2.0s                 | Tier 3 Miss       |
| /mutation/upload (Parsing & Rules) | < 1.5s                      | Bypassed          |
+------------------------------------+-----------------------------+-------------------+
```

---

## Related Workflows

*   [01-system-architecture](./01-system-architecture.md) - Physical database connections.
*   [04-feature1-sql-execution](./04-feature1-sql-execution.md) - SQL validator and gate mechanisms.
*   [05-feature2-mutation-pipeline](./05-feature2-mutation-pipeline.md) - Dynamic transaction rollbacks.
*   [06-feature3-rag-pipeline](./06-feature3-rag-pipeline.md) - Self-corrective loops and evaluations.
