# 11 — Dual Memory Architecture (STM + LTM)

**Modules:** `app/core/memory/stm.py` · `app/core/memory/ltm.py` · `app/core/graph/nodes.py`
**Database:** PostgreSQL 16 (Docker container on EC2)
**LLM:** `get_memory_llm()` — defaults to `llama-3.3-70b-versatile` via LiteLLM Router (not `gpt-4o-mini`)

---

## Overview

IDOP implements a **dual memory architecture** that gives the LangGraph agent both short-term conversational recall and long-term user personalization:

- **Short-Term Memory (STM):** Conversation-scoped checkpoints stored via `AsyncPostgresSaver`. Preserves message history within a thread and automatically summarizes older messages to prevent context window overflow.
- **Long-Term Memory (LTM):** User-scoped factual preferences stored via `AsyncPostgresStore`. Persists user identity, preferences, and context across all conversations.

Both memory systems share the same PostgreSQL Docker instance (configured via `DATABASE_URL`) but use separate LangGraph table schemas. Both use `get_memory_llm()` (defaulting to `llama-3.3-70b-versatile` via LiteLLM Router) for their LLM operations.

---

## Memory Architecture Diagram

```mermaid
flowchart TD
    subgraph UserQuery["Incoming Query"]
        Q["User Message<br/>(thread_id, user_id)"]
    end

    subgraph LTM_Flow["Long-Term Memory (LTM)"]
        Q --> LR["ltm_remember_node"]
        LR --> LS["LTMService.read_memories<br/>(user_id, 'details')"]
        LS --> LP["Load existing facts<br/>from PostgreSQL"]
        LP --> LE["LTMService.extract_and_store<br/>GPT-4o-mini extracts new facts"]
        LE --> LW["Store new MemoryItem<br/>to AsyncPostgresStore"]
        LP --> LC["ltm_context injected<br/>into system prompt"]
    end

    subgraph Pipeline["Main Pipeline"]
        LC --> PP["router → ... → generate_answer"]
    end

    subgraph STM_Flow["Short-Term Memory (STM)"]
        PP --> SM["stm_summarize_node"]
        SM --> SC{"messages > threshold?<br/>(default: 6)"}
        SC -->|Yes| SS["STMSummarizer.summarize<br/>GPT-4o-mini compresses"]
        SS --> SD["Delete older messages<br/>(RemoveMessage ops)"]
        SD --> SP["summary prepended<br/>to system prompt"]
        SC -->|No| SK["Append AI message only"]
        SP --> CP["AsyncPostgresSaver<br/>checkpoint updated"]
        SK --> CP
    end

    CP --> END["Response → Client"]

    style UserQuery fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style LTM_Flow fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Pipeline fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style STM_Flow fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

---

## Short-Term Memory (STM)

### AsyncPostgresSaver

- **Source:** `langgraph-checkpoint-postgres` package
- **Purpose:** Stores conversation checkpoints — message lists and full state snapshots
- **Key:** `thread_id` (unique per conversation)
- **Setup:** [main.py](file:///c:/Users/manis/Downloads/Agentic-AI/IDOP/app/main.py#L40-L43)

```python
async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
    await checkpointer.setup()  # Creates checkpoint tables
```

### STMSummarizer

Defined in [stm.py](file:///c:/Users/manis/Downloads/Agentic-AI/IDOP/app/core/memory/stm.py):

- **Threshold:** `stm_message_threshold = 6` (configurable in settings)
- **LLM:** `get_memory_llm()` for summarization (defaults to `llama-3.3-70b-versatile` via LiteLLM Router)
- **Trigger:** When `len(messages) > threshold`

#### Summarization Logic

```python
class STMSummarizer:
    def should_summarize(self, messages: list) -> bool:
        return len(messages) > self._threshold  # default: 6

    async def summarize(self, messages, existing_summary) -> tuple[str, list]:
        # If existing summary exists, extend it
        # Otherwise, create new summary
        # Delete all but last 2 messages (keep recent context)
        messages_to_delete = messages[:-2]
        remove_ops = [RemoveMessage(id=m.id) for m in messages_to_delete]
        return new_summary, remove_ops
```

#### What Gets Summarized

| Condition | Action |
|---|---|
| `len(messages) ≤ 6` | No summarization; just append AI response |
| `len(messages) > 6`, no existing summary | Create fresh summary of entire conversation |
| `len(messages) > 6`, existing summary | Extend existing summary with new messages |

#### Message Retention

After summarization:
- **Kept:** Last 2 messages (preserves immediate context)
- **Deleted:** All older messages via `RemoveMessage` operations
- **Summary:** Prepended to system prompt for next invocation

### stm_summarize_node

Defined in [nodes.py](file:///c:/Users/manis/Downloads/Agentic-AI/IDOP/app/core/graph/nodes.py#L536-L552):

```python
async def stm_summarize_node(state: CSRAGState) -> dict:
    answer = state.get("answer", "")
    ai_msg = AIMessage(content=answer, id=str(uuid.uuid4()))
    summarizer = get_stm_summarizer()
    all_messages = [*list(state["messages"]), ai_msg]

    if summarizer.should_summarize(all_messages):
        new_summary, remove_ops = await summarizer.summarize(
            messages=all_messages,
            existing_summary=state.get("summary", ""),
        )
        return {"messages": [ai_msg, *remove_ops], "summary": new_summary}

    return {"messages": [ai_msg]}
```

---

## Long-Term Memory (LTM)

### AsyncPostgresStore

- **Source:** `langgraph.store.postgres.aio` package
- **Purpose:** Key-value store for user-specific facts and preferences
- **Namespace:** `("user", user_id, "details")` — scoped per user
- **Setup:** [main.py](file:///c:/Users/manis/Downloads/Agentic-AI/IDOP/app/main.py#L34-L37)

```python
async with AsyncPostgresStore.from_conn_string(settings.database_url) as store:
    await store.setup()  # Creates store tables
```

### LTMService

Defined in [ltm.py](file:///c:/Users/manis/Downloads/Agentic-AI/IDOP/app/core/memory/ltm.py):

- **Extraction LLM:** `get_memory_llm()` with structured output (`MemoryDecision`). Defaults to `llama-3.3-70b-versatile` via LiteLLM Router
- **Deduplication:** Each extracted fact is marked `is_new=true` or `is_new=false` relative to existing memories
- **Storage key:** UUID per fact, stored under `("user", user_id, "details")` namespace

#### Memory Extraction Schema

```python
class MemoryItem(BaseModel):
    text: str     # "User prefers dark mode for all dashboards"
    is_new: bool  # True if genuinely new vs existing memories

class MemoryDecision(BaseModel):
    should_write: bool       # True if any memory-worthy info exists
    memories: list[MemoryItem]
```

#### What Gets Extracted

The LTM extraction prompt specifies:
- **Identity:** Name, role, team, organization
- **Stable preferences:** Tools, languages, formats, UI preferences
- **Ongoing projects:** Active work, deadlines, goals
- **Context facts:** Domain expertise, data interests

#### What Gets Rejected

- Speculation or inferences not explicitly stated
- Duplicate information already in existing memories
- Transient conversational content (greetings, acknowledgments)

### ltm_remember_node

Defined in [nodes.py](file:///c:/Users/manis/Downloads/Agentic-AI/IDOP/app/core/graph/nodes.py#L173-L192):

```python
async def ltm_remember_node(state: CSRAGState, config: RunnableConfig, *, store) -> dict:
    user_id = config.get("configurable", {}).get("user_id", "default")
    ltm = get_ltm_service()

    # Extract the latest human message from state
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_message = last_human.content if last_human else ""

    # Extract new facts from latest user message
    await ltm.extract_and_store(store, user_id, user_message)

    # Read all facts for system prompt injection
    ltm_context = await ltm.read_memories(store, user_id)

    return {"user_id": user_id, "ltm_context": ltm_context}
```

---

## System Prompt Injection

Both memory types contribute to the system prompt via [_build_system_prompt](file:///c:/Users/manis/Downloads/Agentic-AI/IDOP/app/core/graph/nodes.py#L53-L68):

```python
def _build_system_prompt(ltm_context: str, summary: str) -> str:
    base = (
        "You are a knowledgeable and helpful assistant with memory capabilities.\n"
        "Answer questions clearly and concisely using the provided context.\n"
        "If no context is available, use your general knowledge.\n"
        "If you don't know the answer, say so clearly.\n"
        "Do not make up information."
    )
    sections = []
    if ltm_context and ltm_context != "(empty)":
        sections.append(f"Long-term user memory:\n{ltm_context}")
    if summary:
        sections.append(f"Recent conversation summary:\n{summary}")
    if sections:
        return base + "\n\n" + "\n\n".join(sections)
    return base
```

### Example Constructed System Prompt

```
You are a knowledgeable and helpful assistant with memory capabilities.
Answer questions clearly and concisely using the provided context.

Long-term user memory:
User works as a data analyst at Acme Corp.
User prefers CSV export format for reports.
User is working on Q3 sales forecasting project.

Recent conversation summary:
The user asked about top 10 customers by revenue and received a SQL query.
They then asked about applying a 15% discount threshold filter.
```

---

## PostgreSQL Table Structure

Both STM and LTM use the same PostgreSQL Docker instance but different table schemas managed by LangGraph:

| Memory Type | LangGraph Class | Tables Created | Key Structure |
|---|---|---|---|
| **STM** | `AsyncPostgresSaver` | `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` | `thread_id` |
| **LTM** | `AsyncPostgresStore` | `store` (key-value) | `(namespace, key)` where namespace = `("user", user_id, "details")` |

### Database Connection

```python
# From docker-compose.yml
DATABASE_URL: postgresql://postgres:postgres@idop-postgres:5432/postgres?sslmode=disable
```

---

## Memory Lifecycle

```
Request 1 (new user, new thread):
    LTM: read_memories → "(empty)" → extract → store 2 facts
    STM: 2 messages (human + AI) → no summarization
    System prompt: LTM facts prepended

Request 2 (same user, same thread):
    LTM: read_memories → "fact1\nfact2" → extract → store 1 new fact
    STM: 4 messages → no summarization
    System prompt: 3 LTM facts + no summary

Request 3 (same user, same thread):
    LTM: read_memories → "fact1\nfact2\nfact3" → no new facts
    STM: 6 messages → no summarization (threshold = 6, not exceeded)
    System prompt: 3 LTM facts + no summary

Request 4 (same user, same thread):
    LTM: read_memories → 3 facts → extract → 0 new facts
    STM: 8 messages → TRIGGER summarization
         → summarize all → keep last 2 → delete 6
    System prompt: 3 LTM facts + conversation summary
```

---

## Performance Characteristics

| Metric | STM | LTM |
|---|---|---|
| **Storage backend** | PostgreSQL (checkpoints) | PostgreSQL (key-value store) |
| **Scope** | Per thread (conversation) | Per user (cross-conversation) |
| **LLM model** | `get_memory_llm()` (defaults to `llama-3.3-70b-versatile`) | `get_memory_llm()` (defaults to `llama-3.3-70b-versatile`) |
| **Trigger** | Messages > 6 | Every user message |
| **LLM call cost** | ~$0.001 per summarization | ~$0.001 per extraction |
| **Latency** | 0.5–1.0s (summarization) | 0.3–0.8s (extraction) |
| **Data retention** | Per checkpoint (prunable) | Indefinite (append-only) |
| **Deduplication** | N/A (checkpoint overwrite) | `is_new` flag prevents duplicates |

---

## Related Workflows

- [07-langgraph-state-machine.md](./07-langgraph-state-machine.md) — Graph nodes `ltm_remember` and `stm_summarize`
- [10-srag-pipeline.md](./10-srag-pipeline.md) — SRAG verification before STM summarization
- [13-service-initialization.md](./13-service-initialization.md) — PostgreSQL store/checkpointer setup
- [14-deployment.md](./14-deployment.md) — PostgreSQL Docker container configuration
