# Production-Grade Observability for AI Agents Using OpenTelemetry, OpenInference, and Phoenix

## 1. Executive Summary

This document proposes a production-grade observability stack for our AI agents based on three open standards and tools: **OpenTelemetry** for tracing and metrics, **OpenInference** for AI/LLM–specific semantics, and **Arize Phoenix** for AI observability, tracing, and evaluation.

The goal is to make our AI systems **observable by default**: every decision, tool call, and model response can be inspected, measured, and improved. This stack is open source, cloud-agnostic, and aligns with industry direction around OpenTelemetry as the standard for distributed tracing and OpenInference as the emerging standard for AI observability semantics.

---

## 2. Why We Need Robust Observability for AI Agents

### 2.1 Observability gaps in agentic systems



Traditional logs or basic metrics only show the final outcome and a few counters; they do **not** explain *how* the agent arrived at an answer or why a failure occurred. When something goes wrong—hallucination, latency spike, unexpected tool usage—we need to be able to replay the full reasoning path and see every intermediate step.

### 2.2 Business risks without end-to-end tracing

Without deep observability, we face several risks
- **Debuggability risk**: Incident response is slow because we cannot see which step failed (LLM vs tool vs retrieval vs infra).
- **Quality risk**: We cannot attribute low-quality responses to specific prompts, models, or datasets.
- **Cost risk**: We lack visibility into token usage and cost per request, feature, or tenant.
- **Compliance risk**: We cannot reliably audit which data was retrieved or what prompts were sent for a given user.

An AI-specific observability stack addresses these by giving us trace-level visibility, metrics, and evaluation capabilities tailored to LLM/agent workloads.

---

## 3. Proposed Stack: OpenTelemetry + OpenInference + Phoenix

### 3.1 OpenTelemetry: the foundation for tracing and metrics

**OpenTelemetry (OTel)** is the industry-standard open framework for distributed tracing, metrics, and logs. It provides:
- SDKs for Python and other languages.
- Standard trace context (trace IDs, span IDs) across services.
- Exporters to send telemetry to multiple backends (Jaeger, SigNoz, Phoenix, etc.).

Using OTel means our AI observability data fits into the same pipeline as the rest of our services (API, DB, queues), and we are not tied to a single vendor.

### 3.2 OpenInference: AI/LLM semantics on top of OpenTelemetry

**OpenInference** is a specification and set of semantic conventions for representing AI workloads on top of OpenTelemetry.
 It defines:
- **Span kinds** for AI-specific operations: `LLM`, `AGENT`, `CHAIN`, `RETRIEVER`, `EMBEDDING`, `TOOL`, `EVALUATOR`, etc.
- **Standard attributes** for prompts, completions, models, embeddings, token counts, eval scores, and more (e.g., `input.value`, `output.value`, `llm.model_name`, `llm.token_count.total`).

By following OpenInference conventions, our traces encode not just timing, but also *meaning*: which spans are LLM calls, which are tools, which refer to evals, and how they relate to each other.

### 3.3 Phoenix: AI observability, tracing, and evaluation

**Arize Phoenix** is an open-source AI observability and evaluation platform that consumes OpenTelemetry traces annotated with OpenInference semantics. It provides:
- A UI to inspect traces of AI workloads, including prompts, completions, retrieval results, and tool calls.
- Dataset creation from traces, plus evaluation (LLM-as-a-judge, metrics, human labels) on those datasets.
- Integrations with popular AI frameworks (LangChain, LlamaIndex, etc.) and cloud providers.

Phoenix becomes the "glass box" into our AI agents: an engineer or product owner can look at a trace and literally see the agent think step by step.

---

## 4. What We Capture with This Stack

Using OpenTelemetry + OpenInference + Phoenix, each user request is represented as a **trace** composed of multiple **spans**. Below is what we will capture at each layer.

### 4.1 Trace-level information (per request)

For each agent turn (user message or task), we log:
- Unique IDs: trace ID, root span ID, and timestamps.
- Session & user: `session.id`, `user.id`, `tenant.id`, environment (dev/stage/prod). 
- Business context: feature name, use-case label (e.g., "support_chat", "doc_search").
- Outcome summary: final answer status (success/failure), high-level error category if any.

Phoenix and any other OTel backend can use these attributes to filter and group traces by tenant, feature, or environment.

### 4.2 Agent and orchestration spans (AGENT / CHAIN)

For each major step in the agent graph (planner, router, executor, etc.), we capture:
- Span kind `AGENT` or `CHAIN`.
- Inputs: current user query, context state, and relevant parameters (truncated or hashed if sensitive).
- Outputs: updated state, planned actions, or next tool to call.
- Orchestration metadata: node name, iteration number, branch taken, retry count.

This gives a clear picture of the agent’s reasoning path: which nodes ran, in what order, and why.

### 4.3 LLM spans (LLM)

For each call to an LLM provider, we record:
- Prompt input: messages and variables used to render the prompt (usually truncated or hashed in production).
- Output: model responses (again truncated or linked via ID for privacy).
- Model metadata: provider, `llm.model_name`, temperature, generation parameters.
- Token and cost metrics: prompt, completion, and total tokens.
- Status: latency, success/failure, error codes (timeouts, rate limits, safety blocks).

Phoenix can then show each LLM span with its prompt, output, timing, and token usage.

### 4.4 Retrieval, embeddings, and reranking spans (RETRIEVER / EMBEDDING / RERANKER)

For RAG and search operations, we log:
- Query text and retrieval filters.
- Retrieved document IDs, titles/snippets, and scores.
- Embedding model used, vector dimensions (vectors themselves may be hashed or omitted).
- Reranker inputs, new ranking scores, and chosen top results.

This lets us debug "why this document" and measure retrieval quality over time.

### 4.5 Tool and external API spans (TOOL)

For tools (both internal and external APIs), we capture:
- Tool name and description.
- Tool input parameters (with masking for sensitive fields).
- Tool output or a reference ID.
- API metadata: URL or service name, method, status code, retries.

This shows exactly which tools the agent used and how they contributed to the final answer.

### 4.6 Evaluation spans (EVALUATOR) and feedback

For evals and feedback loops, we log spans of kind `EVALUATOR`:
- Eval type and configuration (e.g., "factuality_eval", "safety_eval").
- Link to the evaluated span or trace (e.g., the `LLM` or `AGENT` span).
- Scores and labels: numeric scores, pass/fail, or categorical labels.
- Optional evaluator explanation text.

Phoenix uses this to provide dashboards and filters like "show traces where eval score < 0.7" or "compare models on the same dataset".

---

## 5. Architecture Overview

Conceptually, the architecture looks like this:

```text
User / Client
    |
    v
API Gateway / FastAPI  -- (OTel spans: HTTP)
    |
    v
Agent Service (LangGraph / LangChain)
    |   |- Spans: AGENT / CHAIN for planner/router/executor
    |   |- Spans: LLM for model calls
    |   |- Spans: RETRIEVER / EMBEDDING / RERANKER for RAG
    |   |- Spans: TOOL for external/internal tools
    |   |- Spans: EVALUATOR for quality checks
    v
OpenTelemetry SDK (Python) + OpenInference semantics
    |
    v
OTLP Export
    |   \
    |    \
    v     v
Phoenix (AI observability)    Generic tracing backend (Jaeger/SigNoz/etc.)
```

- **OpenTelemetry** handles trace context and exporting.
- **OpenInference** defines how to represent AI operations within those traces.
- **Phoenix** consumes those traces and provides an AI-optimized UI, datasets, and evals.

---

## 6. Why This Is the Most Robust Option for Us

### 6.1 Open standards and vendor neutrality

- OpenTelemetry is the de facto open standard for distributed tracing and metrics, supported by major clouds and observability vendors.
- OpenInference builds AI semantics on top of OpenTelemetry rather than inventing a proprietary schema.
- Phoenix is open source and uses these standards, so we can self-host or move to another backend without losing our instrumentation investment.

This avoids lock-in while still giving us advanced AI observability features.

### 6.2 Depth of insight across the whole agent lifecycle

Because we instrument at every layer—HTTP, agent graph, LLM, retrieval, tools, and evaluations—we get:
- End-to-end traces for each user request.
- The ability to replay and debug complex agent runs step by step.
- Visibility into both technical SLOs (latency, errors) and quality metrics (eval scores, hallucinations, safety events).

Generic APM tools cannot provide this level of AI-specific insight without significant customization.

### 6.3 Fit for multi-tenant SaaS and compliance

OpenTelemetry and OpenInference allow us to attach tenant IDs, user IDs, and environment tags to every span, enabling:
- Per-tenant metrics and dashboards.
- Tailored sampling and retention policies by tenant or environment.
- Easier compliance and audit trails (who saw what, and when).

We can also centralize sensitive content handling (masking or hashing prompts/outputs) at the instrumentation layer.

### 6.4 Extensibility and ecosystem

By aligning with OTel + OpenInference + Phoenix, we gain access to a broader ecosystem:
- Framework integrations (LangChain, LlamaIndex, etc.) that already emit compatible traces.
- Potential integrations with ML platforms (e.g., MLflow) that map AI attributes into their own schema.
- Community-driven updates to semantic conventions as AI best practices evolve.

We can incrementally extend our instrumentation (new tools, new evals) without changing the underlying stack.

---




