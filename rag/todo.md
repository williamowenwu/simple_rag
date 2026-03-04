# Todo

- [x] Separated db logic

- [x] Chunking strategy (sentence/paragraph, overlap)

- [x] Hybrid retrieval — BM25 + vector + reranking

- [x] Async db client

5. - [x] Better model (7B+)
6. Documents to vector DB (PDFs, multiple sources)

## Productionizing

- [x] Wrap in FastAPI (async, streaming, request timeouts)

- [x] Managing sessions

9. Error handling and logging
10. Input validation (Pydantic — comes free with FastAPI)
11. Observability / OpenTelemetry

## Evaluation

12. RAGAS eval harness
13. Golden sets + regression gates in CI
14. pytest

## Infrastructure

15. CI/CD pipeline
16. Docker multi-stage builds
17. Kubernetes basics
18. README

## Literacy

19. Fine-tuning concepts — SFT vs LoRA/PEFT vs DPO
20. vLLM basics
21. Security/PII handling

## Later

LLM gateway (routing, fallbacks, rate limiting)
Hot swappable LLMs
Frontend
MCP
https://modelcontextprotocol.io/docs/getting-started/intro
Spark/pandas depth
document ingestion as a background job.
