# Todo

1. Separated db logic
2. Chunking strategy (sentence/paragraph, overlap)
3. Hybrid retrieval — BM25 + vector + reranking
4. Async db client
5. Better model (7B+)
6. Documents to vector DB (PDFs, multiple sources)

## Productionizing 
7. Wrap in FastAPI (async, streaming, request timeouts) 
8. Managing sessions 
9. Error handling and logging 
10. Input validation (Pydantic — comes free with FastAPI) 
11. Observability / OpenTelemetry

## Evaluation 
12. RAGAS eval harness 
13. Golden sets + regression gates in CI 
14. pytest

## Infrastructure 
15. CI/CD pipeline 16. Docker multi-stage builds 17. Kubernetes basics 18. README

## Literacy 
19. Fine-tuning concepts — SFT vs LoRA/PEFT vs DPO 20. vLLM basics 21. Security/PII handling

## Later
LLM gateway (routing, fallbacks, rate limiting)
Hot swappable LLMs
Frontend
MCP
https://modelcontextprotocol.io/docs/getting-started/intro
Spark/pandas depth
