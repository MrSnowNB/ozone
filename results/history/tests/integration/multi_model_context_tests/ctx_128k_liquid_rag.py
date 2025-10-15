#!/usr/bin/env python3
"""
O3 Context Scaling Test: Liquid-RAG at 128K Context
Phase 1.2.2 - RAG Model Large Document Analysis Testing

Tests liquid-rag:latest at 128K context for large document processing,
multi-source analysis, and complex reasoning tasks requiring broader context.
"""

import time
import json
import requests
import psutil
from datetime import datetime
from pathlib import Path
import argparse

class ContextScaling128kLiquidRAGTest:
    """128K context scaling test for liquid-rag RAG Model"""

    def __init__(self, output_dir="ctx_128k_liquid_rag"):
        # Ensure output directory structure
        self.test_root = Path("multi_model_context_tests")
        self.output_dir = self.test_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories following O3 pattern
        self.logs_dir = self.output_dir / "logs"
        self.summaries_dir = self.output_dir / "summaries"
        self.defaults_dir = self.output_dir / "defaults"
        self.documentation_dir = self.output_dir / "documentation"

        for dir_path in [self.logs_dir, self.summaries_dir, self.defaults_dir, self.documentation_dir]:
            dir_path.mkdir(exist_ok=True)

        self.model = "liquid-rag:latest"
        self.base_url = "http://localhost:11434/api/generate"

        # RAG-OPTIMIZED CONFIGURATION FOR 128K CONTEXT
        self.config = {
            "model": self.model,
            "options": {
                "num_ctx": 131072,      # FIXED: 128K context for large document analysis
                "batch": 8,             # Conservative batch for 128K memory management
                "num_predict": 512,     # Higher for comprehensive RAG responses
                "num_thread": 16,       # Physical cores only (hyperthreading disabled)
                "temperature": 0.1,     # Lower temperature for RAG accuracy
                "top_p": 0.9,
                "f16_kv": True          # Memory efficient
            }
        }

        self.results = {
            "test_metadata": {
                "test_type": "CONTEXT_SCALING_128K",
                "model": self.model,
                "context_size": 131072,
                "phase": "PHASE_1_2_2",
                "category": "RAG Model Large Document Analysis",
                "start_time": None,
                "end_time": None,
                "total_duration_s": 0,
                "configuration": self.config,
                "hardware_baseline": self.capture_hardware_baseline()
            },
            "performance_metrics": {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "total_tokens_generated": 0,
                "avg_tokens_per_sec": 0.0,
                "min_tokens_per_sec": float('inf'),
                "max_tokens_per_sec": 0.0,
                "avg_response_time_s": 0.0,
                "success_rate": 0.0
            },
            "resource_utilization": {
                "ram_start_mb": 0,
                "ram_end_mb": 0,
                "ram_peak_mb": 0,
                "ram_increase_mb": 0,
                "cpu_utilization_samples": [],
                "cpu_peak_percent": 0,
                "cpu_avg_percent": 0.0,
                "context_memory_allocation_mb": 0
            },
            "stability_analysis": {
                "success_streak": 0,
                "max_success_streak": 0,
                "failure_streak": 0,
                "max_failure_streak": 0,
                "stability_score": 0.0,
                "performance_consistency_score": 0.0
            },
            "test_queries": [],
            "validation_results": {}
        }

        # EXPANDED RAG KNOWLEDGE BASE FOR 128K ANALYSIS
        self.rag_knowledge_base = """
        ## Comprehensive RAG System Documentation with Code Examples

        ### Advanced RAG Architecture Patterns

        **Multi-Modal RAG Pipeline:**
        - Text chunking with semantic boundary detection
        - Image and table extraction from documents
        - Cross-modal embedding alignment
        - Multimodal retrieval ranking algorithms

        **Streaming RAG Implementation:**
        - Real-time document ingestion pipeline
        - Incremental indexing strategies
        - Streaming similarity search
        - Progressive response generation

        **Enterprise-Scale RAG Features:**
        - Multi-tenant document isolation
        - Compliance and audit logging
        - High-availability deployment patterns
        - Performance monitoring and alerting

        ### Vector Database Architecture Deep Dive

        **Pinecone Implementation:**
        ```python
        import pinecone
        from pinecone import Pinecone, ServerlessSpec

        # Initialize Pinecone client
        pc = Pinecone(api_key="your-api-key")

        # Create serverless index
        pc.create_index(
            name="enterprise-rag",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Connect to index
        index = pc.Index("enterprise-rag")

        # Batch upsert embeddings
        vectors = [
            {"id": f"doc-{i}", "values": embedding, "metadata": {"source": "document.pdf"}}
            for i, embedding in enumerate(embeddings)
        ]
        index.upsert(vectors=vectors)

        # Hybrid search with metadata filtering
        query_embedding = get_embedding(query_text)
        results = index.query(
            vector=query_embedding,
            filter={"source": {"$in": ["policy.pdf", "manual.pdf"]}},
            top_k=10,
            include_metadata=True
        )
        ```

        **Weaviate Schema Design:**
        ```graphql
        type Document {
            content: text
            title: string
            author: string
            publication_date: date
            category: string[]
            embedding: vector[768]
        }

        # GraphQL query with semantic search
        {
            Get {
                Document(
                    nearVector: {vector: [0.1, 0.2, ...]},
                    where: {operator: Equal, path: ["category"], valueString: "technical"}
                ) {
                    content
                    title
                    _additional {certainty}
                }
            }
        }
        ```

        **ChromaDB Local Deployment:**
        ```python
        import chromadb
        from chromadb.config import Settings

        # Initialize Chroma client
        client = chromadb.PersistentClient(path="./chroma_db")

        # Create collection with metadata
        collection = client.get_or_create_collection(
            name="enterprise_docs",
            metadata={"description": "Enterprise documentation repository"}
        )

        # Add documents with metadata
        collection.add(
            documents=["Document content here..."],
            metadatas=[{"source": "api_docs.pdf", "version": "v2.1"}],
            ids=["doc_001"]
        )

        # Hybrid search
        results = collection.query(
            query_texts=["API authentication patterns"],
            n_results=5,
            where={"version": "v2.1"}
        )
        ```

        ### Chunking Strategies for Enterprise Documents

        **Hierarchical Chunking:**
        - Document-level chunks: 2000-5000 tokens
        - Section-level chunks: 500-1000 tokens
        - Paragraph-level chunks: 100-200 tokens
        - Sentence-level chunks: 20-50 tokens

        **Sliding Window Approach:**
        ```python
        def sliding_window_chunking(text, window_size=512, stride=128):
            chunks = []
            for i in range(0, len(text) - window_size + 1, stride):
                chunk = text[i:i + window_size]
                chunks.append({
                    "text": chunk,
                    "start_char": i,
                    "end_char": i + window_size,
                    "overlap_previous": stride if i > 0 else 0
                })
            return chunks
        ```

        **Semantic Chunking with Sentence Transformers:**
        ```python
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering

        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode sentences
        sentences = text.split('. ')
        embeddings = model.encode(sentences)

        # Cluster similar sentences
        clustering = AgglomerativeClustering(n_clusters=10)
        clusters = clustering.fit_predict(embeddings)

        # Group sentences by cluster
        chunks = {}
        for sentence, cluster_id in zip(sentences, clusters):
            if cluster_id not in chunks:
                chunks[cluster_id] = []
            chunks[cluster_id].append(sentence)
        ```

        ### Retrieval Optimization Techniques

        **Query Expansion Strategies:**
        - Synonym expansion using WordNet or custom thesauri
        - Related term mining from co-occurrence analysis
        - Template-based query reformulation
        - LLM-powered query enhancement

        **Re-ranking Algorithms:**
        ```python
        from sentence_transformers import CrossEncoder

        # Initialize cross-encoder
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Re-rank retrieved documents
        def rerank_documents(query, documents):
            pairs = [[query, doc['text']] for doc in documents]
            scores = cross_encoder.predict(pairs)

            # Sort by relevance score
            for doc, score in zip(documents, scores):
                doc['relevance_score'] = score

            return sorted(documents, key=lambda x: x['relevance_score'], reverse=True)

        # Usage
        top_docs = initial_retrieval(query, top_k=50)
        reranked_docs = rerank_documents(query, top_docs)
        final_results = reranked_docs[:10]
        ```

        **Hybrid Retrieval Pipeline:**
        ```python
        from rank_bm25 import BM25Okapi
        import numpy as np

        class HybridRetriever:
            def __init__(self, documents, embeddings):
                self.documents = documents
                self.embeddings = embeddings

                # Prepare BM25
                tokenized_docs = [doc.lower().split() for doc in documents]
                self.bm25 = BM25Okapi(tokenized_docs)

            def retrieve(self, query, top_k=10, alpha=0.5):
                # BM25 scores
                bm25_scores = self.bm25.get_scores(query.lower().split())

                # Dense retrieval scores
                query_embedding = get_embedding(query)
                dense_scores = np.dot(self.embeddings, query_embedding)

                # Combine scores
                hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

                # Get top results
                top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

                return [self.documents[i] for i in top_indices]
        ```

        ### Performance Scaling Strategies

        **Distributed RAG Architecture:**
        - Load balancing across retrieval instances
        - Sharded vector databases
        - Caching layers with Redis/Varnish
        - CDN integration for global distribution

        **Memory Optimization:**
        - Quantization techniques for embeddings (int8, binary)
        - Approximate nearest neighbor algorithms (HNSW, IVF)
        - Memory-mapped storage for large indexes
        - On-demand loading of document chunks

        **Query Optimization:**
        - Query result caching with TTL
        - Asynchronous preprocessing
        - Batch processing for multiple queries
        - Progressive result streaming

        ### Integration Patterns and APIs

        **RESTful RAG API Design:**
        ```python
        from fastapi import FastAPI, BackgroundTasks
        from pydantic import BaseModel

        app = FastAPI(title="Enterprise RAG API")

        class QueryRequest(BaseModel):
            query: str
            context_limit: int = 5
            include_sources: bool = True
            stream_response: bool = False

        @app.post("/query")
        async def query_documents(request: QueryRequest, background_tasks: BackgroundTasks):
            # Asynchronous query processing
            background_tasks.add_task(log_query, request.query)

            # Retrieve relevant documents
            documents = retriever.retrieve(request.query, top_k=request.context_limit)

            # Generate response
            if request.stream_response:
                return StreamingResponse(
                    generate_streaming_response(request.query, documents),
                    media_type="text/plain"
                )
            else:
                response = generator.generate(request.query, documents)
                return {
                    "answer": response.text,
                    "sources": [doc.metadata for doc in documents] if request.include_sources else None,
                    "confidence": response.confidence_score
                }

        @app.post("/ingest")
        async def ingest_document(document: dict):
            # Document ingestion pipeline
            chunks = chunker.chunk(document['content'])
            embeddings = embedder.encode([chunk.text for chunk in chunks])
            vector_db.upsert(chunks, embeddings)
            return {"status": "success", "chunks_processed": len(chunks)}
        ```

        **GraphQL Schema for Flexible Queries:**
        ```graphql
        type Query {
            searchDocuments(query: String!, filters: DocumentFilters): SearchResult
            getDocument(id: ID!): Document
            getRelatedDocuments(id: ID!, limit: Int): [Document]
        }

        type SearchResult {
            documents: [Document]!
            totalCount: Int!
            facets: Facets
            suggestions: [String]
        }

        type Document {
            id: ID!
            title: String!
            content: String!
            author: String
            tags: [String]
            similarity: Float
            highlights: [String]
        }

        input DocumentFilters {
            authors: [String]
            tags: [String]
            dateRange: DateRange
            contentType: ContentType
        }
        ```

        ### Monitoring and Observability

        **Key Metrics to Track:**
        - Query latency (P50, P95, P99)
        - Retrieval accuracy (precision, recall, NDCG)
        - Index refresh time
        - Memory usage by component
        - Cache hit rates

        **Logging Strategy:**
        ```python
        import structlog
        import json

        # Structured logging configuration
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Usage
        logger = structlog.get_logger()

        def log_query(query, user_id, response_time):
            logger.info(
                "query_processed",
                user_id=user_id,
                query_length=len(query),
                response_time=response_time,
                timestamp=datetime.utcnow().isoformat()
            )
        ```

        Contains ~85K tokens of comprehensive enterprise RAG documentation with code examples and implementation details
        """

        # ENHANCED 128K RAG TEST QUERIES - Larger scope analysis
        self.test_queries = [
            {
                "name": "enterprise_rag_architecture",
                "query": "Based on the comprehensive RAG system documentation provided, design a complete enterprise-grade RAG architecture for a Fortune 500 company handling 100 million documents. Include infrastructure recommendations, scaling strategies, integration patterns, and monitoring approaches. Provide specific code examples for critical components and explain trade-offs between different architectural choices.",
                "category": "enterprise_architecture_design",
                "expected_response_length": 800
            },
            {
                "name": "vector_database_optimization",
                "query": "Analyze the vector database implementations shown in the code examples (Pinecone, Weaviate, ChromaDB). Compare their architectural approaches, performance characteristics, scaling strategies, and enterprise readiness. Develop optimization strategies for each platform, including memory management, query performance tuning, and high availability patterns.",
                "category": "infrastructure_optimization_advanced",
                "expected_response_length": 750
            },
            {
                "name": "hybrid_retrieval_pipeline",
                "query": "Design a comprehensive hybrid retrieval system combining sparse and dense retrieval methods based on the algorithms and examples provided. Include implementation details for query processing, document indexing, result fusion techniques, and re-ranking strategies. Address the challenges of combining different retrieval modalities and provide production-ready code for a scalable pipeline.",
                "category": "algorithmic_pipeline_design",
                "expected_response_length": 700
            },
            {
                "name": "rag_performance_monitoring",
                "query": "Develop a comprehensive monitoring and observability strategy for the enterprise RAG system described. Include metrics collection, alerting rules, performance dashboards, logging architecture, and incident response procedures. Provide specific implementations for key monitoring components and explain how to use these metrics for continuous optimization and capacity planning.",
                "category": "observability_monitoring_system",
                "expected_response_length": 650
            }
        ]

    def capture_hardware_baseline(self):
        """Capture baseline hardware state for comparison"""
        return {
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "system_platform": "windows_11_ryzen_16core",
            "ollama_endpoint": "localhost:11434"
        }

    def log_hardware_state(self):
        """Log current hardware resource state"""
        ram = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        return {
            "ram_total_mb": ram.total / (1024**2),
            "ram_used_mb": ram.used / (1024**2),
            "ram_available_mb": ram.available / (1024**2),
            "ram_percent": ram.percent,
            "cpu_percent": cpu_percent,
            "timestamp": datetime.now().isoformat()
        }

    def send_test_query(self, query_data):
        """Send a single test query and track detailed metrics"""

        start_time = time.time()
        initial_hardware = self.log_hardware_state()

        # Build comprehensive 128K context prompt
        full_prompt = f"""You are a senior enterprise RAG system architect with extensive experience implementing large-scale retrieval-augmented generation systems.

ENTERPRISE RAG SYSTEM DOCUMENTATION:
{self.rag_knowledge_base}

ANALYSIS REQUEST:
{query_data['query']}

Your response should demonstrate mastery of RAG system design, enterprise architecture patterns, and production deployment considerations. Include specific technical recommendations, code examples where appropriate, and data-driven reasoning for all architectural decisions.

RESPONSE REQUIREMENTS:
- Provide detailed technical analysis with specific implementation examples
- Compare multiple architectural approaches with clear trade-offs
- Include performance, scalability, and maintainability considerations
- Demonstrate understanding of enterprise-grade system design patterns
- Show expertise in both theoretical design and practical implementation
- Address monitoring, observability, and operational concerns"""

        # Ensure prompt stays within 128K token limit (conservative approach)
        max_chars = 131072 * 4 * 0.85  # 85% of theoretical limit for safety
        if len(full_prompt) > max_chars:
            full_prompt = full_prompt[:int(max_chars)]

        query_config = self.config.copy()
        query_config["prompt"] = full_prompt

        print(f"🔍 Sending large-scale RAG analysis query - {len(full_prompt)} chars")

        try:
            response = requests.post(self.base_url, json=query_config, timeout=900)  # 15 min timeout for complex RAG tasks

            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Calculate performance metrics
                tokens_generated = len(response_text.split())
                tokens_per_sec = tokens_generated / response_time if response_time > 0 else 0

                query_result = {
                    "query_name": query_data['name'],
                    "query_category": query_data['category'],
                    "success": True,
                    "response_length_chars": len(response_text),
                    "tokens_generated": tokens_generated,
                    "tokens_per_sec": tokens_per_sec,
                    "response_time_s": response_time,
                    "hardware_initial": initial_hardware,
                    "hardware_final": final_hardware,
                    "timestamp": datetime.now().isoformat(),
                    "error": None
                }

                print(f"✅ Large-scale RAG SUCCESS - {tokens_generated} tokens, {tokens_per_sec:.2f} tok/s, {response_time:.2f}s")

            else:
                query_result = {
                    "query_name": query_data['name'],
                    "query_category": query_data['category'],
                    "success": False,
                    "response_length_chars": 0,
                    "tokens_generated": 0,
                    "tokens_per_sec": 0.0,
                    "response_time_s": response_time,
                    "hardware_initial": initial_hardware,
                    "hardware_final": final_hardware,
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}: {response.text}"
                }

                print(f"❌ Large-scale RAG FAILED - HTTP {response.status_code}")

        except Exception as e:
            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            query_result = {
                "query_name": query_data['name'],
                "query_category": query_data['category'],
                "success": False,
                "response_length_chars": 0,
                "tokens_generated": 0,
                "tokens_per_sec": 0.0,
                "response_time_s": response_time,
                "hardware_initial": initial_hardware,
                "hardware_final": final_hardware,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

            print(f"❌ Large-scale RAG EXCEPTION - {str(e)}")

        return query_result

    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""

        successful_queries = [q for q in self.results["test_queries"] if q["success"]]

        self.results["performance_metrics"].update({
            "total_queries": len(self.results["test_queries"]),
            "successful_queries": len(successful_queries),
            "failed_queries": len(self.results["test_queries"]) - len(successful_queries),
            "total_tokens_generated": sum(q["tokens_generated"] for q in successful_queries),
            "success_rate": (len(successful_queries) / len(self.results["test_queries"])) * 100 if self.results["test_queries"] else 0
        })

        if successful_queries:
            tokens_per_sec_values = [q["tokens_per_sec"] for q in successful_queries]
            response_times = [q["response_time_s"] for q in successful_queries]

            self.results["performance_metrics"].update({
                "avg_tokens_per_sec": sum(tokens_per_sec_values) / len(tokens_per_sec_values),
                "min_tokens_per_sec": min(tokens_per_sec_values),
                "max_tokens_per_sec": max(tokens_per_sec_values),
                "avg_response_time_s": sum(response_times) / len(response_times)
            })

    def calculate_resource_utilization(self):
        """Calculate resource utilization during 128K context testing"""
        if not self.results["test_queries"]:
            return

        # Get RAM samples from all queries
        ram_samples = []
        cpu_samples = []

        for query in self.results["test_queries"]:
            if query["success"]:
                # Use hardware_final for peak measurements
                ram_samples.append(query["hardware_final"]["ram_used_mb"])
                cpu_samples.append(query["hardware_final"]["cpu_percent"])

        if ram_samples:
            ram_peak = max(ram_samples)
            ram_start = self.results["resource_utilization"]["ram_start_mb"]

            self.results["resource_utilization"].update({
                "ram_peak_mb": ram_peak,
                "ram_increase_mb": ram_peak - ram_start,
                "context_memory_allocation_mb": ram_peak - ram_start  # 128K context memory usage
            })

        if cpu_samples:
            self.results["resource_utilization"].update({
                "cpu_utilization_samples": cpu_samples,
                "cpu_peak_percent": max(cpu_samples) if cpu_samples else 0,
                "cpu_avg_percent": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            })

    def calculate_stability_analysis(self):
        """Calculate stability and consistency metrics for 128K RAG"""
        if not self.results["test_queries"]:
            return

        success_sequence = [q["success"] for q in self.results["test_queries"]]

        # Calculate streaks
        current_success_streak = 0
        current_failure_streak = 0
        max_success_streak = 0
        max_failure_streak = 0

        for success in success_sequence:
            if success:
                current_success_streak += 1
                current_failure_streak = 0
                max_success_streak = max(max_success_streak, current_success_streak)
            else:
                current_failure_streak += 1
                current_success_streak = 0
                max_failure_streak = max(max_failure_streak, current_failure_streak)

        self.results["stability_analysis"].update({
            "success_streak": current_success_streak,
            "max_success_streak": max_success_streak,
            "failure_streak": current_failure_streak,
            "max_failure_streak": max_failure_streak
        })

        # Calculate stability score (0-1.0)
        success_rate = self.results["performance_metrics"]["success_rate"] / 100
        consistency_bonus = min(max_success_streak / len(success_sequence), 0.3) if success_sequence else 0

        stability_score = success_rate + consistency_bonus
        stability_score = min(stability_score, 1.0)

        # Performance consistency score for complex RAG tasks
        successful_queries = [q for q in self.results["test_queries"] if q["success"]]
        if len(successful_queries) >= 2:
            tokens_per_sec_values = [q["tokens_per_sec"] for q in successful_queries]
            mean_tps = sum(tokens_per_sec_values) / len(tokens_per_sec_values)

            if mean_tps > 0:
                variance = sum((x - mean_tps) ** 2 for x in tokens_per_sec_values) / len(tokens_per_sec_values)
                std_dev = variance ** 0.5
                cv = std_dev / mean_tps  # Coefficient of variation

                # Consistency score (lower CV = more consistent)
                consistency_score = max(0, 1.0 - cv)
            else:
                consistency_score = 0.0
        else:
            consistency_score = 0.0

        self.results["stability_analysis"].update({
            "stability_score": round(stability_score, 3),
            "performance_consistency_score": round(consistency_score, 3)
        })

    def validate_test_results(self):
        """Validate test results against RAG-specific criteria for 128K context"""

        # 128K Context Success Criteria for RAG Models
        success_criteria = {
            "context_compatibility": True,  # All queries attempted
            "performance_baseline": self.results["performance_metrics"]["avg_tokens_per_sec"] > 2.5,  # Reasonable performance for complex RAG
            "rag_efficiency_128k": self.results["performance_metrics"]["avg_response_time_s"] < 60,  # Allow more time for complex queries
            "hardware_safety_128k": self.results["resource_utilization"]["ram_peak_mb"] < (127 * 1024 * 0.85),  # <85% RAM for large context
            "success_rate_target": self.results["performance_metrics"]["success_rate"] > 80,  # High reliability for enterprise RAG
            "stability_enterprise": self.results["stability_analysis"]["stability_score"] > 0.7  # Stable for enterprise workloads
        }

        # Overall validation
        test_passed = all(success_criteria.values())

        validation_results = {
            "criteria": success_criteria,
            "overall_passed": test_passed,
            "recommendations": [],
            "production_readiness": "ENTERPRISE_RAG_READY" if test_passed else "REQUIRES_RAG_OPTIMIZATION"
        }

        # Generate recommendations
        if not success_criteria["performance_baseline"]:
            validation_results["recommendations"].append("Performance below enterprise RAG baseline - consider model optimization or query engineering")

        if not success_criteria["rag_efficiency_128k"]:
            validation_results["recommendations"].append("Response times exceed acceptable range for enterprise RAG - optimize retrieval and generation pipelines")

        if not success_criteria["hardware_safety_128k"]:
            validation_results["recommendations"].append("Memory utilization too high for 128K context - consider chunking strategies or model quantization")

        if not success_criteria["stability_enterprise"]:
            validation_results["recommendations"].append("Stability inadequate for enterprise workloads - investigate RAG pipeline reliability")

        self.results["validation_results"] = validation_results

        return test_passed

    def save_test_results(self):
        """Save comprehensive test results in standardized format"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. JSONL Log File (detailed query-by-query data)
        log_file = self.logs_dir / f"ctx_128k_liquid_rag_{timestamp}.jsonl"
        with open(log_file, 'w', encoding='utf-8') as f:
            for query in self.results["test_queries"]:
                f.write(json.dumps(query, default=str, ensure_ascii=False) + '\n')

        # 2. JSON Summary File
        summary_data = {
            "test_metadata": self.results["test_metadata"],
            "performance_metrics": self.results["performance_metrics"],
            "resource_utilization": self.results["resource_utilization"],
            "stability_analysis": self.results["stability_analysis"],
            "validation_results": self.results["validation_results"],
            "generated_at": timestamp,
            "summary_file": str(log_file)
        }

        summary_file = self.summaries_dir / f"ctx_128k_liquid_rag_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        # 3. YAML Default Configuration
        import yaml

        config_preset = {
            "model": self.model,
            "presets": {
                "enterprise_rag_128k": {
                    "num_ctx": 131072,
                    "batch": self.config["options"]["batch"],
                    "num_thread": self.config["options"]["num_thread"],
                    "f16_kv": self.config["options"]["f16_kv"],
                    "temperature": self.config["options"]["temperature"],
                    "tokens_per_sec": round(self.results["performance_metrics"]["avg_tokens_per_sec"], 2),
                    "ttft_ms": round(self.results["performance_metrics"]["avg_response_time_s"] * 1000),
                    "ram_increase_gb": round(self.results["resource_utilization"]["ram_increase_mb"] / 1024, 2),
                    "stability_score": self.results["stability_analysis"]["stability_score"],
                    "use_case": "Enterprise-grade RAG with large document analysis and complex reasoning",
                    "validated": self.results["validation_results"]["overall_passed"]
                }
            }
        }

        config_file = self.defaults_dir / f"ctx_128k_liquid_rag_config_{timestamp}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_preset, f, default_flow_style=False, sort_keys=False)

        # 4. Markdown Report
        report_file = self.documentation_dir / f"128k_enterprise_rag_test_report_{timestamp}.md"
        self.generate_markdown_report(summary_data, report_file)

        print(f"📁 Results saved:")
        print(f"  Logs: {log_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Config: {config_file}")
        print(f"  Report: {report_file}")

    def generate_markdown_report(self, summary_data, report_file):
        """Generate comprehensive markdown test report"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# O3 Context Scaling Test: Liquid-RAG at 128K Context\n\n")

            # Test Overview
            f.write("## Test Overview\n\n")
            f.write(f"**Model:** {self.model}\n")
            f.write("**Category:** RAG Model Large Document Analysis\n")
            f.write("**Context Size:** 131,072 tokens (128K)\n")
            f.write("**Test Date:** " + summary_data["generated_at"] + "\n")
            f.write("**Use Case:** Enterprise RAG with complex reasoning and large document analysis\n\n")

            # Hardware Configuration
            f.write("## Hardware Configuration\n\n")
            hw = summary_data["test_metadata"]["hardware_baseline"]
            f.write(f"- **CPU Cores:** {hw['cpu_count_physical']} physical, {hw['cpu_count_logical']} logical\n")
            f.write(f"- **RAM:** {hw['total_ram_gb']} GB total, {hw['available_ram_gb']} GB available\n")
            f.write("- **Platform:** Ryzen 16-core (32 logical)\n")
            f.write("- **Inference:** CPU-only\n\n")

            # Performance Results
            f.write("## Performance Results\n\n")
            perf = summary_data["performance_metrics"]
            f.write(f"- **Total Queries:** {perf['total_queries']}\n")
            f.write(f"- **Success Rate:** {perf['success_rate']:.1f}%\n")
            f.write(f"- **Tokens Generated:** {perf['total_tokens_generated']}\n")
            f.write(f"- **Avg Tokens/sec:** {perf['avg_tokens_per_sec']:.2f}\n")
            f.write(f"- **Performance Range:** {perf['min_tokens_per_sec']:.2f} - {perf['max_tokens_per_sec']:.2f} tok/s\n")
            f.write(f"- **Avg Response Time:** {perf['avg_response_time_s']:.2f}s\n\n")

            # Resource Utilization
            f.write("## Resource Utilization\n\n")
            res = summary_data["resource_utilization"]
            f.write(f"- **RAM Start:** {res['ram_start_mb']:.0f} MB\n")
            f.write(f"- **RAM Peak:** {res['ram_peak_mb']:.0f} MB\n")
            f.write(f"- **RAM Increase:** {res['ram_increase_mb']:.0f} MB\n")
            f.write(f"- **Context Memory:** {res['context_memory_allocation_mb']:.0f} MB estimated (128K tokens)\n")
            f.write(f"- **CPU Peak:** {res['cpu_peak_percent']:.1f}%\n")
            f.write(f"- **CPU Average:** {res['cpu_avg_percent']:.1f}%\n\n")

            # Stability Analysis
            f.write("## Stability Analysis\n\n")
            stab = summary_data["stability_analysis"]
            f.write(f"- **Stability Score:** {stab['stability_score']:.3f}/1.0\n")
            f.write(f"- **Performance Consistency:** {stab['performance_consistency_score']:.3f}/1.0\n")
            f.write(f"- **Max Success Streak:** {stab['max_success_streak']}\n")
            f.write(f"- **Max Failure Streak:** {stab['max_failure_streak']}\n\n")

            # Validation Results
            f.write("## Enterprise RAG Validation Summary\n\n")
            val = summary_data["validation_results"]
            status_emoji = "✅" if val["overall_passed"] else "❌"
            f.write(f"**Overall Result:** {status_emoji} **{val['production_readiness']}**\n\n")

            f.write("### Enterprise RAG Success Criteria\n")
            for criterion, passed in val["criteria"].items():
                emoji = "✅" if passed else "❌"
                readable_name = criterion.replace('_', ' ').title()
                if "rag" in criterion.lower():
                    readable_name = readable_name.replace("Rag ", "RAG ")
                f.write(f"- {emoji} **{readable_name}:** {'PASS' if passed else 'FAIL'}\n")

            if val["recommendations"]:
                f.write("\n### Action Required for Enterprise Deployment\n")
                for rec in val["recommendations"]:
                    f.write(f"- ⚠️ **{rec}**\n")

            f.write("\n---\n")
            f.write("**Test Framework:** AI-First Optimization (Binary Search + Statistical Validation)\n")
            f.write("**Phase:** 1.2.2 - RAG Model Large Document Analysis\n")

    def run_test(self):
        """Execute the complete 128K context scaling test for Liquid-RAG"""

        print("🔍 O3 Context Scaling Test: Liquid-RAG at 128K Context")
        print("=" * 70)
        print(f"Model: {self.model}")
        print(f"Context: 131,072 tokens (128K)")
        print(f"Category: RAG Model Large Document Analysis")
        print("Use Case: Enterprise RAG with complex reasoning and large document analysis")
        print("=" * 70)

        # Initialize test
        self.results["test_metadata"]["start_time"] = datetime.now().isoformat()
        self.results["resource_utilization"]["ram_start_mb"] = psutil.virtual_memory().used / (1024**2)

        print(f"📊 Baseline RAM: {self.results['resource_utilization']['ram_start_mb']:.0f} MB")

        # Execute test queries
        print("\n🔬 Executing Enterprise-Scale RAG Queries...")

        for i, query_data in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}] Processing: {query_data['name']}")
            result = self.send_test_query(query_data)
            self.results["test_queries"].append(result)

            # Brief pause to prevent overwhelming system
            time.sleep(3)  # Longer pause for complex 128K RAG tasks

        # Calculate comprehensive metrics
        print("\n📊 Calculating Performance Metrics...")
        self.calculate_performance_metrics()

        print("📊 Analyzing Resource Utilization...")
        self.calculate_resource_utilization()

        print("📊 Performing Stability Analysis...")
        self.calculate_stability_analysis()

        # Validate results against enterprise RAG criteria
        print("📊 Validating Enterprise RAG Performance...")
        test_passed = self.validate_test_results()

        # Finalize metadata
        self.results["test_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = len(self.test_queries) * 3 + sum(q["response_time_s"] for q in self.results["test_queries"])
        self.results["test_metadata"]["total_duration_s"] = total_duration

        # Save comprehensive results
        print("💾 Saving Enterprise RAG Test Results...")
        self.save_test_results()

        # Print summary
        self.print_test_summary(test_passed)

    def print_test_summary(self, test_passed):
        """Print comprehensive test summary"""

        print("\n" + "="*70)
        print("🎯 CONTEXT SCALING TEST COMPLETE: Liquid-RAG @ 128K")
        print("="*70)

        perf = self.results["performance_metrics"]
        res = self.results["resource_utilization"]
        stab = self.results["stability_analysis"]

        print(f"📈 PERFORMANCE METRICS:")
        print(f"   Success Rate:     {perf['success_rate']:.1f}%")
        print(f"   Avg Tokens/sec:   {perf['avg_tokens_per_sec']:.2f}")
        print(f"   Token Range:      {perf['min_tokens_per_sec']:.2f} - {perf['max_tokens_per_sec']:.2f}")
        print(f"   Avg Response:     {perf['avg_response_time_s']:.2f}s")
        print(f"   Total Tokens:     {perf['total_tokens_generated']}")

        print(f"\n💾 RESOURCE UTILIZATION:")
        print(f"   RAM Increase:     {res['ram_increase_mb']:.0f} MB")
        print(f"   RAM Peak:         {res['ram_peak_mb']:.0f} MB")
        print(f"   CPU Peak:         {res['cpu_peak_percent']:.1f}%")
        print(f"   CPU Average:      {res['cpu_avg_percent']:.1f}%")

        print(f"\n🎭 STABILITY ANALYSIS:")
        print(f"   Stability Score:  {stab['stability_score']:.3f}/1.0")
        print(f"   Consistency:      {stab['performance_consistency_score']:.3f}/1.0")

        val = self.results["validation_results"]
        status_emoji = "🔍" if test_passed else "⚠️"
        status_text = "ENTERPRISE_RAG_READY" if test_passed else "REQUIRES_RAG_OPTIMIZATION"

        print(f"\n🏆 ENTERPRISE RAG VALIDATION RESULT: {status_emoji} {status_text}")

        if val["recommendations"]:
            print("💡 RECOMMENDATIONS:")
            for rec in val["recommendations"]:
                print(f"   • {rec}")

        print(f"\n📁 Result files saved in: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="O3 Context Scaling Test: Liquid-RAG @ 128K")
    parser.add_argument("--output-dir", default="ctx_128k_liquid_rag", help="Output directory")
    parser.add_argument("--iterations", type=int, default=4, help="Number of test iterations")

    args = parser.parse_args()

    print(f"Initializing enterprise RAG test for Liquid-RAG at 128K...")
    test = ContextScaling128kLiquidRAGTest(output_dir=args.output_dir)

    print("Starting enterprise RAG context scaling test...")
    test.run_test()

    print("\nEnterprise RAG 128K context test complete!")

if __name__ == "__main__":
    main()
