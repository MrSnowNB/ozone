# O3 Multi-Model Context Scaling Test Suite

**AI-First Test Suite Documentation**
**Purpose:** Systematically test and optimize other recommended models (non-qwen3-coder:30b) across 64K, 128K, and 256K contexts using binary search + statistical validation framework.

---

## 🚀 EXECUTIVE SUMMARY (AI-Parsable)

**Test Suite Status:** PLANNING
**Framework:** AI-First Optimization (Binary Search + Statistical Validation)
**Hardware Target:** Ryzen 16-core (32 logical), 127GB RAM, CPU-only inference
**Baseline:** qwen3-coder:30b at 256K context (5.13 tok/s, 10.5s TTFT, 24.5GB RAM increase)
**Goal:** Produce optimized configurations for production VS Code agentic workflows

**Key Stakeholders:**
- **Models:** `orieg/gemma3-tools:27b-it-qat`, `liquid-rag:latest`, `qwen2.5:3b-instruct`, `gemma3:latest`
- **Contexts:** 65,536 | 131,072 | 262,144 tokens
- **Metrics:** TTFT, Tokens/sec, RAM/CPU utilization, Stability score (0-1.0)

---

## 📋 TEST MATRIX OVERVIEW

### Model Classification & Context Targets

| Model Category | Models | Context Progression | Target Use Case |
|----------------|--------|---------------------|-----------------|
| **Large Coding** | `orieg/gemma3-tools:27b-it-qat` | 64K → 128K → 256K | Agentic code analysis & refactoring |
| **RAG Systems** | `liquid-rag:latest` | 64K → 128K → 256K | Document retrieval & reasoning |
| **Chat/Instruct** | `qwen2.5:3b-instruct` `gemma3:latest` | 32K → 64K → 128K | Balanced conversation (256K if viable) |

### Context Scaling Strategy Per Model

**64K Base Layer:** All models tested for baseline compatibility
**128K Enhanced:** Primary target for most agentic workflows (+25-50% context expansion)
**256K Extreme:** Limited to large models, maximum depth analysis (4x typical limits)

---

## 🏗️ PHASE 1: CONTEXT SCALING TESTS

**Phase Status:** SCRIPTS_CREATED & VALIDATED ✅
**Methodology:** Context-locked single configuration tests → performance benchmarking
**Validation:** AI-First framework validation passed (syntax, imports, hardware monitoring)
**Inspiration:** Based on `ctx_256k_locked` and `ctx_256k_full_cpu` architectures

### 1.1 Gemma3-Tools-27B Context Scaling

**Model:** `orieg/gemma3-tools:27b-it-qat` _(27B Large Coding Model)_
**Test Category:** Large Codebase Analysis & Multi-File Refactoring
**Hardware Utilization Target:** ~20GB RAM increase, 4-6 tok/s performance

#### Context Level: 64K
- **Status:** ✅ SCRIPT_CREATED & VALIDATED
- **Test Directory:** `ctx_64k_gemma3_tools_27b/`
- **Config:** `num_ctx: 65536, batch: 8, num_thread: 16, f16_kv: true`
- **Expected Performance:** Based on qwen3-coder:30b scaling formulas (target >4 tok/s)
- **Agentic Workload:** Basic codebase overview, architectural analysis
- **Script:** `multi_model_context_tests/ctx_64k_gemma3_tools_27b.py`

#### Context Level: 128K
- **Status:** ✅ SCRIPT_CREATED & VALIDATED
- **Test Directory:** `ctx_128k_gemma3_tools_27b/`
- **Config:** `num_ctx: 131072, batch: 8, num_thread: 16, f16_kv: true`
- **Expected Performance:** Production-ready for complex refactoring (target >3.5 tok/s)
- **Validation:** Sustained agentic workload testing with enterprise-scale analysis
- **Script:** `multi_model_context_tests/ctx_128k_gemma3_tools_27b.py`

#### Context Level: 256K
- **Status:** ⏳ PENDING
- **Test Directory:** `ctx_256k_gemma3_tools_27b/`
- **Config:** `num_ctx: 262144, batch: 8, num_thread: 16, f16_kv: true`
- **Expected Performance:** Extreme context depth, multi-hour code analysis (~2-3 tok/s)
- **Risk Assessment:** Hardware boundary testing (RAM: ~25GB increase required)

**Gemma3-Tools Success Criteria:**
- ✅ **64K:** TTFT < 8s, tok/s > 6, Stability > 0.8
- ✅ **128K:** TTFT < 12s, tok/s > 4, Stability > 0.75 (PRODUCTION TARGET)
- ✅ **256K:** TTFT < 20s, tok/s > 2.5, Stability > 0.7 (EXTREME USE CASE)

### 1.2 Liquid-RAG Context Scaling

**Model:** `liquid-rag:latest` _(Specialized RAG Architecture)_
**Test Category:** Retrieval-Augmented Generation & Document Analysis
**Hardware Target:** Efficient context utilization, fast retrieval performance

#### Context Level: 64K - Document Chunking
- **Status:** ⏳ PENDING
- **Test Directory:** `ctx_64k_liquid_rag/`
- **Config:** RAG-optimized settings + 64K context
- **Workload:** Multi-document retrieval, synthesis tasks

#### Context Level: 128K - Large Document Analysis
- **Status:** ⏳ PENDING
- **Test Directory:** `ctx_128k_liquid_rag/`
- **Config:** `num_ctx: 131072` + RAG optimizations
- **Validation:** Production RAG pipeline simulation

#### Context Level: 256K - Corpus-Level Reasoning
- **Status:** ⏳ PENDING
- **Test Directory:** `ctx_256k_liquid_rag/`
- **Config:** Maximum context for comprehensive knowledge synthesis
- **Success Criteria:** Superior RAG performance vs direct prompting

### 1.3 Chat/Instruct Models Context Scaling

**Models:** `qwen2.5:3b-instruct`, `gemma3:latest` _(3B-8B Efficient Models)_
**Test Category:** Conversational Agents & Instruction Following
**Hardware Target:** RAM-efficient, fast TTFT prioritized over extreme context

#### Baseline: 32K Context
- **Status:** ⏳ PENDING
- **Test Directory:** `ctx_32k_chat_models/`
- **Models:** Both qwen2.5:3b-instruct and gemma3:latest at 32K context

#### Enhanced: 64K Context
- **Status:** ⏳ PENDING
- **Test Directory:** `ctx_64k_chat_models/`
- **Primary Target:** Real-world conversational depth

#### Extended: 128K Context (256K if Viable)
- **Status:** ⏳ PENDING
- **Test Directory:** `ctx_128k_chat_models/`
- **Validation:** Sustained multi-turn conversations with full context

---

## 🧪 PHASE 2: AGENTIC WORKLOAD VALIDATION

**Phase Status:** BLOCKED (Dependent on Phase 1)
**Methodology:** Sustained testing mimicking real VS Code agentic usage
**Inspiration:** Based on `sustained_256k_agentic_test.py` and `final_stress_test_256k.py`

### 2.1 Workload Categories by Model Type

#### Large Coding Models (`gemma3-tools:27b-it-qat`)
**Phase:** Code Analysis & Refactoring (Large Context)
- Architecture analysis (85K+ token codebases)
- Multi-file refactoring scenarios
- Complex debugging sessions
- Documentation generation

**Duration:** 10-15 minutes sustained testing
**Success Metric:** 95% query success, 3.5+ tok/s average

#### RAG Models (`liquid-rag:latest`)
**Phase:** Knowledge Synthesis & Retrieval (Variable Context)
- Multi-document question answering
- Comparative analysis tasks
- Research synthesis workflows
- Citation and evidence gathering

**Duration:** 8-12 minutes specialized testing
**Success Metric:** Superior RAG performance, context efficiency

#### Chat/Instruct Models (3B-8B)
**Phase:** Conversational Continuity (Balanced Context)
- Multi-turn technical discussions
- Progressive problem solving
- Code review conversations
- Tutorial/explanation threads

**Duration:** 5-10 minutes interactive testing
**Success Metric:** Response quality maintained, conversation coherence

### 2.2 Hardware Stress Testing

**Maximum Load Validation:**
- Each model + optimal context combination
- Sustained utilization under load
- Memory stability verification
- Production readiness confirmation

---

## 📊 PHASE 3: COMPARATIVE ANALYSIS & RECOMMENDATIONS

**Phase Status:** BLOCKED (Dependent on Phases 1-2)
**Deliverables:** Cross-model performance matrix, use case recommendations

### 3.1 Performance Matrix Generation

**Metrics Tracked:**
```
Model | Context | TTFT (ms) | Tok/s | RAM Δ (GB) | Stability | Use Case Fit
gemma3-tools:27b | 64K | XXX | XXX | XXX | X.X | Basic coding
gemma3-tools:27b | 128K | XXX | XXX | XXX | X.X | Advanced coding [TARGET]
gemma3-tools:27b | 256K | XXX | XXX | XXX | X.X | Extreme coding
liquid-rag:latest | 128K | XXX | XXX | XXX | X.X | RAG applications [TARGET]
qwen2.5:3b-instruct | 64K | XXX | XXX | XXX | X.X | Chat/instruct [TARGET]
gemma3:latest | 64K | XXX | XXX | XXX | X.X | Chat/instruct [TARGET]
```

### 3.2 Use Case Recommendations

**VS Code Agentic Workflows:**
- **Basic Code Tasks:** gemma3:latest at 32K context
- **Complex Refactoring:** gemma3-tools:27b-it-qat at 128K context
- **Maximum Analysis:** gemma3-tools:27b-it-qat at 256K context
- **RAG Integration:** liquid-rag:latest at 128K context
- **Chat Assistance:** qwen2.5:3b-instruct at 64K context

---

## 🔧 PHASE 4: IMPLEMENTATION FRAMEWORK

**Phase Status:** READY
**Dependencies:** O3 optimizer framework, test harness scripts

### 4.1 Test Script Templates

**Context Scaling Test Template:**
```python
# Based on ctx_256k_locked/test_config.py
class ContextScalingTest:
    def __init__(model, context_size, output_dir):
        # Single config test framework
        # Binary search optimization
        # Statistical validation
```

**Agentic Workload Test Template:**
```python
# Based on sustained_256k_agentic_test.py
class AgenticWorkloadTest:
    def __init__(model, context_config, workload_type):
        # Sustained testing framework
        # Hardware monitoring
        # Stability validation
```

### 4.2 Automated Test Generation

**Configuration Generation:**
- YAML preset creation for each model+context
- Hardware-aware parameter optimization
- Batch size and thread optimization

**Result Processing:**
- JSON summaries for bot integration
- Markdown reports for human review
- CSV data for analysis/comparison

---

## 📈 SUCCESS CRITERIA & QUALITY GATES

### Individual Model Validation
- ✅ **Context Compatibility:** Successfully loads and operates at target context
- ✅ **Performance Baseline:** TTFT < 30s, tok/s > 1, Stability > 0.5
- ✅ **Hardware Safety:** Operates within RAM limits (max 80% utilization)
- ✅ **Production Viability:** Sustained operation for 5+ minutes under load

### Test Suite Completion
- ✅ **Full Coverage:** All 4 models tested across appropriate contexts
- ✅ **Comparative Data:** Performance matrix with clear recommendations
- ✅ **Documentation:** AI-first format, bot-ready outputs
- ✅ **Integration Ready:** VS Code agentic workflow configurations

---

## 🗓️ TIMELINE & MILESTONES

**Week 1 (Current): Phase 1 Planning & Setup**
- Test harness adaptation for new models
- Context scaling test framework creation
- Baseline hardware monitoring calibration

**Week 2: Phase 1 Execution (Context Scaling)**
- Gemma3-tools-27B: 64K → 128K → 256K progression
- Liquid-RAG: 64K → 128K → 256K progression
- Chat models: 32K → 64K → 128K progression

**Week 3: Phase 2 Execution (Workload Validation)**
- Per-model agentic workload testing
- Sustained performance validation
- Hardware stress testing completion

**Week 4: Phase 3 Analysis & Documentation**
- Comparative performance matrix
- Use case recommendations
- Final AI-first documentation completion

---

## 🔗 DEPENDENCIES & PREREQUISITES

### Software Requirements
- ✅ **Ollama Models:** All target models pulled and verified
- ✅ **Python Environment:** Test suite dependencies installed
- ✅ **Hardware Monitoring:** psutil, system resource tracking
- ✅ **O3 Framework:** Optimizer scripts and test harnesses

### Model Availability Verification
- ✅ `orieg/gemma3-tools:27b-it-qat` - PULLED (18GB, 41 hours ago)
- ✅ `liquid-rag:latest` - PULLED (730MB, 2 weeks ago)
- ✅ `qwen2.5:3b-instruct` - PULLED (1.9GB, 4 weeks ago)
- ✅ `gemma3:latest` - PULLED (3.3GB, 4 weeks ago)

### Hardware Readiness
- ✅ **RAM:** 127GB total, minimum 64GB free for testing
- ✅ **CPU:** 16 physical cores (32 logical, optimized for physical)
- ✅ **Storage:** Adequate space for test result logs/summaries
- ✅ **Network:** Reliable Ollama API access (localhost:11434)

---

## 📁 ARTIFACT ORGANIZATION

### Directory Structure
```
multi_model_context_tests/
├── gemma3_tools_27b/
│   ├── ctx_64k_gemma3_tools_27b/
│   ├── ctx_128k_gemma3_tools_27b/
│   └── ctx_256k_gemma3_tools_27b/
├── liquid_rag/
│   ├── ctx_64k_liquid_rag/
│   ├── ctx_128k_liquid_rag/
│   └── ctx_256k_liquid_rag/
├── chat_models/
│   ├── ctx_32k_chat_models/
│   ├── ctx_64k_chat_models/
│   └── ctx_128k_chat_models/
├── agentic_validation/
│   ├── gemma3_tools_agentic_test/
│   ├── liquid_rag_agentic_test/
│   └── chat_models_agentic_test/
└── comparative_analysis/
    ├── performance_matrix.json
    ├── use_case_recommendations.md
    └── integration_configs/
```

### Deliverable Formats
- **🔍 JSON Summaries:** Bot-parsable performance data
- **📊 Markdown Reports:** Human-readable analysis
- **⚙️ YAML Configurations:** Production-ready presets
- **📈 CSV Datasets:** Comparative performance tracking

---

## 🤖 AI-First Optimizations

### Automated Analysis Features
- **Performance Prediction:** Estimate results based on hardware + model characteristics
- **Smart Parameter Selection:** Binary search optimization for each model type
- **Comparative Assessment:** Automated model recommendation engine
- **Stability Scoring:** Data-driven production readiness assessment

### Integration Readiness
- **VS Code Plugin Configs:** Direct integration presets
- **API Endpoint Templates:** Agentic workflow examples
- **Monitoring Dashboards:** Real-time performance tracking
- **Automated Retraining:** Continuous optimization triggers

---

**Document Version:** 1.0
**Last Updated:** 2025-10-15
**Framework Status:** AI-First Test Suite Planning Complete
**Next Action:** Begin Phase 1 implementation
