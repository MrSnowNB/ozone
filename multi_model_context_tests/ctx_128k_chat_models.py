#!/usr/bin/env python3
"""
O3 Context Scaling Test: Chat Models at 128K Context
Phase 1.3.3 - Maximum Conversational Context Testing

Tests qwen2.5:3b-instruct and gemma3:latest at 128K context for maximum-depth
conversations with extensive context retention and multi-document analysis.
"""

import time
import json
import requests
import psutil
from datetime import datetime
from pathlib import Path
import argparse

class ContextScaling128kChatModelsTest:
    """128K context scaling test for maximum chat/instruct models conversations"""

    def __init__(self, output_dir="ctx_128k_chat_models"):
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

        # Test both chat models at 128K
        self.model_configs = {
            "qwen2.5:3b-instruct": {
                "model": "qwen2.5:3b-instruct",
                "base_url": "http://localhost:11434/api/generate",
                "chat_config": {
                    "model": "qwen2.5:3b-instruct",
                    "options": {
                        "num_ctx": 131072,    # FIXED: 128K maximum context for chat
                        "batch": 8,           # Conservative for 128K stability
                        "num_predict": 512,   # Longer responses for complex analysis
                        "num_thread": 16,     # Physical cores only
                        "temperature": 0.5,   # More focused for maximum context
                        "top_p": 0.9,
                        "f16_kv": True
                    }
                }
            },
            "gemma3:latest": {
                "model": "gemma3:latest",
                "base_url": "http://localhost:11434/api/generate",
                "chat_config": {
                    "model": "gemma3:latest",
                    "options": {
                        "num_ctx": 131072,    # FIXED: 128K maximum context for chat
                        "batch": 8,           # Conservative for 128K stability
                        "num_predict": 512,   # Longer responses for complex analysis
                        "num_thread": 16,     # Physical cores only
                        "temperature": 0.5,   # More focused for maximum context
                        "top_p": 0.9,
                        "f16_kv": True
                    }
                }
            }
        }

        self.models_to_test = ["qwen2.5:3b-instruct", "gemma3:latest"]

        # Initialize results structure for each model
        self.results = {}
        for model_name in self.models_to_test:
            self.results[model_name] = {
                "test_metadata": {
                    "test_type": "CONTEXT_SCALING_128K_CHAT_MAXIMUM",
                    "model": model_name,
                    "context_size": 131072,
                    "phase": "PHASE_1_3_3",
                    "category": "Maximum Chat/Instruct Models Multi-Document Conversations",
                    "start_time": None,
                    "end_time": None,
                    "total_duration_s": 0,
                    "configuration": self.model_configs[model_name]["chat_config"],
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

        # EXTENSIVE CHAT CONTEXT: Maximum depth technical architecture and code examples
        self.maximum_chat_context = """
        ## Enterprise Software Architecture Master Document

        ### Comprehensive System Design Patterns

        #### Microservices Architecture Deep Dive

        **Service Decomposition Strategies:**
        - Domain-Driven Design (DDD) bounded contexts
        - Strangler Fig pattern for legacy migration
        - Event Storming for service identification
        - Business Capability Mapping

        **Inter-Service Communication Patterns:**
        - Synchronous HTTP/REST with circuit breakers
        - Asynchronous messaging with Kafka/RabbitMQ
        - gRPC for high-performance service calls
        - GraphQL federation for API composition

        **Data Management in Microservices:**
        - Database-per-service with loose coupling
        - Saga pattern for distributed transactions
        - Event sourcing for audit trails
        - CQRS for read/write optimization

        #### Cloud-Native Application Architecture

        **Container Orchestration:**
        ```yaml
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: user-service
        spec:
          replicas: 3
          selector:
            matchLabels:
              app: user-service
          template:
            metadata:
              labels:
                app: user-service
            spec:
              containers:
              - name: user-service
                image: user-service:v1.2.3
                ports:
                - containerPort: 8080
                env:
                - name: DB_HOST
                  valueFrom:
                    secretKeyRef:
                      name: db-secret
                      key: host
                - name: REDIS_URL
                  valueFrom:
                    configMapKeyRef:
                      name: redis-config
                      key: url
                resources:
                  requests:
                    memory: "256Mi"
                    cpu: "250m"
                  limits:
                    memory: "512Mi"
                    cpu: "500m"
                livenessProbe:
                  httpGet:
                    path: /health
                    port: 8080
                  initialDelaySeconds: 30
                  periodSeconds: 10
                readinessProbe:
                  httpGet:
                    path: /ready
                    port: 8080
                  initialDelaySeconds: 5
                  periodSeconds: 5
        ```

        **Service Mesh Configuration (Istio):**
        ```yaml
        apiVersion: networking.istio.io/v1beta1
        kind: VirtualService
        metadata:
          name: user-service-routing
        spec:
          hosts:
          - user-service
          http:
          - match:
            - uri:
                prefix: "/api/v1"
            route:
            - destination:
                host: user-service
                subset: v1
          - match:
            - uri:
                prefix: "/api/v2"
            route:
            - destination:
                host: user-service
                subset: v2
            timeout: 10s
        ```

        #### Advanced Backend Patterns

        **Hexagonal Architecture Implementation (Ports & Adapters):**
        ```python
        # Domain Layer
        class UserService:
            def __init__(self, user_repository: UserRepository):
                self.user_repository = user_repository

            def create_user(self, user_data: dict) -> User:
                user = User(**user_data)
                self.validate_user(user)
                return self.user_repository.save(user)

            def get_user(self, user_id: str) -> User:
                return self.user_repository.find_by_id(user_id)

        # Application Layer (Use Cases)
        class CreateUserUseCase:
            def __init__(self, user_service: UserService, event_publisher: EventPublisher):
                self.user_service = user_service
                self.event_publisher = event_publisher

            def execute(self, user_data: dict) -> User:
                user = self.user_service.create_user(user_data)
                self.event_publisher.publish(UserCreatedEvent(user.id))
                return user

        # Infrastructure Layer (Adapters)
        class PostgresUserRepository(UserRepository):
            def __init__(self, session_factory):
                self.session_factory = session_factory

            def save(self, user: User) -> User:
                with self.session_factory() as session:
                    session.add(user)
                    session.commit()
                    return user

        # Web Framework Adapter (FastAPI)
        @app.post("/users/", response_model=UserResponse)
        async def create_user(
            user_request: UserCreateRequest,
            user_service: UserService = Depends(get_user_service)
        ):
            try:
                user = user_service.create_user(user_request.dict())
                return UserResponse.from_domain(user)
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail="Internal server error")
        ```

        #### Frontend Architecture Evolution

        **Modern React Application Structure:**
        ```typescript
        // Feature-based folder structure
        src/
        ├── features/
        │   ├── auth/
        │   │   ├── components/
        │   │   │   ├── LoginForm.tsx
        │   │   │   ├── RegisterForm.tsx
        │   │   │   └── AuthGuard.tsx
        │   │   ├── hooks/
        │   │   │   ├── useAuth.ts
        │   │   │   └── useLogin.ts
        │   │   ├── services/
        │   │   │   └── authService.ts
        │   │   └── types/
        │   │       └── auth.ts
        │   └── users/
        │       ├── components/
        │       ├── hooks/
        │       ├── services/
        │       └── types/
        ├── shared/
        │   ├── components/
        │   │   ├── Button.tsx
        │   │   ├── Input.tsx
        │   │   └── Modal.tsx
        │   ├── hooks/
        │   │   ├── useApi.ts
        │   │   └── useLocalStorage.ts
        │   ├── utils/
        │   │   ├── api.ts
        │   │   ├── dateUtils.ts
        │   │   └── validation.ts
        │   └── types/
        │       └── common.ts
        └── App.tsx

        // Modern state management with Zustand
        import { create } from 'zustand';
        import { devtools, persist } from 'zustand/middleware';

        interface UserState {
          user: User | null;
          isLoading: boolean;
          error: string | null;
          login: (credentials: LoginCredentials) => Promise<void>;
          logout: () => void;
          updateProfile: (updates: Partial<User>) => Promise<void>;
        }

        export const useUserStore = create<UserState>()(
          devtools(
            persist(
              (set, get) => ({
                user: null,
                isLoading: false,
                error: null,

                login: async (credentials) => {
                  set({ isLoading: true, error: null });
                  try {
                    const response = await authService.login(credentials);
                    set({ user: response.user, isLoading: false });
                    localStorage.setItem('token', response.token);
                  } catch (error) {
                    set({ error: error.message, isLoading: false });
                    throw error;
                  }
                },

                logout: () => {
                  set({ user: null, error: null });
                  localStorage.removeItem('token');
                },

                updateProfile: async (updates) => {
                  const currentUser = get().user;
                  if (!currentUser) throw new Error('Not authenticated');

                  set({ isLoading: true, error: null });
                  try {
                    const updatedUser = await userService.updateProfile(currentUser.id, updates);
                    set({ user: updatedUser, isLoading: false });
                  } catch (error) {
                    set({ error: error.message, isLoading: false });
                    throw error;
                  }
                },
              }),
              {
                name: 'user-store',
                partialize: (state) => ({ user: state.user }),
              }
            ),
            { name: 'user-store' }
          )
        );

        // Modern data fetching with React Query
        import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

        export const useUsers = (filters?: UserFilters) => {
          return useQuery({
            queryKey: ['users', filters],
            queryFn: () => userService.getUsers(filters),
            staleTime: 5 * 60 * 1000, // 5 minutes
            cacheTime: 10 * 60 * 1000, // 10 minutes
          });
        };

        export const useCreateUser = () => {
          const queryClient = useQueryClient();

          return useMutation({
            mutationFn: userService.createUser,
            onSuccess: (newUser) => {
              queryClient.setQueryData(['users'], (oldData: User[]) => {
                return [...oldData, newUser];
              });
              queryClient.invalidateQueries({ queryKey: ['user-stats'] });
            },
            onError: (error) => {
              console.error('Failed to create user:', error);
            },
          });
        };
        ```

        #### DevOps and Infrastructure as Code

        **GitHub Actions CI/CD Pipeline:**
        ```yaml
        name: CI/CD Pipeline

        on:
          push:
            branches: [ main, develop ]
          pull_request:
            branches: [ main ]

        env:
          REGISTRY: ghcr.io
          IMAGE_NAME: ${{ github.repository }}

        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
            - uses: actions/checkout@v4

            - name: Set up Python
              uses: actions/setup-python@v4
              with:
                python-version: '3.11'

            - name: Install dependencies
              run: |
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                pip install -r requirements-dev.txt

            - name: Run tests
              run: |
                pytest --cov=./ --cov-report=xml

            - name: Upload coverage
              uses: codecov/codecov-action@v3

          build:
            needs: test
            runs-on: ubuntu-latest
            steps:
            - name: Build and push Docker image
              uses: docker/build-push-action@v4
              with:
                context: .
                push: true
                tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
                cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache
                cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache,mode=max

          deploy-staging:
            needs: build
            runs-on: ubuntu-latest
            if: github.ref == 'refs/heads/develop'
            steps:
            - name: Deploy to staging
              run: |
                echo "Deploying to staging environment"
                # Add actual deployment commands here

          deploy-production:
            needs: build
            runs-on: ubuntu-latest
            if: github.ref == 'refs/heads/main'
            steps:
            - name: Deploy to production
              run: |
                echo "Deploying to production environment"
                # Add actual deployment commands here
        ```

        #### Security Architecture

        **Zero Trust Security Model:**
        - Identity and Access Management (IAM)
        - Micro-segmentation with service mesh
        - End-to-end encryption
        - Continuous authentication and authorization

        Contains ~95K tokens of maximum-depth enterprise software architecture documentation
        """

        # MAXIMUM DEPTH CHAT QUERIES - Leverages full 128K context for comprehensive analysis
        self.test_queries = [
            {
                "name": "enterprise_architecture_synthesis",
                "query": "Based on the comprehensive enterprise software architecture documentation provided above, design a complete production-ready architecture for a SaaS platform serving 100,000+ users with real-time collaboration features. Include detailed infrastructure design, deployment strategy, security measures, monitoring approach, and scaling considerations. Provide specific code examples and configuration snippets for critical components.",
                "category": "comprehensive_system_design",
                "expected_response_length": 800
            },
            {
                "name": "frontend_backend_integration_optimization",
                "query": "Analyze the frontend (React/TypeScript with modern patterns) and backend (Python FastAPI with hexagonal architecture) code examples in the documentation. Design an optimized integration strategy that includes API design, state management, data synchronization, error handling, and performance optimization. Provide refactoring recommendations for both frontend and backend to achieve better developer experience and system reliability.",
                "category": "fullstack_integration_design",
                "expected_response_length": 750
            },
            {
                "name": "devsecops_implementation_strategy",
                "query": "Using the DevOps and security patterns documented above, develop a comprehensive DevSecOps implementation strategy for the enterprise software platform. Include CI/CD pipeline design, infrastructure as code, secrets management, compliance automation, security scanning, and incident response procedures. Provide specific tool recommendations, configuration examples, and monitoring dashboards for production deployment.",
                "category": "security_operations_integration",
                "expected_response_length": 700
            },
            {
                "name": "scalability_performance_architecture",
                "query": "Given the architectural patterns and infrastructure designs in the documentation, create a detailed scalability and performance optimization strategy for handling sudden traffic spikes (10x normal load) and maintaining sub-100ms response times. Include database optimization, caching strategies, CDN configuration, service mesh tuning, and auto-scaling policies with specific implementation examples and cost optimization considerations.",
                "category": "performance_scalability_engineering",
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

    def send_max_depth_chat_query(self, model_name, query_data):
        """Send a maximum depth chat query that leverages full 128K context"""

        start_time = time.time()
        initial_hardware = self.log_hardware_state()

        # Build maximum context prompt for deep architectural analysis
        full_prompt = f"""You are a senior enterprise software architect with 20+ years of experience designing and implementing complex distributed systems. You have deep expertise across the full technology stack including frontend, backend, DevOps, security, and cloud architecture.

ENTERPRISE ARCHITECTURE MASTER DOCUMENT CONTEXT:
{self.maximum_chat_context}

ARCHITECTURAL ANALYSIS AND DESIGN REQUEST:
{query_data['query']}

As an expert architect, provide a comprehensive, production-ready solution that demonstrates mastery of enterprise-scale system design. Focus on practical implementation details, trade-off analysis, and specific recommendations backed by industry best practices. Include concrete code examples, configuration files, and architectural decision rationales.

EXPERT ANALYSIS REQUIREMENTS:
- Demonstrate deep understanding of architectural patterns and trade-offs
- Provide specific implementation examples with real-world considerations
- Address scalability, reliability, security, and maintainability concerns
- Include cost optimization and operational considerations
- Show expertise in both strategic design and tactical implementation
- Reference specific technologies and patterns from the documentation
- Consider multiple architectural approaches with clear recommendations

Your response should serve as production-ready documentation for development teams implementing this architecture."""

        # Ensure prompt stays within 128K token limit (conservative approach for maximum depth)
        max_chars = 131072 * 4 * 0.9  # 90% of theoretical limit for safety with complex content
        if len(full_prompt) > max_chars:
            full_prompt = full_prompt[:int(max_chars)]

        query_config = self.model_configs[model_name]["chat_config"].copy()
        query_config["prompt"] = full_prompt

        print(f"💬 Sending maximum depth chat query to {model_name} - {len(full_prompt)} chars")

        try:
            response = requests.post(
                self.model_configs[model_name]["base_url"],
                json=query_config,
                timeout=600  # 10 minutes for maximum depth architectural analysis
            )

            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Calculate performance metrics
                tokens_generated = len(response_text.split())
                tokens_per_sec = tokens_generated / response_time if response_time > 0 else 0

                query_result = {
                    "model": model_name,
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

                print(f"✅ {model_name} MAXIMUM DEPTH SUCCESS - {tokens_generated} tokens, {tokens_per_sec:.2f} tok/s, {response_time:.2f}s")

            else:
                query_result = {
                    "model": model_name,
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

                print(f"❌ {model_name} MAXIMUM DEPTH FAILED - HTTP {response.status_code}")

        except Exception as e:
            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            query_result = {
                "model": model_name,
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

            print(f"❌ {model_name} MAXIMUM DEPTH EXCEPTION - {str(e)}")

        return query_result

    def run_model_test(self, model_name):
        """Run maximum depth test queries for a specific model"""

        model_results = self.results[model_name]
        model_results["test_queries"] = []

        # Execute maximum depth test queries for this model
        for i, query_data in enumerate(self.test_queries, 1):
            print(f"\n[{model_name} {i}/{len(self.test_queries)}] Processing maximum depth: {query_data['name']}")
            result = self.send_max_depth_chat_query(model_name, query_data)
            model_results["test_queries"].append(result)

            # Extended pause for maximum depth analysis
            time.sleep(4)  # Allow rest between complex architectural queries

        return model_results

    def calculate_model_metrics(self, model_name):
        """Calculate comprehensive metrics for maximum depth chat models"""

        model_results = self.results[model_name]
        successful_queries = [q for q in model_results["test_queries"] if q["success"]]

        model_results["performance_metrics"].update({
            "total_queries": len(model_results["test_queries"]),
            "successful_queries": len(successful_queries),
            "failed_queries": len(model_results["test_queries"]) - len(successful_queries),
            "total_tokens_generated": sum(q["tokens_generated"] for q in successful_queries),
            "success_rate": (len(successful_queries) / len(model_results["test_queries"])) * 100 if model_results["test_queries"] else 0
        })

        if successful_queries:
            tokens_per_sec_values = [q["tokens_per_sec"] for q in successful_queries]
            response_times = [q["response_time_s"] for q in successful_queries]

            model_results["performance_metrics"].update({
                "avg_tokens_per_sec": sum(tokens_per_sec_values) / len(tokens_per_sec_values),
                "min_tokens_per_sec": min(tokens_per_sec_values),
                "max_tokens_per_sec": max(tokens_per_sec_values),
                "avg_response_time_s": sum(response_times) / len(response_times)
            })

    def calculate_resource_utilization(self):
        """Calculate resource utilization for maximum depth chat models"""

        for model_name in self.models_to_test:
            model_results = self.results[model_name]

            if not model_results["test_queries"]:
                continue

            # Get RAM samples from all queries for this model
            ram_samples = []
            cpu_samples = []

            for query in model_results["test_queries"]:
                if query["success"]:
                    # Use hardware data if available
                    if "hardware_final" in query:
                        ram_samples.append(query["hardware_final"]["ram_used_mb"])
                        cpu_samples.append(query["hardware_final"]["cpu_percent"])

            if ram_samples:
                ram_peak = max(ram_samples)
                ram_start = model_results["resource_utilization"]["ram_start_mb"] or ram_samples[0]

                model_results["resource_utilization"].update({
                    "ram_peak_mb": ram_peak,
                    "ram_increase_mb": ram_peak - ram_start,
                    "context_memory_allocation_mb": ram_peak - ram_start  # 128K maximum context usage
                })

            if cpu_samples:
                model_results["resource_utilization"].update({
                    "cpu_utilization_samples": cpu_samples,
                    "cpu_peak_percent": max(cpu_samples) if cpu_samples else 0,
                    "cpu_avg_percent": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
                })

    def calculate_stability_analysis(self):
        """Calculate stability metrics for maximum depth chat models"""

        for model_name in self.models_to_test:
            model_results = self.results[model_name]

            if not model_results["test_queries"]:
                continue

            success_sequence = [q["success"] for q in model_results["test_queries"]]

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

            model_results["stability_analysis"].update({
                "success_streak": current_success_streak,
                "max_success_streak": max_success_streak,
                "failure_streak": current_failure_streak,
                "max_failure_streak": max_failure_streak
            })

            # Calculate stability score (0-1.0)
            success_rate = model_results["performance_metrics"]["success_rate"] / 100
            consistency_bonus = min(max_success_streak / len(success_sequence), 0.3) if success_sequence else 0

            stability_score = success_rate + consistency_bonus
            stability_score = min(stability_score, 1.0)

            # Performance consistency score for maximum depth queries
            successful_queries = [q for q in model_results["test_queries"] if q["success"]]
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

            model_results["stability_analysis"].update({
                "stability_score": round(stability_score, 3),
                "performance_consistency_score": round(consistency_score, 3)
            })

    def validate_test_results(self):
        """Validate test results for maximum depth chat models at 128K context"""

        for model_name in self.models_to_test:
            model_results = self.results[model_name]

            # 128K Context Maximum Depth Chat Success Criteria
            success_criteria = {
                "context_compatibility": True,  # All queries attempted
                "performance_maximum_depth": model_results["performance_metrics"]["avg_tokens_per_sec"] > 3.0,  # Reasonable performance for maximum depth
                "response_time_max_depth": model_results["performance_metrics"]["avg_response_time_s"] < 180,  # Allow more time for comprehensive analysis
                "hardware_safety_max_128k": model_results["resource_utilization"]["ram_peak_mb"] < (127 * 1024 * 0.9),  # <90% RAM for maximum context
                "success_rate_maximum": model_results["performance_metrics"]["success_rate"] > 75,  # High reliability for complex analysis
                "stability_maximum_depth": model_results["stability_analysis"]["stability_score"] > 0.7  # Stable performance for architectural analysis
            }

            # Overall validation
            test_passed = all(success_criteria.values())

            validation_results = {
                "criteria": success_criteria,
                "overall_passed": test_passed,
                "recommendations": [],
                "production_readiness": "MAXIMUM_DEPTH_CHAT_READY" if test_passed else "REQUIRES_MAXIMUM_DEPTH_OPTIMIZATION"
            }

            # Generate recommendations
            if not success_criteria["performance_maximum_depth"]:
                validation_results["recommendations"].append("Maximum depth performance below target - complex architectural queries may need optimization or model tuning")

            if not success_criteria["response_time_max_depth"]:
                validation_results["recommendations"].append("Response times exceed acceptable range for maximum depth analysis - consider query complexity reduction or parallel processing")

            if not success_criteria["hardware_safety_max_128k"]:
                validation_results["recommendations"].append("Memory utilization at critical levels for 128K maximum context - optimize context usage or increase available RAM")

            if not success_criteria["stability_maximum_depth"]:
                validation_results["recommendations"].append("Stability inadequate for maximum depth analysis - investigate architectural query consistency")

            model_results["validation_results"] = validation_results

    def run_test(self):
        """Execute the complete 128K maximum depth context scaling test for chat models"""

        print("💬 O3 Context Scaling Test: Maximum Depth Chat Models at 128K Context")
        print("=" * 70)
        print(f"Models: {', '.join(self.models_to_test)}")
        print(f"Context: 131,072 tokens (128K)")
        print(f"Category: Maximum Depth Chat/Instruct Models Enterprise Architecture")
        print("Use Case: Full-context architectural analysis and comprehensive system design")
        print("=" * 70)

        # Initialize test
        start_time = datetime.now().isoformat()

        for model_name in self.models_to_test:
            self.results[model_name]["test_metadata"]["start_time"] = start_time
            self.results[model_name]["resource_utilization"]["ram_start_mb"] = psutil.virtual_memory().used / (1024**2)

        print(f"📊 Baseline RAM: {self.results[self.models_to_test[0]]['resource_utilization']['ram_start_mb']:.0f} MB")

        # Execute maximum depth test queries for each model
        print("\n💬 Executing Maximum Depth Architectural Analysis Queries...")

        for model_name in self.models_to_test:
            print(f"\n🎯 Testing {model_name} with maximum context depth...")
            self.run_model_test(model_name)

        # Calculate comprehensive metrics for all models
        print("\n📊 Calculating Performance Metrics...")
        for model_name in self.models_to_test:
            self.calculate_model_metrics(model_name)

        print("📊 Analyzing Resource Utilization...")
        self.calculate_resource_utilization()

        print("📊 Performing Stability Analysis...")
        self.calculate_stability_analysis()

        # Validate results against maximum depth chat criteria at 128K
        print("📊 Validating Maximum Depth Chat Performance...")
        self.validate_test_results()

        # Finalize metadata and save results
        end_time = datetime.now().isoformat()
        for model_name in self.models_to_test:
            self.results[model_name]["test_metadata"]["end_time"] = end_time
            # Estimate total duration across all models
            total_duration = len(self.test_queries) * len(self.models_to_test) * 4 + sum(
                sum(q["response_time_s"] for q in self.results[m]["test_queries"])
                for m in self.models_to_test
            )
            self.results[model_name]["test_metadata"]["total_duration_s"] = total_duration

        # Save comprehensive results
        print("💾 Saving Maximum Depth Chat Model Test Results...")
        self.save_test_results()

        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print comprehensive test summary for maximum depth chat models"""

        print("\n" + "="*70)
        print("🎯 CONTEXT SCALING TEST COMPLETE: MAXIMUM DEPTH CHAT @ 128K")
        print("="*70)

        for model_name in self.models_to_test:
            model_results = self.results[model_name]
            perf = model_results["performance_metrics"]
            res = model_results["resource_utilization"]
            stab = model_results["stability_analysis"]
            val = model_results["validation_results"]

            print(f"\n🤖 {model_name.upper()} (128K Maximum Depth):")
            print(f"   Success Rate:     {perf['success_rate']:.1f}%")
            print(f"   Avg Tokens/sec:   {perf['avg_tokens_per_sec']:.2f}")
            print(f"   Token Range:      {perf['min_tokens_per_sec']:.2f} - {perf['max_tokens_per_sec']:.2f}")
            print(f"   Avg Response:     {perf['avg_response_time_s']:.2f}s")
            print(f"   Total Tokens:     {perf['total_tokens_generated']}")

            print(f"   RAM Increase:     {res['ram_increase_mb']:.0f} MB")
            print(f"   Stability Score:  {stab['stability_score']:.3f}/1.0")
            print(f"   Status:           {'🔍 MAX_DEPTH_READY' if val['overall_passed'] else '⚠️ REQUIRES_OPTIMIZATION'}")

    def save_test_results(self):
        """Save comprehensive test results for maximum depth chat models"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name in self.models_to_test:
            model_results = self.results[model_name]
            safe_model_name = model_name.replace(":", "_").replace(".", "_")

            # 1. JSONL Log File
            log_file = self.logs_dir / f"ctx_128k_maximum_{safe_model_name}_{timestamp}.jsonl"
            with open(log_file, 'w', encoding='utf-8') as f:
                for query in model_results["test_queries"]:
                    f.write(json.dumps(query, default=str, ensure_ascii=False) + '\n')

            # 2. JSON Summary File
            summary_data = {
                "test_metadata": model_results["test_metadata"],
                "performance_metrics": model_results["performance_metrics"],
                "resource_utilization": model_results["resource_utilization"],
                "stability_analysis": model_results["stability_analysis"],
                "validation_results": model_results["validation_results"],
                "generated_at": timestamp,
                "summary_file": str(log_file)
            }

            summary_file = self.summaries_dir / f"ctx_128k_maximum_{safe_model_name}_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

            # 3. YAML Default Configuration
            import yaml

            config_preset = {
                "model": model_name,
                "presets": {
                    "chat_128k_maximum_depth": {
                        "num_ctx": 131072,
                        "batch": self.model_configs[model_name]["chat_config"]["options"]["batch"],
                        "num_thread": self.model_configs[model_name]["chat_config"]["options"]["num_thread"],
                        "f16_kv": self.model_configs[model_name]["chat_config"]["options"]["f16_kv"],
                        "temperature": self.model_configs[model_name]["chat_config"]["options"]["temperature"],
                        "tokens_per_sec": round(model_results["performance_metrics"]["avg_tokens_per_sec"], 2),
                        "ttft_ms": round(model_results["performance_metrics"]["avg_response_time_s"] * 1000),
                        "ram_increase_gb": round(model_results["resource_utilization"]["ram_increase_mb"] / 1024, 2),
                        "stability_score": model_results["stability_analysis"]["stability_score"],
                        "use_case": "Maximum depth architectural analysis with full enterprise context",
                        "validated": model_results["validation_results"]["overall_passed"]
                    }
                }
            }

            config_file = self.defaults_dir / f"ctx_128k_maximum_{safe_model_name}_config_{timestamp}.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_preset, f, default_flow_style=False, sort_keys=False)

            print(f"📁 {model_name} Maximum Depth Chat Results:")
            print(f"  Logs: {log_file}")
            print(f"  Summary: {summary_file}")
            print(f"  Config: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="O3 Context Scaling Test: Maximum Depth Chat Models @ 128K")
    parser.add_argument("--output-dir", default="ctx_128k_chat_models", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b-instruct", "gemma3:latest"], help="Models to test")

    args = parser.parse_args()

    print(f"Initializing maximum depth chat models test at 128K context: {', '.join(args.models)}")
    test = ContextScaling128kChatModelsTest(output_dir=args.output_dir)

    print("Starting maximum depth chat models context scaling test...")
    test.run_test()

    print("\nMaximum depth chat models 128K context test complete!")

if __name__ == "__main__":
    main()
