import logging
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd

from vanna import Agent
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.postgres import PostgresRunner
from vanna.core.registry import ToolRegistry
from vanna.tools import RunSqlTool
from vanna.core.user import UserResolver, User, RequestContext
from vanna.integrations.local.agent_memory import DemoAgentMemory

from app.config import get_settings

logger = logging.getLogger("idop_app.vanna_service")


class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        return User(
            id="sql_service_user",
            email="sql@service.local",
            group_memberships=["user", "admin"],
        )


class VannaAgentWrapper:
    def __init__(
        self,
        openai_api_key: str,
        database_url: str,
    ):
        settings = get_settings()
        self.llm = OpenAILlmService(api_key=openai_api_key, model=settings.llm_model)

        logger.info(
            f"Configuring SQL LLM with deterministic settings: "
            f"temperature={settings.llm_temperature}"
        )

        original_build_payload = self.llm._build_payload

        def deterministic_build_payload(request):
            payload = original_build_payload(request)
            payload["temperature"] = settings.llm_temperature
            payload["top_p"] = 0.1
            payload["seed"] = 42
            payload["max_tokens"] = 2000
            logger.debug(f"SQL LLM payload: {payload}")
            return payload

        self.llm._build_payload = deterministic_build_payload
        self.postgres_runner = PostgresRunner(connection_string=database_url)

        self.tools = ToolRegistry()
        self.tools.register_local_tool(
            RunSqlTool(sql_runner=self.postgres_runner), access_groups=["admin", "user"]
        )

        self.user_resolver = SimpleUserResolver()
        self.memory = DemoAgentMemory()

        self.agent = Agent(
            llm_service=self.llm,
            tool_registry=self.tools,
            user_resolver=self.user_resolver,
            agent_memory=self.memory,
        )
        logger.info("Vanna Agent Wrapper initialized successfully")

    async def generate_sql_async(self, question: str, schema_context: str = "") -> str:
        if schema_context:
            full_message = f"{schema_context}\n\n Question: {question}"
        else:
            full_message = question
        return await self._extract_sql_from_agent(full_message)

    async def _extract_sql_from_agent(self, message: str) -> str:
        request_context = RequestContext()
        sql = None

        async for component in self.agent.send_message(
            request_context=request_context, message=message
        ):
            rich_comp = component.rich_component
            if hasattr(rich_comp, "metadata") and rich_comp.metadata:
                if "sql" in rich_comp.metadata:
                    sql = rich_comp.metadata["sql"]

            if hasattr(rich_comp, "content") and rich_comp.content:
                content = str(rich_comp.content)
                if "```sql" in content.lower():
                    parts = content.split("```")
                    for part in parts:
                        if part.strip().lower().startswith("sql"):
                            sql = part[3:].strip()

        if not sql:
            raise ValueError(
                "Agent did not generate SQL. Please try rephrasing your question"
            )
        return sql

    async def execute_sql_async(self, sql: str) -> List[Dict[str, Any]]:
        return await self._execute_and_extract_results(sql)

    async def _execute_and_extract_results(self, sql: str) -> List[Dict[str, Any]]:
        logger.info(f"Executing SQL directly: {sql[:100]}...")
        try:
            import psycopg2
            import psycopg2.extras
            import socket
            from urllib.parse import urlparse

            conn_str = self.postgres_runner.connection_string
            parsed = urlparse(conn_str)
            hostname = parsed.hostname

            try:
                logger.debug(f"Resolving hostname {hostname} to IPv4...")
                addr_info = socket.getaddrinfo(hostname, None, socket.AF_INET)
                ipv4_address = addr_info[0][4][0]
                logger.info(f"Resolved {hostname} to IPv4: {ipv4_address}")
                conn_str = conn_str.replace(hostname, ipv4_address)
            except Exception as e:
                logger.warning(f"Failed to resolve hostname to IPv4: {e}, using original hostname")

            conn = psycopg2.connect(conn_str)
            try:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(sql)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                cursor.close()
                conn.close()
                logger.info(f"SQL executed successfully: {len(results)} rows returned")
                return results
            except Exception as e:
                conn.close()
                raise e
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise ValueError(f"Failed to execute SQL: {str(e)}")


class TextToSQLService:
    def __init__(
        self,
        database_url: str | None = None,
        openai_api_key: str | None = None,
        query_cache_service=None,
    ):
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.query_cache_service = query_cache_service

        if not self.database_url:
            raise ValueError("DATABASE_URL is required for Text-to-SQL features.")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for Text-to-SQL features.")

        self.vanna = VannaAgentWrapper(
            openai_api_key=self.openai_api_key,
            database_url=self.database_url,
        )
        self.pending_queries: Dict[str, Dict[str, Any]] = {}
        self.is_trained = False
        self.schema_context = ""

    def complete_training(self):
        logger.info("Preparing schema context for Vanna 2.0...")
        self.schema_context = self._build_schema_context()
        self.is_trained = True
        logger.info("Schema prepared for Vanna 2.0 Agent!")

    def _build_schema_context(self) -> str:
        schema_parts = []
        schema_parts.append("DATABASE SCHEMA DOCUMENTATION")
        schema_parts.append("=" * 60)

        documentation = """
    This is an e-commerce database with three main tables:
    - customers: Contains customer information including name, email, segment (SMB, Enterprise, Individual), and country
    - products: Product catalog with name, category, price, stock quantity, and description
    - orders: Customer orders with order date, total amount, status (Pending, Delivered, Cancelled, Processing), and shipping address

    The customers table has a one-to-many relationship with orders (one customer can have many orders).

    IMPORTANT NOTES:
    - For order revenue/pricing, use orders.total_amount (NOT 'price')
    - Customer segments: 'SMB', 'Enterprise', 'Individual' (case-sensitive)
    - Order statuses: 'Pending', 'Delivered', 'Cancelled', 'Processing' (case-sensitive)
    - To join customers and orders: JOIN orders ON customers.id = orders.customer_id
    """
        schema_parts.append(documentation)
        schema_parts.append("\nTABLE SCHEMAS:")
        schema_parts.append("-" * 60)

        schema_parts.append("""
    Table: customers
    Columns:
    - id (SERIAL PRIMARY KEY)
    - name (VARCHAR) - Customer full name
    - email (VARCHAR) - Customer email address
    - segment (VARCHAR) - One of: 'SMB', 'Enterprise', 'Individual'
    - country (VARCHAR) - Customer country
    - created_at (TIMESTAMP)
    - updated_at (TIMESTAMP)
    """)

        schema_parts.append("""
    Table: products
    Columns:
    - id (SERIAL PRIMARY KEY)
    - name (VARCHAR) - Product name
    - category (VARCHAR) - Product category (Electronics, Software, Hardware, etc.)
    - price (DECIMAL) - Product unit price
    - stock_quantity (INT) - Current inventory count
    - description (TEXT)
    - created_at (TIMESTAMP)
    - updated_at (TIMESTAMP)
    """)

        schema_parts.append("""
    Table: orders
    Columns:
    - id (SERIAL PRIMARY KEY)
    - customer_id (INT) - Foreign key to customers.id
    - order_date (DATE) - Date of order
    - total_amount (DECIMAL) - TOTAL ORDER PRICE (use this for revenue, NOT 'price'!)
    - status (VARCHAR) - One of: 'Pending', 'Delivered', 'Cancelled', 'Processing'
    - shipping_address (TEXT)
    - created_at (TIMESTAMP)
    - updated_at (TIMESTAMP)
    """)

        schema_parts.append("\nEXAMPLE QUERIES:")
        schema_parts.append("-" * 60)

        examples = [
            (
                "How many customers do we have?",
                "SELECT COUNT(*) as customer_count FROM customers;",
            ),
            (
                "What is the total revenue from all orders?",
                "SELECT SUM(total_amount) as total_revenue FROM orders;",
            ),
            (
                "List all delivered orders",
                "SELECT * FROM orders WHERE status = 'Delivered' ORDER BY order_date DESC;",
            ),
            (
                "How many orders per customer segment?",
                "SELECT c.segment, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.segment;",
            ),
            (
                "Top 10 customers by total spending",
                "SELECT c.name, c.email, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name, c.email ORDER BY total_spent DESC LIMIT 10;",
            ),
        ]

        for i, (question, sql) in enumerate(examples, 1):
            schema_parts.append(f"\nExample {i}:")
            schema_parts.append(f"Question: {question}")
            schema_parts.append(f"SQL: {sql}")

        return "\n".join(schema_parts)

    async def generate_sql_for_approval(self, question: str) -> Dict[str, Any]:
        settings = get_settings()
        if not self.is_trained:
            self.complete_training()

        if self.query_cache_service and self.query_cache_service.enabled:
            cache_key = self.query_cache_service.get_sql_gen_key(question)
            cached_result = self.query_cache_service.get(cache_key, cache_type="sql_gen")

            if cached_result and "sql" in cached_result:
                logger.info(f"SQL generation cache HIT for question: '{question[:50]}...'")
                query_id = str(uuid.uuid4())
                self.pending_queries[query_id] = {
                    "question": question,
                    "sql": cached_result["sql"],
                    "status": "pending_approval",
                    "generated_at": pd.Timestamp.now().isoformat(),
                    "cache_hit": True,
                }
                return {
                    "query_id": query_id,
                    "question": question,
                    "sql": cached_result["sql"],
                    "explanation": cached_result.get(
                        "explanation",
                        "This SQL will retrieve data from your database. Please review before approving.",
                    ),
                    "status": "pending_approval",
                    "generated_at": pd.Timestamp.now().isoformat(),
                    "cache_hit": True,
                    "cost_saved": "$0.08",
                }

        try:
            try:
                sql = await self.vanna.generate_sql_async(
                    question=question, schema_context=self.schema_context
                )
                explanation = "This SQL will retrieve data from your database. Please review before approving."
            except Exception as vanna_err:
                logger.warning(f"Vanna SQL generation failed: {vanna_err}. Falling back to direct LLM SQL generation...")
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                prompt = f"""
You are a senior database administrator.
Generate a PostgreSQL SQL query to answer the user question.
Use the schema context provided.

{self.schema_context}

Question: {question}

Respond strictly with the SQL query in a markdown code block starting with ```sql and ending with ```.
Do not include any additional text outside the code block.
"""
                response = client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                # Extract SQL
                sql = None
                if "```sql" in content.lower():
                    parts = content.split("```")
                    for part in parts:
                        if part.strip().lower().startswith("sql"):
                            sql = part[3:].strip()
                if not sql:
                    sql = content.strip()
                
                logger.info(f"Direct LLM SQL generation fallback succeeded! Generated SQL: {sql[:100]}...")
                explanation = "⚠️ Direct LLM Fallback: Generated directly using OpenAI GPT-4o as the core Vanna agent was unavailable."

            if self.query_cache_service and self.query_cache_service.enabled:
                cache_key = self.query_cache_service.get_sql_gen_key(question)
                cache_value = {
                    "sql": sql,
                    "explanation": explanation,
                    "question": question,
                }
                ttl = settings.cache_ttl_sql_gen
                self.query_cache_service.set(
                    cache_key, cache_value, ttl=ttl, cache_type="sql_gen"
                )
                logger.info(f"SQL generation cache MISS - cached for '{question[:50]}...' (TTL: {ttl}s)")

            query_id = str(uuid.uuid4())
            self.pending_queries[query_id] = {
                "question": question,
                "sql": sql,
                "status": "pending_approval",
                "generated_at": pd.Timestamp.now().isoformat(),
                "cache_hit": False,
            }

            return {
                "query_id": query_id,
                "question": question,
                "sql": sql,
                "explanation": explanation,
                "status": "pending_approval",
                "cache_hit": False,
                "cost_saved": "$0.00",
            }
        except Exception as e:
            raise Exception(f"Failed to generate SQL: {str(e)}")

    async def execute_approved_query(
        self, query_id: str, approved: bool
    ) -> dict[str, Any]:
        settings = get_settings()
        if query_id not in self.pending_queries:
            return {"error": "Query ID not found", "status": "error"}

        query_info = self.pending_queries[query_id]

        if not approved:
            del self.pending_queries[query_id]
            return {
                "query_id": query_id,
                "status": "rejected",
                "message": "Query execution cancelled by user",
            }

        sql = query_info["sql"]
        is_select_query = sql.strip().upper().startswith("SELECT")

        if (
            is_select_query
            and self.query_cache_service
            and self.query_cache_service.enabled
        ):
            cache_key = self.query_cache_service.get_sql_result_key(sql)
            cached_result = self.query_cache_service.get(cache_key, cache_type="sql_result")

            if cached_result and "results" in cached_result:
                logger.info(f"SQL result cache HIT for query: '{sql[:50]}...'")
                del self.pending_queries[query_id]
                return {
                    "query_id": query_id,
                    "question": query_info["question"],
                    "sql": sql,
                    "results": cached_result["results"],
                    "result_count": cached_result["result_count"],
                    "status": "executed",
                    "cache_hit": True,
                    "cached_at": cached_result.get("executed_at"),
                }

        try:
            results = await self.vanna.execute_sql_async(sql)

            if (
                is_select_query
                and self.query_cache_service
                and self.query_cache_service.enabled
            ):
                cache_key = self.query_cache_service.get_sql_result_key(sql)
                cache_value = {
                    "results": results,
                    "result_count": len(results),
                    "sql": sql,
                    "executed_at": pd.Timestamp.now().isoformat(),
                }
                ttl = settings.cache_ttl_sql_result
                self.query_cache_service.set(
                    cache_key, cache_value, ttl=ttl, cache_type="sql_result"
                )
                logger.info(f"SQL result cache MISS - cached for '{sql[:50]}...' (TTL: {ttl}s)")

            del self.pending_queries[query_id]

            return {
                "query_id": query_id,
                "question": query_info["question"],
                "sql": sql,
                "results": results,
                "result_count": len(results),
                "status": "executed",
                "cache_hit": False,
            }

        except Exception as e:
            return {"query_id": query_id, "error": str(e), "status": "error"}

    def get_pending_queries(self) -> List[Dict[str, Any]]:
        return [{"query_id": qid, **info} for qid, info in self.pending_queries.items()]
