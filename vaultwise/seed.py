"""Demo data seeder for Vaultwise. Runs on first startup when the database is empty."""

import json
import random
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from vaultwise.database import get_connection
from vaultwise.search import compute_embedding

# ---------------------------------------------------------------------------
# Seed documents -- substantive, realistic content
# ---------------------------------------------------------------------------

SEED_DOCUMENTS = [
    {
        "title": "Employee Handbook 2026",
        "source": "upload",
        "doc_type": "markdown",
        "content": (
            "# Employee Handbook 2026\n\n"
            "## Welcome\n\n"
            "Welcome to Acme Corp. This handbook outlines our policies, benefits, and expectations "
            "for all team members. We believe in transparency, collaboration, and continuous learning.\n\n"
            "## Work Schedule\n\n"
            "Standard business hours are 9:00 AM to 5:30 PM, Monday through Friday. We offer flexible "
            "scheduling with manager approval. Core hours (when everyone should be available) are "
            "10:00 AM to 3:00 PM. Remote work is available up to 3 days per week for eligible roles.\n\n"
            "## Paid Time Off\n\n"
            "Full-time employees receive 20 days of PTO per year, accruing at 1.67 days per month. "
            "PTO rolls over up to 5 unused days into the next calendar year. Requests should be "
            "submitted at least 2 weeks in advance for planned time off. Sick leave is separate and "
            "unlimited with manager notification.\n\n"
            "## Benefits\n\n"
            "All full-time employees are eligible for: health insurance (medical, dental, vision), "
            "401(k) with 4% company match, life insurance at 2x annual salary, employee assistance "
            "program, professional development budget of $2,000 per year, and gym membership "
            "reimbursement up to $75 per month.\n\n"
            "## Code of Conduct\n\n"
            "We expect all employees to act with integrity, respect colleagues and customers, "
            "protect confidential information, report concerns through proper channels, and "
            "maintain a professional and inclusive workplace. Harassment, discrimination, and "
            "retaliation of any kind are strictly prohibited and will result in disciplinary "
            "action up to and including termination.\n\n"
            "## Performance Reviews\n\n"
            "Performance reviews are conducted bi-annually in June and December. The review process "
            "includes self-assessment, peer feedback (360-degree), manager evaluation, and goal "
            "setting for the next period. Compensation adjustments are considered during the "
            "December review cycle."
        ),
    },
    {
        "title": "REST API Documentation v3.0",
        "source": "upload",
        "doc_type": "markdown",
        "content": (
            "# REST API Documentation v3.0\n\n"
            "## Authentication\n\n"
            "All API requests require a Bearer token in the Authorization header. Tokens are obtained "
            "via the /auth/token endpoint using OAuth 2.0 client credentials flow. Tokens expire after "
            "1 hour and must be refreshed using the refresh_token grant type.\n\n"
            "## Rate Limiting\n\n"
            "API requests are rate-limited to 1000 requests per minute per API key. When the rate limit "
            "is exceeded, the API returns HTTP 429 with a Retry-After header indicating how many seconds "
            "to wait. Batch endpoints have a separate limit of 100 requests per minute.\n\n"
            "## Core Endpoints\n\n"
            "### GET /api/v3/users\n"
            "Returns a paginated list of users. Query parameters: page (default 1), per_page (default 20, "
            "max 100), sort_by (name, created_at, email), order (asc, desc). Response includes total_count "
            "and pagination metadata.\n\n"
            "### POST /api/v3/users\n"
            "Creates a new user. Required fields: email (unique), name, role (admin, member, viewer). "
            "Optional fields: department, phone, timezone. Returns the created user object with HTTP 201.\n\n"
            "### GET /api/v3/users/{id}\n"
            "Returns a single user by ID. Returns HTTP 404 if not found. Includes related resources "
            "(teams, projects) in the response when ?include=teams,projects is specified.\n\n"
            "## Error Handling\n\n"
            "All errors follow the format: {error: {code: string, message: string, details: object}}. "
            "Common error codes: VALIDATION_ERROR (400), UNAUTHORIZED (401), FORBIDDEN (403), "
            "NOT_FOUND (404), RATE_LIMITED (429), INTERNAL_ERROR (500). Clients should implement "
            "exponential backoff for 429 and 500 errors.\n\n"
            "## Webhooks\n\n"
            "Configure webhooks at /settings/webhooks to receive real-time notifications. Events "
            "include: user.created, user.updated, user.deleted, project.completed, invoice.paid. "
            "Webhook payloads are signed with HMAC-SHA256 using your webhook secret. Delivery is "
            "retried up to 5 times with exponential backoff on failure."
        ),
    },
    {
        "title": "New Employee Onboarding Guide",
        "source": "upload",
        "doc_type": "markdown",
        "content": (
            "# New Employee Onboarding Guide\n\n"
            "## Week 1: Getting Started\n\n"
            "Day 1: Arrive at 9:00 AM. HR orientation covers paperwork, badge access, and building tour. "
            "Meet your assigned buddy who will help you navigate the first month. Set up your workstation: "
            "laptop, monitors, and peripherals will be pre-configured by IT. Activate your accounts: email "
            "(Google Workspace), Slack, GitHub, and JIRA.\n\n"
            "Day 2-3: Complete required compliance training modules in the LMS: Information Security "
            "Awareness (45 min), Anti-Harassment Policy (30 min), Data Privacy and GDPR (60 min), and "
            "Code of Conduct Acknowledgment (15 min). All modules must be completed within the first week.\n\n"
            "Day 4-5: Shadow team members during standups and working sessions. Review the team's "
            "documentation and project boards. Complete your first pull request (even if it's just fixing "
            "a typo in docs -- the goal is to verify your development environment works end-to-end).\n\n"
            "## Week 2-4: Ramping Up\n\n"
            "Meet with your manager to set 30-60-90 day goals. Start attending all team ceremonies "
            "(standups, planning, retros). Take ownership of a small, well-defined task. Explore the "
            "codebase and ask questions in #engineering-help on Slack.\n\n"
            "## Key Contacts\n\n"
            "IT Support: it-help@acmecorp.com or Slack #it-support (response time: 2 hours). "
            "HR Questions: hr@acmecorp.com or Slack #hr-questions. Facilities: facilities@acmecorp.com. "
            "Security Incidents: security@acmecorp.com (immediate response for urgent issues).\n\n"
            "## Tools and Access\n\n"
            "Development: GitHub Enterprise, JIRA, Confluence, VS Code (standard), Docker Desktop. "
            "Communication: Google Workspace (email, calendar, meet), Slack, Zoom (for external calls). "
            "Design: Figma (view access by default, edit access upon request to your manager). "
            "Infrastructure: AWS Console (read-only initially, elevated access via PIM request)."
        ),
    },
    {
        "title": "Information Security Policy",
        "source": "upload",
        "doc_type": "markdown",
        "content": (
            "# Information Security Policy\n\n"
            "## Purpose and Scope\n\n"
            "This policy establishes the security requirements for all employees, contractors, and third "
            "parties who access Acme Corp systems and data. It applies to all information assets including "
            "hardware, software, data, and network resources.\n\n"
            "## Data Classification\n\n"
            "All data must be classified into one of four levels: Public (freely shareable, e.g., marketing "
            "materials), Internal (not publicly shared but low risk if exposed, e.g., internal memos), "
            "Confidential (business-sensitive, e.g., financial reports, strategic plans), and Restricted "
            "(highest sensitivity, e.g., PII, credentials, medical records). Data handling requirements "
            "increase with each classification level.\n\n"
            "## Access Control\n\n"
            "Access follows the principle of least privilege. All accounts require multi-factor authentication "
            "(MFA). Privileged access requires approval from the security team and is reviewed quarterly. "
            "Service accounts must use API keys or certificates, never passwords. Access is automatically "
            "revoked upon termination and reviewed during role changes.\n\n"
            "## Password Policy\n\n"
            "Minimum 14 characters for user accounts, 20 characters for service accounts. Must include "
            "uppercase, lowercase, numbers, and special characters. Passwords must not be reused across "
            "the last 12 passwords. Password managers are mandatory -- LastPass Enterprise is provided to "
            "all employees.\n\n"
            "## Incident Response\n\n"
            "Security incidents must be reported immediately to security@acmecorp.com and the #security-incidents "
            "Slack channel. The incident response team will triage, classify severity (P1-P4), and coordinate "
            "response. Post-incident reviews are conducted within 5 business days. All incidents are tracked "
            "in the security incident management system.\n\n"
            "## Device Security\n\n"
            "All company devices must have: full-disk encryption enabled, endpoint protection (CrowdStrike) "
            "installed and active, automatic OS updates enabled, screen lock after 5 minutes of inactivity. "
            "Personal devices accessing company data must be enrolled in the MDM solution and meet minimum "
            "security requirements."
        ),
    },
    {
        "title": "System Architecture Overview",
        "source": "upload",
        "doc_type": "markdown",
        "content": (
            "# System Architecture Overview\n\n"
            "## High-Level Design\n\n"
            "The Acme platform follows a microservices architecture deployed on AWS. The system is organized "
            "into four layers: API Gateway (Kong), Service Mesh (AWS App Mesh), Core Services (12 microservices), "
            "and Data Layer (PostgreSQL, Redis, S3). All inter-service communication uses gRPC with Protocol "
            "Buffers for internal calls and REST/JSON for external-facing APIs.\n\n"
            "## Core Services\n\n"
            "User Service: Authentication, authorization, and user profile management. Uses PostgreSQL for "
            "storage and Redis for session caching. Handles OAuth 2.0, SAML SSO, and API key management.\n\n"
            "Order Service: Order lifecycle management from creation to fulfillment. Implements the saga "
            "pattern for distributed transactions across inventory, payment, and shipping services.\n\n"
            "Notification Service: Multi-channel notifications (email via SES, SMS via Twilio, push via "
            "Firebase, in-app via WebSocket). Uses a priority queue (SQS) to handle burst traffic.\n\n"
            "Analytics Service: Real-time event processing using Kinesis Data Streams. Events are stored "
            "in S3 (Parquet format) and queried via Athena. Dashboard powered by a custom React application.\n\n"
            "## Infrastructure\n\n"
            "Production runs on AWS in us-east-1 (primary) and us-west-2 (DR). Services are deployed as "
            "Docker containers on ECS Fargate. Infrastructure is managed with Terraform (v1.6+). CI/CD "
            "pipeline: GitHub Actions builds, pushes to ECR, deploys via CodeDeploy with blue-green strategy.\n\n"
            "## Monitoring and Observability\n\n"
            "Metrics: Prometheus + Grafana for service metrics, CloudWatch for infrastructure metrics. "
            "Logging: Structured JSON logs shipped to Elasticsearch via Fluentd. Tracing: OpenTelemetry "
            "with Jaeger for distributed tracing across all services. Alerting: PagerDuty integration with "
            "tiered escalation policies. SLOs: 99.9% availability for customer-facing APIs, p99 latency "
            "under 500ms for read operations."
        ),
    },
]

# ---------------------------------------------------------------------------
# Seed questions
# ---------------------------------------------------------------------------

SEED_QUESTIONS = [
    {"query": "How many PTO days do employees get per year?", "answer": "Full-time employees receive 20 days of PTO per year, accruing at 1.67 days per month. Up to 5 unused days can roll over into the next calendar year.", "confidence": 0.92},
    {"query": "What is the API rate limit?", "answer": "API requests are rate-limited to 1000 requests per minute per API key. Batch endpoints have a separate limit of 100 requests per minute. When exceeded, the API returns HTTP 429 with a Retry-After header.", "confidence": 0.88},
    {"query": "What training modules must new employees complete?", "answer": "New employees must complete four modules in their first week: Information Security Awareness (45 min), Anti-Harassment Policy (30 min), Data Privacy and GDPR (60 min), and Code of Conduct Acknowledgment (15 min).", "confidence": 0.95},
    {"query": "What is the minimum password length?", "answer": "Minimum 14 characters for user accounts and 20 characters for service accounts. Passwords must include uppercase, lowercase, numbers, and special characters.", "confidence": 0.91},
    {"query": "How does the notification service work?", "answer": "The Notification Service handles multi-channel notifications: email via SES, SMS via Twilio, push via Firebase, and in-app via WebSocket. It uses a priority queue (SQS) to handle burst traffic.", "confidence": 0.85},
    {"query": "What benefits does the company offer?", "answer": "Full-time employees are eligible for: health insurance (medical, dental, vision), 401(k) with 4% company match, life insurance at 2x annual salary, employee assistance program, $2,000/year professional development budget, and gym membership reimbursement up to $75/month.", "confidence": 0.93},
    {"query": "How do I report a security incident?", "answer": "Security incidents must be reported immediately to security@acmecorp.com and the #security-incidents Slack channel. The incident response team will triage and classify severity (P1-P4).", "confidence": 0.90},
    {"query": "What CI/CD pipeline does the company use?", "answer": "The CI/CD pipeline uses GitHub Actions for builds, pushes container images to ECR, and deploys via CodeDeploy with a blue-green deployment strategy. Services run as Docker containers on ECS Fargate.", "confidence": 0.82},
    {"query": "When are performance reviews conducted?", "answer": "Performance reviews are conducted bi-annually in June and December. The process includes self-assessment, 360-degree peer feedback, manager evaluation, and goal setting. Compensation adjustments are considered during the December review cycle.", "confidence": 0.89},
    {"query": "What data classification levels exist?", "answer": "Four levels: Public (freely shareable), Internal (not publicly shared, low risk), Confidential (business-sensitive like financial reports), and Restricted (highest sensitivity like PII, credentials, medical records).", "confidence": 0.94},
]

# ---------------------------------------------------------------------------
# Seed articles
# ---------------------------------------------------------------------------

SEED_ARTICLES = [
    {
        "title": "Getting Started at Acme Corp: A Complete Guide",
        "status": "published",
        "content": (
            "# Getting Started at Acme Corp\n\n"
            "Welcome to the team! This guide combines the most important information from our "
            "employee handbook and onboarding materials.\n\n"
            "## Your First Week\n\n"
            "You'll start with HR orientation on Day 1, covering paperwork, badge access, and a building "
            "tour. Your assigned buddy will help you navigate. IT will have your workstation pre-configured "
            "with all necessary tools.\n\n"
            "## Key Policies\n\n"
            "- **Work Schedule:** Core hours are 10 AM - 3 PM. Flexible scheduling available with approval.\n"
            "- **PTO:** 20 days per year, accruing monthly. Up to 5 days roll over.\n"
            "- **Remote Work:** Up to 3 days per week for eligible roles.\n\n"
            "## Required Training\n\n"
            "Complete these modules within your first week:\n"
            "1. Information Security Awareness (45 min)\n"
            "2. Anti-Harassment Policy (30 min)\n"
            "3. Data Privacy and GDPR (60 min)\n"
            "4. Code of Conduct (15 min)\n\n"
            "## Benefits Highlights\n\n"
            "Health insurance, 401(k) with 4% match, $2,000 annual learning budget, and gym reimbursement. "
            "See the full Employee Handbook for details."
        ),
    },
    {
        "title": "API Integration Best Practices",
        "status": "published",
        "content": (
            "# API Integration Best Practices\n\n"
            "This article consolidates key information about working with the Acme API.\n\n"
            "## Authentication\n\n"
            "Use OAuth 2.0 client credentials flow. Tokens expire after 1 hour -- implement automatic "
            "refresh logic in your client. Never hardcode tokens in source code.\n\n"
            "## Rate Limiting\n\n"
            "Standard limit: 1000 requests/minute. Batch endpoints: 100 requests/minute. Handle HTTP 429 "
            "responses with exponential backoff using the Retry-After header.\n\n"
            "## Error Handling\n\n"
            "All errors return a consistent format with error code, message, and details. Implement "
            "retry logic for 429 (rate limited) and 500 (server error) responses.\n\n"
            "## Webhook Integration\n\n"
            "Set up webhooks for real-time event notifications. Always verify webhook signatures using "
            "HMAC-SHA256 before processing payloads. Implement idempotent handlers since events may "
            "be delivered more than once."
        ),
    },
    {
        "title": "Security Essentials for Every Employee",
        "status": "draft",
        "content": (
            "# Security Essentials for Every Employee\n\n"
            "Understanding and following our security practices is everyone's responsibility.\n\n"
            "## Password Management\n\n"
            "Use LastPass Enterprise (provided by the company) for all password management. Minimum "
            "14 characters with mixed character types. Never reuse passwords across services.\n\n"
            "## Data Handling\n\n"
            "Always classify data before sharing. Restricted data (PII, credentials) requires "
            "encryption at rest and in transit. Never share confidential data via unencrypted email.\n\n"
            "## Device Security\n\n"
            "Keep full-disk encryption enabled. Install and maintain CrowdStrike endpoint protection. "
            "Enable automatic OS updates. Lock your screen after 5 minutes of inactivity.\n\n"
            "## Reporting Incidents\n\n"
            "If you suspect a security issue, report it immediately to security@acmecorp.com and "
            "the #security-incidents Slack channel. Quick reporting can prevent minor issues from "
            "becoming major breaches."
        ),
    },
]

# ---------------------------------------------------------------------------
# Seed quizzes
# ---------------------------------------------------------------------------

SEED_QUIZZES = [
    {
        "title": "Quiz: Onboarding Essentials",
        "questions": [
            {
                "question": "How many days of PTO do full-time employees receive per year?",
                "options": ["10 days", "15 days", "20 days", "25 days"],
                "correct_index": 2,
                "explanation": "Full-time employees receive 20 days of PTO per year, accruing at 1.67 days per month."
            },
            {
                "question": "What are the core hours when all employees should be available?",
                "options": ["8 AM - 4 PM", "9 AM - 5 PM", "10 AM - 3 PM", "11 AM - 4 PM"],
                "correct_index": 2,
                "explanation": "Core hours are 10:00 AM to 3:00 PM, during which everyone should be available."
            },
            {
                "question": "How many days per week can eligible employees work remotely?",
                "options": ["1 day", "2 days", "3 days", "5 days"],
                "correct_index": 2,
                "explanation": "Remote work is available up to 3 days per week for eligible roles."
            },
            {
                "question": "What is the annual professional development budget per employee?",
                "options": ["$500", "$1,000", "$1,500", "$2,000"],
                "correct_index": 3,
                "explanation": "Each employee receives a professional development budget of $2,000 per year."
            },
        ],
    },
    {
        "title": "Quiz: Security Awareness",
        "questions": [
            {
                "question": "What is the minimum password length for user accounts?",
                "options": ["8 characters", "10 characters", "12 characters", "14 characters"],
                "correct_index": 3,
                "explanation": "Minimum 14 characters for user accounts, 20 characters for service accounts."
            },
            {
                "question": "Which tool is provided for password management?",
                "options": ["1Password", "Bitwarden", "LastPass Enterprise", "KeePass"],
                "correct_index": 2,
                "explanation": "LastPass Enterprise is provided to all employees for password management."
            },
            {
                "question": "How quickly must security incidents be reported?",
                "options": ["Within 24 hours", "Within 1 hour", "Immediately", "By end of business day"],
                "correct_index": 2,
                "explanation": "Security incidents must be reported immediately to security@acmecorp.com and Slack."
            },
            {
                "question": "What classification level applies to PII and credentials?",
                "options": ["Public", "Internal", "Confidential", "Restricted"],
                "correct_index": 3,
                "explanation": "Restricted is the highest sensitivity level, covering PII, credentials, and medical records."
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Seed knowledge gaps
# ---------------------------------------------------------------------------

SEED_GAPS = [
    {"topic": "vpn setup instructions", "frequency": 8, "queries": ["How do I set up VPN?", "VPN configuration guide", "Connect to office network remotely", "VPN client download"]},
    {"topic": "expense reimbursement process", "frequency": 6, "queries": ["How do I submit expenses?", "Expense report deadline", "What expenses are reimbursable?"]},
    {"topic": "promotion criteria and timeline", "frequency": 5, "queries": ["How do promotions work?", "What are the promotion criteria?", "When are promotions decided?"]},
    {"topic": "disaster recovery procedures", "frequency": 4, "queries": ["What is our DR plan?", "Disaster recovery runbook", "Failover procedures"]},
    {"topic": "third-party integration guidelines", "frequency": 3, "queries": ["How to integrate third-party APIs?", "Vendor integration approval process"]},
]


def run_seed() -> None:
    """Populate the database with realistic demo data.

    This function is idempotent -- it only runs when the documents table is empty.
    """
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc)
        doc_ids: list[str] = []

        # --- Seed documents and chunks ---
        for i, doc_data in enumerate(SEED_DOCUMENTS):
            doc_id = uuid4().hex
            doc_ids.append(doc_id)
            content = doc_data["content"]
            word_count = len(content.split())
            created = (now - timedelta(days=30 - i * 5)).isoformat()

            conn.execute(
                "INSERT INTO documents (id, title, source, content, doc_type, word_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, doc_data["title"], doc_data["source"], content,
                 doc_data["doc_type"], word_count, created, created),
            )

            # Chunk each document
            words = content.split()
            chunk_size = max(len(words) // 4, 50)
            chunk_index = 0
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_content = " ".join(words[start:end])
                chunk_id = uuid4().hex
                embedding = compute_embedding(chunk_content)
                embedding_json = json.dumps(embedding) if embedding else None
                conn.execute(
                    "INSERT INTO chunks (id, doc_id, content, chunk_index, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, chunk_content, chunk_index, embedding_json, created),
                )
                chunk_index += 1
                start = end

        # --- Seed questions ---
        for i, q_data in enumerate(SEED_QUESTIONS):
            q_id = uuid4().hex
            created = (now - timedelta(days=6 - i % 7, hours=random.randint(1, 12))).isoformat()
            source_ids = [doc_ids[i % len(doc_ids)]]
            conn.execute(
                "INSERT INTO questions (id, query, answer, sources, confidence, helpful, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (q_id, q_data["query"], q_data["answer"], json.dumps(source_ids),
                 q_data["confidence"], random.choice([0, 1, None]), created),
            )

        # --- Seed articles ---
        article_ids: list[str] = []
        for i, art_data in enumerate(SEED_ARTICLES):
            art_id = uuid4().hex
            article_ids.append(art_id)
            created = (now - timedelta(days=20 - i * 7)).isoformat()
            source_doc_ids = json.dumps(doc_ids[:2] if i == 0 else [doc_ids[i % len(doc_ids)]])
            conn.execute(
                "INSERT INTO articles (id, title, content, source_doc_ids, status, auto_generated, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (art_id, art_data["title"], art_data["content"], source_doc_ids,
                 art_data["status"], 1, created),
            )

        # --- Seed quizzes ---
        for i, quiz_data in enumerate(SEED_QUIZZES):
            quiz_id = uuid4().hex
            art_id = article_ids[i] if i < len(article_ids) else article_ids[0]
            created = (now - timedelta(days=15 - i * 5)).isoformat()
            conn.execute(
                "INSERT INTO quizzes (id, article_id, title, questions_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (quiz_id, art_id, quiz_data["title"], json.dumps(quiz_data["questions"]), created),
            )

        # --- Seed knowledge gaps ---
        for gap_data in SEED_GAPS:
            gap_id = uuid4().hex
            created = (now - timedelta(days=random.randint(3, 14))).isoformat()
            last_asked = (now - timedelta(days=random.randint(0, 2))).isoformat()
            conn.execute(
                "INSERT INTO knowledge_gaps (id, topic, frequency, sample_queries, status, last_asked, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (gap_id, gap_data["topic"], gap_data["frequency"],
                 json.dumps(gap_data["queries"]), "open", last_asked, created),
            )

        # --- Seed usage log ---
        actions = ["search", "ask", "view_article", "take_quiz"]
        sample_queries = [
            "PTO policy", "API authentication", "onboarding steps",
            "password requirements", "security incident reporting",
            "remote work policy", "benefits overview", "deployment process",
            None, None,  # Some actions don't have queries
        ]
        for _ in range(20):
            log_id = uuid4().hex
            action = random.choice(actions)
            query = random.choice(sample_queries) if action in ("search", "ask") else None
            response_time = random.randint(50, 800)
            created = (now - timedelta(
                days=random.randint(0, 6),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )).isoformat()
            conn.execute(
                "INSERT INTO usage_log (id, action, query, response_time_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (log_id, action, query, response_time, created),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
