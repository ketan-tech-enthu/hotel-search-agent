# Conversational Hotel Search Agent

An agentic AI case study: a **cloud-agnostic, conversational hotel search agent** built with
**LangGraph**. Ask for hotels in plain language and keep refining. There is **no fixed list of
filters**: the agent discovers what fields the live data contains and filters on any of them.

```
You: Hotels in Goa from 2026-12-20 to 2026-12-24
You: only 4 and 5 star with parking
You: which ones can I walk to a beach from? cheapest first
You: at least 80% positive reviews
You: what's the check-in time at the Taj?
You: which of these do guests say is quiet?
[approval] Read reviews for Taj Fort Aguada, Novotel Candolim? This uses 2 SerpApi credit(s). [y/N] y
You: same filters for Lonavala
```

---

## Highlights

- **Conversational and open-ended**: star class, amenities, nearby places, review sentiment, price,
  location score… any field in the data, plus follow-ups that build on earlier answers.
- **Field catalog instead of hard-coded filters**: new data fields become filterable with no code change.
- **Grounded output**: Python filters, sorts and prints the table; the LLM never re-types prices.
- **Cloud and model agnostic**: switch OpenAI / Azure OpenAI / Anthropic / Gemini / Bedrock / Ollama
  and memory / SQLite / Postgres through configuration only.
- **Durable conversations**: state is checkpointed; restart and resume the same thread.
- **Human-in-the-loop**: asks before spending API credits on reviews.
- **Framework-free core**: all business logic lives in `hotel_core.py`, reusable behind a REST API
  or another agent framework.

---

## Architecture

```
        START ─► agent ──(tool calls?)──► tools ─┐
                   ▲          │ no               │
                   │          ▼                  │
                   │         END                 │
                   └─────────────────────────────┘

  agent  = any chat model + bound tools + system prompt
  tools  = search_hotels · query_hotels · get_hotel_details · read_reviews
  state  = messages + search + hotels + current result + review cache  (checkpointed)
```

```
┌──────────────── hotel_graph.py (LangGraph layer) ────────────────┐
│ CLI ─► graph ─► agent (LLM)  ⇄  tools  ─► checkpointer            │
└──────────────────────────────────┬────────────────────────────────┘
                                   ▼
┌──────────────── hotel_core.py (framework-free) ──────────────────┐
│ records · % positive reviews · field catalog · query engine       │
│ SerpApi adapters · table / CSV rendering                          │
└──────────────────────────────────┬────────────────────────────────┘
                                   ▼
                     SerpApi (Google Hotels + Reviews)
```

| Responsibility | Done by |
|---|---|
| Decide which tool to call, map words to fields and values, keep context across turns | LLM |
| Fetch, page, de-duplicate, filter, sort, compute derived fields | Python (`hotel_core`) |
| Run the agent ⇄ tools loop, persist state, pause for approval | LangGraph |
| Print the hotel table and CSV | Python (never the LLM) |

### Tools

| Tool | Purpose | API cost |
|---|---|---|
| `search_hotels` | Fetch live hotels for a location and dates; cache them; return a **field catalog** | 1 credit per page (~20 hotels) |
| `query_hotels` | Filter and sort cached hotels on **any** catalog field | free |
| `get_hotel_details` | Full record of named hotels (check-in time, nearby places, …) | free |
| `read_reviews` | Recent guest reviews for qualities no field captures (quiet, clean, family-friendly) | 1 credit per hotel, cached, asks for approval |

### Generic query language

`query_hotels` takes conditions `{field, op, value}`:

- **Operators:** `eq ne gt gte lt lte in contains not_contains exists not_exists`
- **Groups:** `all_of` (AND) and `any_of` (OR)
- **Sorting:** multiple keys, ascending or descending
- **Scope:** conditions that must hold on the *same* list item, e.g. a nearby place that is a
  beach **and** reachable on foot

| User says | Condition sent by the LLM |
|---|---|
| "4 and 5 star" | `extracted_hotel_class in ['4','5']` |
| "with parking" | `amenities contains 'parking'` |
| "80% positive reviews" | `positive_review_pct gte '80'` |
| "under 5k a night" | `price_per_night lte '5000'` |
| "walk to the beach" | scope `nearby_places`: `name contains 'beach'` AND `transportations.type contains 'walk'` |
| "cheapest first" | sort `price_per_night` ascending |

---

## Project structure

```
.
├── hotel_core.py        # framework-free domain logic (no LLM / cloud / framework imports)
├── hotel_graph.py       # LangGraph state, tools, graph, model + checkpointer factories, CLI
├── requirements.txt
├── .env.example         # configuration template (copy to .env)
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Getting started

### Prerequisites

- Python **3.10+**
- An API key for one LLM provider (OpenAI, Azure OpenAI, Anthropic, Google, AWS Bedrock), or a
  local [Ollama](https://ollama.com) install
- A **SerpApi** key (see below)

### 1. Get a SerpApi key

1. Register at [serpapi.com](https://serpapi.com) and verify your account.
2. Copy **Your Private API Key** from the dashboard.
3. The free plan includes 250 searches per month (50 per hour). See SerpApi's pricing page for current limits.

### 2. Install

```bash
git clone <your-repo-url> hotel-agent && cd hotel-agent
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # uncomment your LLM provider's package first if not OpenAI
cp .env.example .env                 # then fill in your keys
```

### 3. Configure `.env`

| Variable | Required | Example | Purpose |
|---|---|---|---|
| `SERPAPI_API_KEY` | yes | `abc123…` | SerpApi key |
| `SERPAPI_GL` | no | `in` | Country for Google results (default `in`) |
| `LLM_MODEL` | no | `openai:gpt-5-mini` | `<provider>:<model>` (default `openai:gpt-5-mini`) |
| `OPENAI_API_KEY` / `AZURE_OPENAI_*` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | per provider | | Credentials for the chosen provider |
| `CHECKPOINTER` | no | `sqlite:hotel_agent.db` | `memory`, `sqlite:<file>` or `postgresql://…` |
| `REQUIRE_APPROVAL` | no | `1` | Ask before spending credits on reviews (`0` to disable) |

**Switching model provider** needs only configuration:

```
LLM_MODEL=openai:gpt-5-mini
LLM_MODEL=azure_openai:<your-deployment-name>
LLM_MODEL=anthropic:<model-id>
LLM_MODEL=google_genai:<model-id>
LLM_MODEL=bedrock_converse:<model-id>
LLM_MODEL=ollama:<model-name>          # fully local
```

Install the matching package (`langchain-anthropic`, `langchain-google-genai`, `langchain-aws`,
`langchain-ollama`), as listed in `requirements.txt`.

### 4. Run

```bash
python hotel_graph.py --thread goa-trip
```

- Type your requests; an empty line or `quit` exits.
- Run the same command again to **resume** the conversation, including its cached hotels.
- Use a different `--thread` for a separate conversation.
- Each result is printed as a table and saved to `hotels_<location>_<checkin>_<checkout>.csv`.

### Docker

```bash
docker build -t hotel-agent .
docker run -it --env-file .env -v hotel-data:/data hotel-agent --thread goa-trip
```

Checkpoints are stored in the `/data` volume, so conversations survive container restarts.

---

## Costs and quotas

| Action | SerpApi credits |
|---|---|
| Hotel search (default 5 pages) | 5 |
| "Search more pages" | up to 25 |
| Reading reviews | 1 per hotel (max 5 per request, cached) |
| Follow-up filters, sorting, hotel details | 0 |

On the free plan, 250 credits is roughly 50 searches a month. While developing, ask for fewer pages
("search 1 page") or lower `DEFAULT_MAX_PAGES` in `hotel_core.py`.

LLM cost depends on your provider. Each turn makes 2–4 model calls; the field catalog keeps tool
results compact, and the full hotel list is never sent to the model.

---

## Design decisions

- **Field catalog over fixed fields.** `field_catalog()` walks the live data and lists every field
  path with its type and example values, so the LLM maps user language onto real fields.
- **Thin LLM, deterministic core.** The LLM plans; Python executes. Results and prices come from
  code, which removes a whole class of hallucination.
- **State in the graph.** Hotels, the current result (stored as indices to keep checkpoints small)
  and reviews live in LangGraph state and are checkpointed with the conversation.
- **Reducers.** `keep_last` and `merge_dicts` make parallel tool updates safe.
- **Human-in-the-loop.** `read_reviews` calls `interrupt()`; the CLI resumes with `Command(resume=…)`.
- **Safety limits.** `recursion_limit: 30` caps steps per turn, so a looping agent stops quickly
  instead of spending tokens and credits.

---

## Limitations and production considerations

- **Data source.** SerpApi is an independent third party that retrieves Google Hotels results by
  scraping; it is not affiliated with Google. Google filed a lawsuit against SerpApi in December
  2025. This is fine for learning and demos; for production, use a licensed hotel API
  (e.g. LiteAPI, Expedia Rapid, Booking.com Demand API) and review its terms.
- **Prices** are what Google Hotels shows at search time; always confirm on the booking site.
- **Single-user CLI.** `thread_id` is a conversation name, not a user identity. For a multi-user
  deployment, scope threads per authenticated user (e.g. `"{user_id}:{conversation}"`) and check
  ownership before loading a thread.
- **Error handling.** `GraphRecursionError` is not yet caught in the CLI.
- **Tests and evaluations** are not included in this repository yet.

---

## Roadmap

- [ ] `HotelProvider` interface with SerpApi and LiteAPI adapters (provider-agnostic data)
- [ ] REST API (FastAPI) with authentication and user-scoped threads
- [ ] Long-term user preferences with the LangGraph Store
- [ ] Unit tests for `hotel_core` and scripted-model tests for the graph
- [ ] Evaluation set for prompt → filter mapping
- [ ] Streaming responses

---

## Tech stack

Python · LangGraph · LangChain (`init_chat_model`) · Pydantic · httpx · SQLite / PostgreSQL
checkpointers · SerpApi · Docker

---

## Disclaimer

This is an educational case study, not a booking service. Hotel data is provided by third-party
services under their own terms. Verify prices and availability with the hotel or booking site
before making any booking.

## License

Add a license of your choice (e.g. MIT) before publishing.
