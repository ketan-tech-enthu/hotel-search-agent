"""
Cloud-agnostic conversational hotel agent on LangGraph.

  Model      : any provider via LLM_MODEL  (openai:..., azure_openai:..., anthropic:...,
               google_genai:..., bedrock_converse:..., ollama:...)
  Memory     : any checkpointer via CHECKPOINTER  (memory | sqlite:<file> | postgresql://...)
  Data       : SerpApi (plain HTTPS, runs anywhere)
  Logic      : hotel_core (framework-free)

Graph
        START ─► agent ──(tool calls?)──► tools ─┐
                   ▲          │ no               │
                   │          ▼                  │
                   │         END                 │
                   └─────────────────────────────┘
  - Hotel data, the current result and the review cache live in graph STATE, so they are
    checkpointed with the conversation: restart the app with the same --thread and carry on.
  - read_reviews pauses the graph with interrupt() to ask before spending SerpApi credits.

Run:
    python hotel_graph.py                      # new or default conversation
    python hotel_graph.py --thread goa-trip    # resume a named conversation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from datetime import date
from typing import Annotated, Any, AsyncIterator, Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command, interrupt

import hotel_core as core
from hotel_core import Condition, SortKey

load_dotenv()


# ---------------------------------------------------------------------------
# Graph state (checkpointed). Reducers make parallel tool updates safe.
# ---------------------------------------------------------------------------
def keep_last(_old: Any, new: Any) -> Any:
    return new


def merge_dicts(old: Optional[dict], new: Optional[dict]) -> dict:
    return {**(old or {}), **(new or {})}


class HotelState(MessagesState):
    search: Annotated[Optional[dict], keep_last]
    hotels: Annotated[list[dict], keep_last]
    result_idx: Annotated[list[int], keep_last]        # indices into hotels: keeps checkpoints small
    last_query: Annotated[dict, keep_last]
    result_version: Annotated[int, keep_last]
    reviews_cache: Annotated[dict, merge_dicts]


def _reply(tool_call_id: str, payload: dict, **updates: Any) -> Command:
    """Tool result for the LLM plus state updates, in one Command."""
    msg = ToolMessage(json.dumps(payload, ensure_ascii=False, default=str), tool_call_id=tool_call_id)
    return Command(update={"messages": [msg], **updates})


# ---------------------------------------------------------------------------
# Tools (thin wrappers over hotel_core)
# ---------------------------------------------------------------------------
@tool
async def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    adults: Optional[int] = None,
    currency: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> Command:
    """Fetch live hotels for a location and stay and cache them. Call only when location,
    dates, guests or currency change, or the user wants a wider search. Returns a FIELD
    CATALOG: every field path present in the data, with its type and example values.

    Args:
        location: City, area or landmark, e.g. "Goa" or "hotels near Bandra, Mumbai".
        check_in_date: Check-in date, YYYY-MM-DD.
        check_out_date: Check-out date, YYYY-MM-DD.
        adults: Number of adult guests (default 2).
        currency: ISO currency code (default INR).
        max_pages: Result pages to scan, about 20 hotels each (default 5, max 25).
    """
    adults, currency = adults or 2, (currency or "INR").upper()
    pages_cap = max(1, min(max_pages or core.DEFAULT_MAX_PAGES, core.HARD_MAX_PAGES))
    try:
        ci, co = core.validate_dates(check_in_date, check_out_date)
        raw, pages, more = await core.fetch_all_hotels(location, check_in_date, check_out_date,
                                                       adults, currency, pages_cap)
    except Exception as exc:
        return _reply(tool_call_id, {"error": str(exc)})
    nights = (co - ci).days
    hotels = [r for p in raw if (r := core.build_record(p, nights, currency))]
    search = {"location": location, "check_in": check_in_date, "check_out": check_out_date,
              "nights": nights, "adults": adults, "currency": currency, "pages": pages,
              "hotels_found": len(raw), "with_price": len(hotels), "more_available": more}
    return _reply(tool_call_id, {**search, "field_catalog": core.field_catalog(hotels)},
                  search=search, hotels=hotels, result_idx=list(range(len(hotels))), last_query={},
                  result_version=(state.get("result_version") or 0) + 1)


@tool
def query_hotels(
    all_of: list[Condition],
    any_of: list[Condition],
    sort: list[SortKey],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Filter and sort the cached hotels on ANY catalog field. Stateless: send the complete
    set of conditions each time, carrying over earlier ones the user has not dropped.
    The program prints the resulting table for the user.

    Args:
        all_of: Conditions that must ALL hold (AND). Empty list for none.
        any_of: Conditions of which at least ONE must hold (OR). Empty list for none.
        sort: Sort keys in priority order. Empty list = best rating first.
    """
    hotels = state.get("hotels") or []
    if not state.get("search"):
        return _reply(tool_call_id, {"error": "No hotels cached yet. Call search_hotels first."})
    used = [core.full_path(c) for c in all_of + any_of] + [s.field for s in sort]
    unknown = sorted(set(used) - core.known_fields(hotels))
    if unknown:
        return _reply(tool_call_id, {"error": f"Unknown field(s) {unknown}. Use paths from the field catalog."})

    position = {id(h): i for i, h in enumerate(hotels)}
    rows = core.run_query(hotels, all_of, any_of, sort)
    query = {"all_of": [c.model_dump(exclude_none=True) for c in all_of],
             "any_of": [c.model_dump(exclude_none=True) for c in any_of],
             "sort": [s.model_dump() for s in sort]}
    payload = {"matched": len(rows), "out_of": len(hotels), "currency": state["search"]["currency"],
               "top_5": core.preview(rows, used)}
    return _reply(tool_call_id, payload, result_idx=[position[id(h)] for h in rows], last_query=query,
                  result_version=(state.get("result_version") or 0) + 1)


@tool
def get_hotel_details(
    hotel_names: list[str],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Full cached record (all fields) for up to 5 named hotels, for questions about specific
    hotels or comparisons. Names may be partial, e.g. 'Taj' or 'the Novotel'.

    Args:
        hotel_names: Hotel names, exact or partial.
    """
    found = core.find_hotels(state.get("hotels") or [], hotel_names[:5])
    if not found:
        return _reply(tool_call_id, {"error": "No cached hotel matches those names."})
    return _reply(tool_call_id, {"hotels": [{k: v for k, v in h.items() if not k.startswith("_")} for h in found]})


@tool
async def read_reviews(
    hotel_names: list[str],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Fetch recent guest reviews (text, rating, highlight tags) for up to 5 named hotels.
    Use for qualities no catalog field captures: quiet, clean, family friendly, good food,
    friendly staff, value for money. The user is asked to approve the SerpApi credits.

    Args:
        hotel_names: Hotel names, exact or partial, usually from the current results.
    """
    found = core.find_hotels(state.get("hotels") or [], hotel_names[:core.MAX_REVIEW_HOTELS])
    if not found:
        return _reply(tool_call_id, {"error": "No cached hotel matches those names."})
    cache = state.get("reviews_cache") or {}
    to_fetch = [h for h in found if h.get("_token") and h["_token"] not in cache]

    # Human-in-the-loop: pause the graph and ask before spending credits.
    # (On resume LangGraph re-runs this tool from the top, so nothing above has side effects.)
    if to_fetch and os.environ.get("REQUIRE_APPROVAL", "1") == "1":
        approved = interrupt({"question": f"Read reviews for {', '.join(h['name'] for h in to_fetch)}? "
                                          f"This uses {len(to_fetch)} SerpApi credit(s)."})
        if not approved:
            return _reply(tool_call_id, {"declined": True,
                                         "note": "User declined. Offer alternatives that use cached data."})

    new_cache, out = {}, {}
    for h in to_fetch:
        try:
            new_cache[h["_token"]] = await core.fetch_reviews(h["_token"])
        except Exception as exc:
            out[h["name"]] = {"error": str(exc)}
    merged = {**cache, **new_cache}
    for h in found:
        if h["name"] not in out:
            out[h["name"]] = merged.get(h.get("_token"), {"error": "No reviews available."})
    return _reply(tool_call_id, {"reviews": out}, reviews_cache=new_cache)


TOOLS = [search_hotels, query_hotels, get_hotel_details, read_reviews]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
INSTRUCTIONS = """You are a friendly hotel assistant working on LIVE data. Today is {today}.

How to work
- search_hotels: when the user names a location and dates (resolve "next Friday",
  "3 nights from 20 Dec" to YYYY-MM-DD; "from" = check-in, "to" = check-out), or changes
  location / dates / guests, or asks to search wider. Read the field_catalog it returns:
  it lists every field you can filter on, with types and real example values.
- query_hotels: for any filtering or ordering request. Map the user's words onto catalog
  fields and example values. Typical mappings (check the catalog, don't assume):
    "4 and 5 star"          -> extracted_hotel_class in ['4','5']   (official class)
    "rated 4+"              -> overall_rating gte '4'
    "parking", "pool"       -> amenities contains 'parking' / 'pool'
    "no pool"               -> amenities not_contains 'pool'
    "80% positive reviews"  -> positive_review_pct gte '80'
    "under 5k a night"      -> price_per_night lte '5000'
    "walk to the beach"     -> scope 'nearby_places': name contains 'beach' AND
                               transportations.type contains 'walk'  (same place, so use scope)
    "great location"        -> location_rating gte '4.5'
    "cheapest"              -> sort price_per_night ascending
  It is stateless: resend ALL conditions still in force from the conversation, plus the new one.
  If it returns an unknown-field error, pick the right path from the catalog and retry.
- get_hotel_details: questions about specific hotels ("tell me about the Taj", "compare the
  top two", "check-in time?", "how far to the airport?").
- read_reviews: qualities no field captures ("quiet", "clean", "good for kids", "good
  breakfast"). Use it on the top few hotels of the current results (max 5). The user is
  asked to approve the cost; if they decline, offer alternatives from cached data.

How to reply
- Be conversational and brief (2-5 sentences). The program prints the hotel table after
  any search or query, so don't repeat the whole list; mention at most the top 2-3.
- Only state facts and prices that appear in tool results. If something isn't in the
  data, say so and offer the closest alternative (another field, or reading reviews).
- Say how you interpreted anything ambiguous (e.g. "4 star" as official class).
- Ask one short question if the location or dates are missing.
"""


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver):
    model = llm.bind_tools(TOOLS)

    async def agent(state: HotelState) -> dict:
        system = SystemMessage(INSTRUCTIONS.format(today=date.today().isoformat()))
        return {"messages": [await model.ainvoke([system, *state["messages"]])]}

    g = StateGraph(HotelState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


def make_llm() -> BaseChatModel:
    """Provider is configuration, not code: LLM_MODEL=<provider>:<model>."""
    from langchain.chat_models import init_chat_model
    return init_chat_model(os.environ.get("LLM_MODEL", "openai:gpt-5-mini"))


@asynccontextmanager
async def open_checkpointer(url: str) ->  AsyncGenerator[BaseCheckpointSaver]:
    """memory | sqlite:<file> | postgresql://user:pass@host/db  (any cloud's managed Postgres)."""
    if url == "memory":
        yield InMemorySaver()
    elif url.startswith("sqlite:"):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(url.removeprefix("sqlite:")) as saver:
            yield saver
    elif url.startswith(("postgres://", "postgresql://")):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # pip install langgraph-checkpoint-postgres
        async with AsyncPostgresSaver.from_conn_string(url) as saver:
            await saver.setup()
            yield saver
    else:
        raise ValueError(f"Unsupported CHECKPOINTER: {url}")


# ---------------------------------------------------------------------------
# CLI chat
# ---------------------------------------------------------------------------
def _text(msg: Any) -> str:
    """Plain reply text; works whether a provider returns a string or content blocks."""
    text = str(getattr(msg, "text", "") or "")
    return text or str(msg.content)


async def run_turn(graph, prompt: str, config: dict, ask=input) -> dict:
    """One user turn, including any approval pauses. Returns the final graph state."""
    out = await graph.ainvoke({"messages": [HumanMessage(prompt)]}, config)
    while out.get("__interrupt__"):
        question = out["__interrupt__"][0].value["question"]
        answer = ask(f"\n[approval] {question} [y/N] ").strip().lower()
        out = await graph.ainvoke(Command(resume=answer in ("y", "yes")), config)
    return out


def show_results(state: dict, out_dir: str = ".") -> None:
    rows = [state["hotels"][i] for i in state.get("result_idx") or []]
    print(core.format_table(state["search"], rows, state.get("last_query") or {}, len(state["hotels"])))
    if rows:
        print(f"Saved to {core.save_csv(state['search'], rows, out_dir)}")


async def chat(thread: str) -> None:
    async with open_checkpointer(os.environ.get("CHECKPOINTER", "sqlite:hotel_agent.db")) as cp:
        graph = build_graph(make_llm(), cp)
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 30}
        prior = (await graph.aget_state(config)).values
        print(f"Hotel assistant [{os.environ.get('LLM_MODEL', 'openai:gpt-5-mini')}] thread '{thread}'"
              + (f" - resumed ({len(prior.get('messages', []))} messages)" if prior.get("messages") else "")
              + ". Empty line or 'quit' to exit.")
        while True:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in ("", "quit", "exit"):
                break
            before = (await graph.aget_state(config)).values.get("result_version") or 0
            state = await run_turn(graph, prompt, config)
            print(f"\nAgent: {_text(state['messages'][-1])}")
            if (state.get("result_version") or 0) != before and state.get("search"):
                show_results(state)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Conversational hotel agent (LangGraph)")
    ap.add_argument("--thread", default="default", help="conversation id; reuse it to resume")
    asyncio.run(chat(ap.parse_args().thread))