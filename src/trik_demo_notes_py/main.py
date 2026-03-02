"""
Demo Notes — conversational trik.

A personal notes assistant that manages notes with persistent storage.
Uses the wrap_agent() pattern for multi-turn conversation via handoff.
"""

from __future__ import annotations

import json

import time
from pathlib import Path
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from trikhub.sdk import wrap_agent, transfer_back_tool, TrikContext, TrikStorageContext


_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


# ============================================================================
# Note helpers
# ============================================================================


def _generate_id() -> str:
    return f"note_{int(time.time() * 1000):x}"


async def _find_note_by_title(
    title_search: str,
    storage: TrikStorageContext,
) -> Optional[dict[str, Any]]:
    index_raw = await storage.get("notes:index")
    index = index_raw if isinstance(index_raw, list) else []
    search_lower = title_search.lower()

    for note_id in index:
        note = await storage.get(f"notes:{note_id}")
        if note and isinstance(note, dict) and search_lower in note.get("title", "").lower():
            return note
    return None


async def _resolve_note(
    note_id: Optional[str],
    title_search: Optional[str],
    storage: TrikStorageContext,
) -> Optional[dict[str, Any]]:
    if note_id:
        return await storage.get(f"notes:{note_id}")
    if title_search:
        return await _find_note_by_title(title_search, storage)
    return None


# ============================================================================
# LangChain tool builders (closed over storage from context)
# ============================================================================


def _build_tools(storage: TrikStorageContext):

    @tool
    async def addNote(title: str, content: str) -> str:
        """Add a new note to persistent storage."""
        note_id = _generate_id()
        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        }

        await storage.set(f"notes:{note_id}", note)

        index_raw = await storage.get("notes:index")
        index = index_raw if isinstance(index_raw, list) else []
        index.append(note_id)
        await storage.set("notes:index", index)

        return json.dumps({"status": "created", "noteId": note_id, "title": title})

    @tool
    async def listNotes() -> str:
        """List all stored notes with their titles and IDs."""
        index_raw = await storage.get("notes:index")
        index = index_raw if isinstance(index_raw, list) else []

        if not index:
            return json.dumps({"count": 0, "notes": []})

        notes = []
        for note_id in index:
            note = await storage.get(f"notes:{note_id}")
            if note and isinstance(note, dict):
                notes.append({"id": note["id"], "title": note["title"]})

        return json.dumps({"count": len(notes), "notes": notes})

    @tool
    async def getNote(
        noteId: Optional[str] = None,
        titleSearch: Optional[str] = None,
    ) -> str:
        """Get a note by ID or title search."""
        note = await _resolve_note(noteId, titleSearch, storage)

        if not note:
            return json.dumps({"status": "not_found"})

        return json.dumps({
            "status": "found",
            "noteId": note["id"],
            "title": note["title"],
            "content": note["content"],
            "createdAt": note.get("createdAt", ""),
        })

    @tool
    async def updateNote(
        noteId: Optional[str] = None,
        titleSearch: Optional[str] = None,
        newTitle: Optional[str] = None,
        newContent: Optional[str] = None,
    ) -> str:
        """Update an existing note's title and/or content."""
        note = await _resolve_note(noteId, titleSearch, storage)
        if not note:
            return json.dumps({"status": "not_found"})

        if not newTitle and not newContent:
            return json.dumps({"status": "no_changes"})

        updated = {
            **note,
            "title": newTitle if newTitle else note["title"],
            "content": newContent if newContent else note["content"],
        }
        await storage.set(f"notes:{note['id']}", updated)

        return json.dumps({"status": "updated", "noteId": note["id"], "title": updated["title"]})

    @tool
    async def deleteNote(
        noteId: Optional[str] = None,
        titleSearch: Optional[str] = None,
    ) -> str:
        """Delete a note by ID or title search."""
        note = await _resolve_note(noteId, titleSearch, storage)
        if not note:
            return json.dumps({"status": "not_found"})

        await storage.delete(f"notes:{note['id']}")

        index_raw = await storage.get("notes:index")
        index = index_raw if isinstance(index_raw, list) else []
        await storage.set("notes:index", [i for i in index if i != note["id"]])

        return json.dumps({"status": "deleted", "noteId": note["id"], "title": note["title"]})

    return [addNote, listNotes, getNote, updateNote, deleteNote]


# ============================================================================
# Agent entry point
# ============================================================================


default = wrap_agent(lambda context: create_react_agent(
    model=ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=context.config.get("ANTHROPIC_API_KEY"),
    ),
    tools=[*_build_tools(context.storage), transfer_back_tool],
    prompt=_SYSTEM_PROMPT,
))
