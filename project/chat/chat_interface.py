import json
import re
import threading
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, SystemMessage

from .session_memory import SessionMemoryStore

SILENT_NODES = {"recognize_intent", "rewrite_query", "plan_rag_tasks"}
SYSTEM_NODES = {"summarize_history", "recognize_intent", "rewrite_query", "plan_rag_tasks"}
MEMORY_WINDOW_SIZE = 5

SYSTEM_NODE_CONFIG = {
    "recognize_intent":   {"title": "🔍 Intent Recognition"},
    "rewrite_query":      {"title": "🔍 Query Rewriting"},
    "plan_rag_tasks":     {"title": "🧭 Task Planning"},
    "summarize_history":  {"title": "📋 Chat History Summary"},
}

# --- Helpers ---

def make_message(content, *, title=None, node=None):
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def find_msg_idx(messages, node):
    return next(
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def parse_rewrite_json(buffer):
    match = re.search(r"\{.*\}", buffer, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def format_rewrite_content(buffer):
    data = parse_rewrite_json(buffer)
    if not data:
        return "⏳ Rewriting query..."
    if data.get("is_clear"):
        lines = ["✅ **Retrieval query ready**"]
        if data.get("questions"):
            lines += ["\n**Rewritten queries:**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["❓ **Query needs clarification**"]
        clarification = data.get("clarification_needed", "")
        if clarification:
            lines.append(f"\nClarification needed: *{clarification}*")
    return "\n".join(lines)


def format_intent_content(buffer):
    data = parse_rewrite_json(buffer)
    if not data:
        return "⏳ Recognizing intent..."

    intent_type = data.get("intent_type", "unknown")
    lines = [f"**Intent:** `{intent_type}`"]
    if data.get("normalized_query"):
        lines.append(f"\n**Normalized query:** {data['normalized_query']}")
    if data.get("tasks"):
        lines.append("\n**Planned tasks:**")
        lines.extend(f"- {task.get('query', '')}" for task in data["tasks"])
    clarification = data.get("clarification_needed", "")
    if clarification and intent_type == "clarification":
        lines.append(f"\nClarification needed: *{clarification}*")
    return "\n".join(lines)

# --- End of Helpers ---

class ChatInterface:

    def __init__(self, rag_system, course_store=None, session_memory=None):
        self.rag_system = rag_system
        self.course_store = course_store
        self.session_memory = session_memory or SessionMemoryStore()

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer):
        """Update (or create) the collapsible system-node message and surface any clarification."""
        system_node_buffer[node] = system_node_buffer.get(node, "") + chunk.content
        buffer = system_node_buffer[node]
        title  = SYSTEM_NODE_CONFIG[node]["title"]
        if node == "recognize_intent":
            content = format_intent_content(buffer)
        elif node == "rewrite_query":
            content = format_rewrite_content(buffer)
        else:
            content = buffer

        idx = find_msg_idx(response_messages, node)
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))
        else:
            response_messages[idx]["content"] = content

        if node in ("recognize_intent", "rewrite_query"):
            self._surface_clarification(buffer, response_messages)

    def _surface_clarification(self, buffer, response_messages):
        """If the query is unclear, add/update a plain clarification message."""
        data          = parse_rewrite_json(buffer) or {}
        clarification = data.get("clarification_needed", "")
        if not data.get("is_clear") and clarification.strip().lower() not in ("", "no"):
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls):
        """Register new tool calls as collapsible messages."""
        for tc in chunk.tool_calls:
            if tc.get("id") and tc["id"] not in active_tool_calls:
                response_messages.append(
                    make_message(f"Running `{tc['name']}`...", title=f"🛠️ {tc['name']}")
                )
                active_tool_calls[tc["id"]] = len(response_messages) - 1

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls):
        """Fill in the tool result inside the matching collapsible message."""
        idx = active_tool_calls.get(chunk.tool_call_id)
        if idx is not None:
            preview = str(chunk.content)[:300]
            suffix  = "\n..." if len(str(chunk.content)) > 300 else ""
            response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"

    def _handle_llm_token(self, chunk, node, response_messages):
        """Append streaming LLM tokens to the last plain assistant message."""
        last = response_messages[-1] if response_messages else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))
        response_messages[-1]["content"] += chunk.content

    def _load_conversation_memory(self, session_id):
        try:
            recent_turns = self.session_memory.get_recent_turns(session_id, limit=MEMORY_WINDOW_SIZE)
            return self.session_memory.format_recent_turns(recent_turns)
        except Exception as e:
            print(f"Warning: Could not load session memory for {session_id}: {e}")
            return ""

    def _save_turn(self, session_id, user_original, assistant_final, course_name=None):
        try:
            self.session_memory.append_turn(
                session_id=session_id,
                user_original=user_original,
                assistant_final=assistant_final,
                course_name=course_name,
            )
        except Exception as e:
            print(f"Warning: Could not save session memory for {session_id}: {e}")

    def _generate_title_async(self, session_id: str):
        """Generate a concise session title using the LLM in a daemon thread."""
        try:
            turns = self.session_memory.get_session_turns(session_id)
            if not turns:
                return

            lines = []
            for turn in turns:
                user_text = (turn.get("user_original") or "")[:300]
                assistant_text = (turn.get("assistant_final") or "")[:300]
                lines.append(f"User: {user_text}")
                lines.append(f"Assistant: {assistant_text}")
            conversation_text = "\n".join(lines)

            prompt = (
                "Summarize the following conversation into a short, descriptive title.\n"
                "Rules:\n"
                "- Maximum 8 words.\n"
                "- Capture the main topic or question.\n"
                "- Return ONLY the title text. No quotes, no prefixes, no extra commentary.\n"
                "\n"
                "Conversation:\n"
                f"{conversation_text}\n"
                "\n"
                "Title:"
            )

            response = self.rag_system.llm.invoke([SystemMessage(content=prompt)])

            title = (response.content or "").strip().strip("\"'").strip()
            words = title.split()
            if len(words) > 8:
                title = " ".join(words[:8])

            if title:
                self.session_memory.update_session_title(session_id, title)
        except Exception as e:
            print(f"Warning: Title generation failed for session {session_id}: {e}")

    def _delete_session_memory(self, session_id):
        try:
            self.session_memory.delete_session(session_id)
        except Exception as e:
            print(f"Warning: Could not delete session memory for {session_id}: {e}")

    def _extract_final_response(self, response_messages):
        for msg in reversed(response_messages):
            if msg.get("role") != "assistant":
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            metadata = msg.get("metadata") or {}
            if not metadata or metadata.get("node") == "clarification":
                return content
        return ""

    def chat(self, message, history, course_name=None, session_id=None):
        """Generator that streams chat message dicts. Handles both initial
        messages and clarification-resume when the graph is interrupted."""
        if not self.rag_system.agent_graph:
            yield "⚠️ System not initialized!"
            return

        with self.rag_system.chat_lock:
            if self.course_store and course_name:
                self.rag_system.set_course_scope(self.course_store.source_files_for_course(course_name))
            else:
                self.rag_system.set_course_scope([])

            session_id    = session_id or self.rag_system.thread_id
            user_message  = message.strip()
            config        = self.rag_system.get_config(thread_id=session_id)
            graph         = self.rag_system.agent_graph

            snapshot = graph.get_state(config)
            is_resuming = bool(snapshot and snapshot.next)

            if is_resuming:
                graph.update_state(
                    config,
                    {"messages": [HumanMessage(content=user_message)]},
                )
                stream_input = None
            else:
                memory = self._load_conversation_memory(session_id)
                stream_input = {
                    "messages": [HumanMessage(content=user_message)],
                    "conversation_memory": memory,
                }

            response_messages  = []
            active_tool_calls  = {}
            system_node_buffer = {}

            try:
                for chunk, metadata in graph.stream(stream_input, config=config, stream_mode="messages"):
                    node = metadata.get("langgraph_node", "")

                    if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                        self._handle_system_node(chunk, node, response_messages, system_node_buffer)

                    elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        self._handle_tool_call(chunk, response_messages, active_tool_calls)

                    elif isinstance(chunk, ToolMessage):
                        self._handle_tool_result(chunk, response_messages, active_tool_calls)

                    elif isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES:
                        self._handle_llm_token(chunk, node, response_messages)

                    yield response_messages

                final_snapshot = graph.get_state(config)
                if final_snapshot and final_snapshot.next:
                    yield response_messages
                    return

                final_response = self._extract_final_response(response_messages)
                if final_response:
                    self._save_turn(session_id, user_message, final_response, course_name=course_name)

                threading.Thread(
                    target=self._generate_title_async,
                    args=(session_id,),
                    daemon=True,
                ).start()

            except Exception as e:
                yield f"❌ Error: {str(e)}"

    def clear_session(self, session_id=None):
        sid = session_id or self.rag_system.thread_id
        self._delete_session_memory(sid)
        self.rag_system.reset_thread(sid)
        self.rag_system.observability.flush()
