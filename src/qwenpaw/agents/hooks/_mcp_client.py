"""HTTP MCP client for calling mempalace MCP server from hooks."""
import json
import requests

MCP_URL = "http://localhost:8999/mcp"
_session_id = None


def mcp_call(tool_name, params=None):
    """Call a mempalace MCP tool via HTTP. Returns parsed result dict."""
    global _session_id
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if _session_id:
        headers["Mcp-Session-Id"] = _session_id

    # Init session if needed
    if not _session_id:
        try:
            resp = requests.post(MCP_URL, headers=headers, json={
                "jsonrpc": "2.0", "id": "init", "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "qwenpaw-bgsave", "version": "1.0"}},
            }, timeout=10)
            sid = resp.headers.get("Mcp-Session-Id", "")
            if sid:
                _session_id = sid
                headers["Mcp-Session-Id"] = sid
        except Exception as e:
            return {"error": f"MCP init failed: {e}"}

    # Call tool
    try:
        resp = requests.post(MCP_URL, headers=headers, json={
            "jsonrpc": "2.0", "id": "call", "method": "tools/call",
            "params": {"name": tool_name, "arguments": params or {}},
        }, timeout=30)
        for line in resp.content.decode("utf-8").split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "result" in data:
                    content = data["result"].get("content", [])
                    if content and isinstance(content[0], dict):
                        return json.loads(content[0].get("text", "{}"))
                elif "error" in data:
                    return {"error": data["error"].get("message", "unknown")}
        return {"error": "no result"}
    except Exception as e:
        return {"error": str(e)}
