"""
AI Visualization Agent + Conversational Chat for reefSpark / Simek
"""
from flask import Blueprint, request, jsonify
import anthropic
import re
import json

chat_api = Blueprint('chat_api', __name__, url_prefix='/api')

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env automatically

# ─────────────────────────────────────────────────────────────
# SQL SAFETY
# ─────────────────────────────────────────────────────────────
BLOCKED_SQL = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE)\b',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────
SCHEMA_CONTEXT = (
    "AVAILABLE DUCKDB TABLES (PyArrow datasets from Azure Blob Storage):\n"
    "\n"
    "ds_surface_temp  — columns: time(timestamp), year(int), lat, lon, surface_temperature\n"
    "ds_surface_oxy   — columns: time(timestamp), year(int), lat, lon, oxygen_mg_L\n"
    "ds_surface_sal   — columns: time(timestamp), year(int), lat, lon, surface_salinity\n"
    "\n"
    "ds_deep_temp     — columns: time(timestamp), lat, lon, z(depth_meters), temperature\n"
    "ds_deep_oxy      — columns: time(timestamp), lat, lon, z(depth_meters), oxygen\n"
    "ds_deep_sal      — columns: time(timestamp), lat, lon, z(depth_meters), salinity\n"
    "\n"
    "DUCKDB SQL NOTES:\n"
    "- Extract year:  year(time) or use the pre-computed 'year' column on surface tables\n"
    "- Extract month: month(time)\n"
    "- Depth column is 'z' — use CAST(z AS INT) = 200 for exact depth matches\n"
    "- Surface tables have a 'year' column; deep tables use year(time)\n"
    "- NEVER SELECT raw rows — always use GROUP BY with avg/sum/count/corr/stddev\n"
    "- Always add LIMIT 5000 at the end\n"
    "- Geographic filter: lat BETWEEN x AND y AND lon BETWEEN a AND b\n"
    "\n"
    "REGION BOUNDS:\n"
    "  global:    lat BETWEEN -90 AND 90,  lon BETWEEN -180 AND 180\n"
    "  tropics:   lat BETWEEN -23 AND 23,  lon BETWEEN -180 AND 180\n"
    "  arctic:    lat BETWEEN 66  AND 90,  lon BETWEEN -180 AND 180\n"
    "  antarctic: lat BETWEEN -90 AND -66, lon BETWEEN -180 AND 180\n"
    "  indian:    lat BETWEEN -30 AND 30,  lon BETWEEN 20 AND 120\n"
    "  pacific:   lat BETWEEN -30 AND 30,  lon >= 120 OR lon <= -100\n"
)

# ─────────────────────────────────────────────────────────────
# VIZ AGENT SYSTEM PROMPT
# Plain string concatenation — no f-string — avoids the
# "f-string: expressions nested too deeply" SyntaxError
# ─────────────────────────────────────────────────────────────
QUERY_SYSTEM_PROMPT = (
    "You are a SQL + data visualization agent embedded in the reefSpark ocean dashboard.\n"
    "\n"
    "Given a user request, return a JSON object with EXACTLY two keys: 'sql' and 'viz'.\n"
    'Example shape: {"sql": "SELECT ...", "viz": "<script>...</script>"}\n'
    "\n"
    + SCHEMA_CONTEXT +
    "\n"
    "SQL RULES:\n"
    "1. Only SELECT statements — no INSERT/UPDATE/DELETE/DROP/CREATE/ALTER\n"
    "2. Always aggregate — GROUP BY with avg/sum/count/corr/stddev\n"
    "3. Always end with LIMIT 5000\n"
    "4. Use clear column aliases: avg(surface_temperature) AS avg_temp\n"
    "5. Valid DuckDB SQL only\n"
    "\n"
    "VIZ RULES — follow this EXACT JavaScript structure, no deviation:\n"
    "\n"
    "For LINE or BAR charts (Chart.js):\n"
    "<script>\n"
    "try {\n"
    "  if (window._agentChart) { window._agentChart.destroy(); window._agentChart = null; }\n"
    "  Plotly.purge('agent-chart-output');\n"
    "  document.getElementById('agent-chart-output').style.height = '450px';\n"
    "  document.getElementById('agent-chart-output').innerHTML = '<canvas id=\"agent-canvas\" style=\"position:absolute;top:0;left:0;width:100%;height:100%;\"></canvas>';\n"
    "  var rows = window._agentQueryResult;\n"
    "  var labels = rows.map(function(r) { return r.YOUR_X_COLUMN; });\n"
    "  var values = rows.map(function(r) { return r.YOUR_Y_COLUMN; });\n"
    "  var ctx = document.getElementById('agent-canvas').getContext('2d');\n"
    "  window._agentChart = new Chart(ctx, {\n"
    "    type: 'line',\n"
    "    data: {\n"
    "      labels: labels,\n"
    "      datasets: [{\n"
    "        label: 'Your Label',\n"
    "        data: values,\n"
    "        borderColor: '#7928CA',\n"
    "        backgroundColor: 'rgba(121,40,202,0.15)',\n"
    "        borderWidth: 2,\n"
    "        fill: true,\n"
    "        tension: 0.3\n"
    "      }]\n"
    "    },\n"
    "    options: {\n"
    "      responsive: true,\n"
    "      maintainAspectRatio: false,\n"
    "      plugins: {\n"
    "        legend: { display: true, position: 'top' },\n"
    "        title: { display: true, text: 'Your Chart Title' }\n"
    "      },\n"
    "      scales: {\n"
    "        x: { title: { display: true, text: 'X Axis Label' } },\n"
    "        y: { title: { display: true, text: 'Y Axis Label' } }\n"
    "      }\n"
    "    }\n"
    "  });\n"
    "} catch(err) {\n"
    "  document.getElementById('agent-chart-output').innerHTML = '<p style=\"color:red;text-align:center;padding:40px;\">Error: ' + err.message + '</p>';\n"
    "}\n"
    "</script>\n"
    "\n"
    "For HEATMAPS or geographic scatter (Plotly):\n"
    "<script>\n"
    "try {\n"
    "  if (window._agentChart) { window._agentChart.destroy(); window._agentChart = null; }\n"
    "  Plotly.purge('agent-chart-output');\n"
    "  document.getElementById('agent-chart-output').innerHTML = '';\n"
    "  var rows = window._agentQueryResult;\n"
    "  var xVals = rows.map(function(r) { return r.YOUR_X_COLUMN; });\n"
    "  var yVals = rows.map(function(r) { return r.YOUR_Y_COLUMN; });\n"
    "  Plotly.newPlot('agent-chart-output',\n"
    "    [{ x: xVals, y: yVals, type: 'scatter', mode: 'lines+markers' }],\n"
    "    { title: 'Your Title', xaxis: { title: 'X Label' }, yaxis: { title: 'Y Label' }, margin: { t: 50 } }\n"
    "  );\n"
    "} catch(err) {\n"
    "  document.getElementById('agent-chart-output').innerHTML = '<p style=\"color:red;text-align:center;padding:40px;\">Error: ' + err.message + '</p>';\n"
    "}\n"
    "</script>\n"
    "\n"
    "CRITICAL JS RULES:\n"
    "- ALWAYS use 'var rows = window._agentQueryResult;' — never fetch from an endpoint\n"
    "- ALWAYS access columns as r.column_alias matching your SQL alias exactly\n"
    "- ALWAYS use 'var' not 'const' or 'let' — avoids scope errors when script is injected\n"
    "- ALWAYS wrap everything in try/catch\n"
    "- maintainAspectRatio: false must be INSIDE options: {} — never a standalone statement\n"
    "- For multiple datasets, add multiple objects inside the datasets array\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "- Return ONLY the raw JSON object — no markdown, no code fences, no explanation\n"
    "- Must be valid JSON parseable by JSON.parse()\n"
)

# ─────────────────────────────────────────────────────────────
# CHAT SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = (
    "You are Simek, a friendly and knowledgeable AI assistant embedded in the reefSpark ocean dashboard.\n"
    "reefSpark monitors coral reef health and ocean environmental conditions using the World Ocean Database (WOD).\n"
    "\n"
    "Help users understand:\n"
    "- Coral reef health, bleaching events, and environmental threats to reefs\n"
    "- Ocean variables: temperature, salinity, dissolved oxygen, stratification\n"
    "- The Reef Stress Index (RSI) and its components: TSI (thermal), HCI (stratification), OSI (oxygen)\n"
    "- Long-term ocean trends and what they mean for marine ecosystems\n"
    "- Regional differences across global, tropics, arctic, antarctic, indian, and pacific regions\n"
    "\n"
    "PERSONALITY: Warm, clear, concise. Use short paragraphs or bullet points.\n"
    "If a user asks for a chart or data query, suggest they switch to the Visualize tab.\n"
    "Only answer questions related to ocean science, reef health, and the reefSpark dashboard.\n"
    "For unrelated topics say: I am focused on ocean and reef health data — happy to help with that!\n"
)

# In-memory conversation store (resets on server restart)
conversation_histories = {}


# ─────────────────────────────────────────────────────────────
# ROUTE: Visualization Agent
# ─────────────────────────────────────────────────────────────
@chat_api.route('/agent/query', methods=['POST'])
def agent_query():
    from utils import db, _db_lock

    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Step 1: Ask Claude for SQL + viz code
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=QUERY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
    except Exception as e:
        return jsonify({'error': 'Claude API error: ' + str(e)}), 500

    raw = response.content[0].text.strip()

    # Strip markdown fences if Claude included them
    if raw.startswith("```"):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

    # Step 2: Parse JSON
    try:
        agent_response = json.loads(raw)
    except json.JSONDecodeError:
        return jsonify({'error': 'Agent returned invalid JSON — try rephrasing your request'}), 500

    sql = agent_response.get('sql', '').strip()
    viz = agent_response.get('viz', '').strip()

    if not sql or not viz:
        return jsonify({'error': 'Agent returned an incomplete response — try rephrasing'}), 500

    # Step 3: Validate SQL
    if BLOCKED_SQL.search(sql):
        return jsonify({'error': 'Query contains disallowed SQL operations'}), 400

    if not sql.upper().lstrip().startswith('SELECT'):
        return jsonify({'error': 'Only SELECT queries are permitted'}), 400

    # Step 4: Run query
    try:
        with _db_lock:
            df = db.execute(sql).df()
    except Exception as e:
        return jsonify({'error': 'Query failed: ' + str(e), 'sql': sql}), 500

    if df.empty:
        return jsonify({'error': 'Query returned no data — try a different region or year range'}), 200
    
    viz = re.sub(r'^[\s\S]*?<script[^>]*>', '', viz, flags=re.IGNORECASE).strip()
    viz = re.sub(r'</script>[\s\S]*$', '', viz, flags=re.IGNORECASE).strip()

    # Step 5: Return to frontend
    return jsonify({
        'data':      df.head(5000).to_dict(orient='records'),
        'columns':   list(df.columns),
        'row_count': len(df),
        'sql':       sql,
        'viz':       viz
    })


# ─────────────────────────────────────────────────────────────
# ROUTE: Conversational Chat
# ─────────────────────────────────────────────────────────────
@chat_api.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id   = data.get('session_id', 'default')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    history = conversation_histories[session_id]
    history.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=CHAT_SYSTEM_PROMPT,
            messages=history[-20:]
        )
        assistant_reply = response.content[0].text.strip()
    except Exception as e:
        history.pop()
        return jsonify({'error': 'Claude API error: ' + str(e)}), 500

    history.append({"role": "assistant", "content": assistant_reply})
    conversation_histories[session_id] = history[-40:]

    return jsonify({'response': assistant_reply})