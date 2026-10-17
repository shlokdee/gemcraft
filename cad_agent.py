"""Image + text -> Gemma/Gemini -> SolidCraft MCP -> 3D model.

The model only plans and writes the `batch` (plus `measure` to check it).
The script itself starts the design, takes the screenshot and exports the files,
so a demo always ends with a picture and a STEP file.

Run:
    python cad_agent.py "phone stand: ..." [photo.png]

Environment options:
    CAD_MODEL=gemma-4-26b-a4b-it        model (default)
    CAD_FALLBACKS=gemini-2.5-flash      comma list, used after a 429 / unknown model
    CAD_KNOWLEDGE=reference.md          files appended to the system prompt ("none" = nothing)
    CAD_STEPS=10                        max model turns
    SOLIDCRAFT_CLI=/path/to/solidcraft-cli   (default ./solidcraft-cli)

Needs: pip install mcp google-genai python-dotenv   and GEMINI_API_KEY in .env
"""
import asyncio, base64, json, mimetypes, os, sys, time
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---- settings -------------------------------------------------------------
MODEL = os.environ.get("CAD_MODEL", "gemma-4-26b-a4b-it")
FALLBACKS = [m for m in os.environ.get("CAD_FALLBACKS", "").split(",") if m]
CLI = os.path.abspath(os.environ.get("SOLIDCRAFT_CLI", "./solidcraft-cli"))
OUT_DIR = os.path.abspath("out")
MAX_STEPS = int(os.environ.get("CAD_STEPS", "10"))
MAX_RESULT_CHARS = 1500
MAX_DESC_CHARS = 250
ALLOWED = {"batch", "measure", "list_commands"}      # the model needs nothing else
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM = """You build 3D parts in SolidCraft by calling tools. Units: mm. A new empty design is already open.
1. Work out the numbers first (sizes, positions, expected volume).
2. Call batch ONCE with every step, always with "verbose": false. Steps look like {"command":id,"params":{...}}. Example, a 40x30x20 block with a 10mm hole:
[{"command":"sketch.create","params":{"plane":"XY"}},
{"command":"sketch.rectangle.two_point","params":{"p0":[0,0],"p1":[40,30]}},
{"command":"sketch.finish","params":{}},
{"command":"solid.extrude","params":{"distance":20}},
{"command":"sketch.create","params":{"plane":"XY"}},
{"command":"sketch.circle.center","params":{"center":[20,15],"radius":5}},
{"command":"solid.extrude","params":{"distance":20,"operation":"cut"}}]
3. Call measure and compare the volume with your expected value.
4. If the batch failed, fix only the failing step and call batch again. If it succeeded and the volume matches, reply with one short sentence and stop.
Screenshot and export are done for you; never call them. Call list_commands only with a short filter word.""".strip()

_here = os.path.dirname(os.path.abspath(__file__))
_loaded = []
for _name in os.environ.get("CAD_KNOWLEDGE", "reference.md").split(","):
    _name = _name.strip()
    if not _name or _name.lower() == "none":
        continue
    _path = os.path.join(_here, _name)
    if os.path.exists(_path):
        SYSTEM += f"\n\n--- {_name} ---\n" + open(_path, encoding="utf-8").read()
        _loaded.append(_name)
    else:
        print(f"(note: knowledge file {_name} not found, skipping)")

# 60 s per request and NO hidden SDK retries: a slow or failing model fails fast and we switch models
client = genai.Client(http_options=types.HttpOptions(
    timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)))
T0 = time.time()
_models = [MODEL] + FALLBACKS
_model_idx = 0


# ---- progress helpers -----------------------------------------------------
def stamp():
    s = int(time.time() - T0)
    return f"[{s // 60:02d}:{s % 60:02d}]"


def log(msg):
    print(f"{stamp()} {msg}", flush=True)


def short(obj, n=160):
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


async def ticker(label):
    start = time.time()
    try:
        while True:
            print(f"\r{stamp()} ⏳ {label} {int(time.time() - start)}s ...", end="", flush=True)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("\r" + " " * 70 + "\r", end="", flush=True)
        raise


# ---- model plumbing -------------------------------------------------------
def thinking_for(model):
    if "gemma" in model:
        return types.ThinkingConfig(thinking_level="minimal")
    if "2.5-flash" in model:
        return types.ThinkingConfig(thinking_budget=0)
    return None


def make_config(model, decls):
    kw = dict(
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    tc = thinking_for(model)
    if tc is not None:
        kw["thinking_config"] = tc
    return types.GenerateContentConfig(**kw)


def to_declaration(t):
    schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
    return types.FunctionDeclaration(
        name=t.name, description=(t.description or "")[:MAX_DESC_CHARS],
        parameters_json_schema=schema)


def generate(contents, decls):
    global _model_idx
    for attempt in range(6):
        model = _models[_model_idx % len(_models)]
        try:
            return model, client.models.generate_content(
                model=model, contents=contents, config=make_config(model, decls))
        except Exception as e:
            msg = str(e)
            limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            missing = "404" in msg or "NOT_FOUND" in msg
            slow = any(k in msg.lower() for k in ("timeout", "timed out", "deadline", "unavailable", "internal"))
            if (limited or missing or slow) and len(_models) > 1:
                _model_idx += 1
                why = "rate-limited" if limited else "not found" if missing else "slow/erroring"
                print(f"\r{stamp()} ⚠ {model} {why} -> {_models[_model_idx % len(_models)]}", flush=True)
                time.sleep(2)
            else:
                wait = 15 if limited else 5 * (attempt + 1)
                print(f"\r{stamp()} ⚠ {model} error (try {attempt + 1}/6), waiting {wait}s: {short(msg, 160)}", flush=True)
                time.sleep(wait)
    raise RuntimeError("API kept failing")


async def ask_model(contents, decls, step):
    task = asyncio.create_task(ticker(f"Turn {step}/{MAX_STEPS}: model is thinking"))
    t = time.time()
    try:
        model, resp = await asyncio.to_thread(generate, contents, decls)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    u = getattr(resp, "usage_metadata", None)
    pt, tt = getattr(u, "prompt_token_count", None), getattr(u, "total_token_count", None)
    log(f"🧠 {model} answered in {time.time() - t:.1f}s (prompt {pt} tok, total {tt} tok)")
    return resp


def text_of(result):
    return "\n".join(c.text for c in result.content if c.type == "text") or "(no text)"


# ---- the part the script does itself: picture + files ----------------------
async def finalize(session, ok):
    log("Finishing up: screenshot + export" + ("" if ok else "  (model did not finish; showing what exists)"))
    try:
        res = await session.call_tool("screenshot", {"view": "iso", "source": "model", "width": 960, "height": 720})
        for c in res.content:
            if c.type == "image":
                path = os.path.join(OUT_DIR, "final.png")
                open(path, "wb").write(base64.b64decode(c.data))
                log(f"📷 {path}")
    except Exception as e:
        log(f"⚠ screenshot failed: {short(str(e), 200)}")
    for tool, name in (("export", "result.step"), ("export", "result.stl"), ("save", "result.solidcraft")):
        try:
            res = await session.call_tool(tool, {"path": os.path.join(OUT_DIR, name)})
            log(("✔ " if not res.is_error else "❌ ") + f"{name}" + ("" if not res.is_error else f": {short(text_of(res), 200)}"))
        except Exception as e:
            log(f"⚠ {name} failed: {short(str(e), 200)}")


# ---- main loop ------------------------------------------------------------
async def run(prompt, image_path=None):
    log(f"Model: {_models[0]}   fallbacks: {_models[1:] or 'none'}   knowledge: {_loaded or 'none'}")
    log(f"Request: {prompt}" + (f"  (+ image {image_path})" if image_path else ""))
    server = StdioServerParameters(command=CLI, args=["mcp", "--allow-path", OUT_DIR])
    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = [t for t in (await session.list_tools()).tools if t.name in ALLOWED]
            decls = [to_declaration(t) for t in tools]
            await session.call_tool("new_design", {})          # script starts the design, not the model
            log(f"Design ready. Tools for the model: {', '.join(t.name for t in tools)}")

            parts = [types.Part.from_text(text=prompt)]
            if image_path:
                mime = mimetypes.guess_type(image_path)[0] or "image/png"
                parts.append(types.Part.from_bytes(data=open(image_path, "rb").read(), mime_type=mime))
            contents = [types.Content(role="user", parts=parts)]

            ok, seen = False, {}
            for step in range(1, MAX_STEPS + 1):
                resp = await ask_model(contents, decls, step)
                contents.append(resp.candidates[0].content)
                calls = resp.function_calls
                if not calls:
                    ok = True
                    log(f"✅ Model finished: {short(resp.text or '', 200)}")
                    break

                reply_parts, stuck = [], False
                for call in calls:
                    args = dict(call.args or {})
                    if call.name == "batch":
                        args.setdefault("verbose", False)
                    sig = call.name + json.dumps(args, sort_keys=True, default=str)
                    seen[sig] = seen.get(sig, 0) + 1
                    if seen[sig] >= 3:
                        stuck = True
                    log(f"🔧 Turn {step}: {call.name}  {short(args, 200)}")
                    t = time.time()
                    result = await session.call_tool(call.name, args)
                    dt = time.time() - t
                    text = text_of(result)
                    if result.is_error:
                        log(f"   ❌ failed in {dt:.1f}s: {short(text, 300)}")
                        text = "ERROR: " + text
                    else:
                        log(f"   ✔ ok in {dt:.1f}s: {short(text, 140)}")
                    if len(text) > MAX_RESULT_CHARS:
                        text = text[:MAX_RESULT_CHARS] + " …[truncated]"
                    reply_parts.append(types.Part.from_function_response(
                        name=call.name, response={"result": text}))
                contents.append(types.Content(role="user", parts=reply_parts))
                if stuck:
                    log("⚠ The model is repeating the same call; stopping.")
                    break
            else:
                log("⚠ Hit the turn limit.")

            await finalize(session, ok)
            log(f"Total time {time.time() - T0:.1f}s. Files are in {OUT_DIR}")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else input("What should I build? ")
    image = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(run(prompt, image))