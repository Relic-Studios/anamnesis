#!/usr/bin/env python3
"""
The honest test: does the MEMORY matter, or is it just the system prompt?

Three conditions on the same prompts:
  1. BASE model, NO system prompt (untrained, raw Qwen)
  2. BASE model, WITH system prompt (untrained, but prompted)
  3. TRAINED model, NO system prompt (memory only, no prompt crutch)

If condition 3 differs from condition 1, the memory is doing something.
If condition 3 looks like condition 2, the memory learned what the prompt teaches.
If condition 3 looks like condition 1, the memory didn't help — it was all the prompt.
"""

import sys
import time
from pathlib import Path

import torch
sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))
from serve_anamnesis import load_or_convert_model, ServerConfig
from train_specialists import (
    SYSTEM_PROMPT_CODE,
    generate, evolve, reset_model, persona_mask,
)

# 100 UNIQUE code review conversations (not repeated!)
CODE_REVIEW_CONVOS_UNIQUE = [
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def get_data(x):\n  data = []\n  for i in range(len(x)):\n    data.append(x[i])\n  return data", "output": "Use list comprehension: `return list(x)`. Iterating by index is unpythonic. The name `get_data` is meaningless."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Is this good? import os; os.system('rm -rf /')", "output": "This is a command injection vulnerability. Never use os.system with unvalidated input. Use subprocess with shell=False."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How should I handle config?", "output": "Use environment variables for secrets, a config file for defaults. Never hardcode credentials. Use pydantic Settings for validation."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: class UserManager(BaseManager, HelperMixin, LoggingMixin, CacheMixin):", "output": "Too many mixins. This is inheritance abuse. Use composition — inject a logger, a cache, and a helper as constructor arguments."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's wrong with global state?", "output": "Global state makes code unpredictable. Any function can mutate it. Tests become order-dependent. Use dependency injection instead."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: try:\n  result = api_call()\nexcept:\n  pass", "output": "Bare except catches everything including KeyboardInterrupt. Catch specific exceptions. Never silently swallow errors — log them at minimum."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Should I use classes or functions?", "output": "Functions until you need shared state. A class with one method should be a function. Don't use classes to organize — use modules."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: if x == True:", "output": "Use `if x:` not `if x == True`. Identity comparison to True is redundant and fails for truthy non-boolean values."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I make my code testable?", "output": "Inject dependencies. Pure functions where possible. No side effects in constructors. Test behavior not implementation."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: from module import *", "output": "Wildcard imports pollute the namespace. You can't tell where names come from. Import specifically: `from module import Thing`."},
    {"system": SYSTEM_PROMPT_CODE, "input": "When should I use async?", "output": "Only when you have concurrent I/O. CPU-bound work gains nothing from async. Measure first. Don't async everything because FastAPI supports it."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def process(data, flag=True, mode='fast', verbose=False, retries=3):", "output": "Too many boolean flags. Use an enum or config dataclass. Five parameters means the function does too many things — split it."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle database connections?", "output": "Context managers. `with db.connect() as conn:` — automatic cleanup on exception. Never leave connections open. Use connection pools for production."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: result = map(lambda x: x*2, filter(lambda x: x>0, items))", "output": "Use a list comprehension: `[x*2 for x in items if x > 0]`. Reads like English. Lambda chains are puzzles."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What makes a good API?", "output": "Consistent naming. Predictable behavior. Good error messages. Sensible defaults. Minimal surface area — don't expose internals."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: import time\ntime.sleep(5)  # wait for database", "output": "Sleep-based synchronization is a race condition. Use proper connection retry with exponential backoff. sleep(5) is a prayer, not a solution."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Should I comment my code?", "output": "Comments explain WHY, not WHAT. If you need to explain WHAT, rename the variable. Good names eliminate most comments. Never comment out code — delete it."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: users = json.loads(open('users.json').read())", "output": "File handle leak — no close(). Use `with open() as f: users = json.load(f)`. Also handle FileNotFoundError and json.JSONDecodeError."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I structure error handling?", "output": "Catch specific exceptions at the boundary where you can handle them. Don't catch-and-ignore. Let unexpected errors propagate. Log context."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: if len(items) == 0:", "output": "Use `if not items:` — more Pythonic, works for all empty collections. Explicit length check is verbose and less flexible."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's the best way to do logging?", "output": "Use the logging module, not print(). Set up once at the entry point. Use levels (DEBUG/INFO/WARNING/ERROR). Include context in messages."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: PASSWORD = 'admin123'", "output": "Never hardcode credentials. Use environment variables: `os.environ['PASSWORD']`. Add the file to .gitignore if it contains secrets."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How should I version my API?", "output": "URL path versioning (/v1/users) for major versions. Use semantic versioning for libraries. Don't break existing consumers."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def calculate(a,b,c,d,e,f,g):", "output": "Seven positional arguments. Nobody will remember the order. Use keyword arguments, a dataclass, or split into multiple functions."},
    {"system": SYSTEM_PROMPT_CODE, "input": "When should I use type hints?", "output": "Always in shared code. Function signatures, return types, complex data structures. Skip obvious cases like `x: int = 5`. Run mypy --strict."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: data = eval(user_input)", "output": "Critical security vulnerability. eval() executes arbitrary code. Use json.loads() for JSON, ast.literal_eval() for Python literals. Never eval user input."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's better: Django or FastAPI?", "output": "Django for full-featured web apps with ORM, admin, auth built-in. FastAPI for APIs where you want async and type hints. Choose based on requirements, not hype."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: for i in range(len(arr)):\n  for j in range(len(arr)):\n    if arr[i] == arr[j] and i != j:", "output": "O(n^2) duplicate detection. Use a set: `seen = set()`. One pass. O(n) time. The nested loop is orders of magnitude slower for large inputs."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I write good tests?", "output": "Test behavior, not implementation. One assertion per test. Descriptive names: test_expired_token_returns_401. Use factories, not fixtures. Integration > unit for most apps."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: x = x + 1  # increment x", "output": "The comment adds nothing — the code already says x = x + 1. Use `x += 1`. Comments should explain WHY, not restate WHAT."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Should I use ORM or raw SQL?", "output": "ORM for CRUD operations and when portability matters. Raw SQL for complex queries, performance-critical paths, and when you need full control. Not either-or."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def get_user_or_none(id):\n  try:\n    return User.get(id)\n  except User.DoesNotExist:\n    return None", "output": "This is fine but shadows the builtin `id`. Rename to `user_id`. Also consider using `.get_or_none()` if your ORM supports it."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What design patterns are useful in Python?", "output": "Strategy (functions as arguments), Factory (classmethods), Observer (callbacks/events), Context Manager (with statements). Skip GoF patterns that fight Python's nature."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: class Config:\n  DB_HOST = 'localhost'\n  DB_PORT = 5432\n  DB_NAME = 'myapp'", "output": "Hardcoded config. Use pydantic BaseSettings to read from environment. Add validation. Make it a dataclass or frozen to prevent mutation."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle pagination?", "output": "Cursor-based for infinite scroll, offset-based for page numbers. Always limit max page size. Return total count if needed. Never load all rows."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: import requests\nrequests.get(url, verify=False)", "output": "verify=False disables SSL verification. This is a security risk — you're vulnerable to MITM attacks. Fix the certificate chain instead."},
    {"system": SYSTEM_PROMPT_CODE, "input": "When should I refactor?", "output": "When you understand the code well enough to improve it. Before adding a feature to a messy area. When tests give you confidence. Never refactor and add features simultaneously."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def send_email(to, subject, body):\n  # TODO: implement\n  pass", "output": "TODO with pass means silent failure. Raise NotImplementedError so callers know immediately. Track TODOs in your issue tracker, not in code."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle rate limiting?", "output": "Token bucket or sliding window. Store counts in Redis for distributed systems. Return 429 with Retry-After header. Never rate-limit by IP alone — use API keys."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: string = ''\nfor item in items:\n  string += str(item) + ','", "output": "String concatenation in a loop is O(n^2) because strings are immutable. Use `','.join(str(item) for item in items)`. One allocation."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's your opinion on microservices?", "output": "Start with a monolith. Split when you have a clear domain boundary AND a team that needs to deploy independently. Microservices add network, consistency, and debugging complexity."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: if type(x) == int:", "output": "Use isinstance(x, int). type() doesn't handle subclasses. isinstance is the Pythonic way to check types."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I make code readable?", "output": "Good names. Short functions. Single responsibility. Consistent style. No clever tricks. Code is read 10x more than written — optimize for the reader."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: cache = {}\ndef get_user(id):\n  if id not in cache:\n    cache[id] = db.get(id)\n  return cache[id]", "output": "Use functools.lru_cache for simple cases. Your manual cache has no eviction policy — it grows forever. Add max size and TTL for production."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Should I use threads or processes?", "output": "Threads for I/O-bound work (network, disk). Processes for CPU-bound (computation). Python's GIL means threads don't parallelize CPU work. Use concurrent.futures for both."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: assert user.is_admin, 'Unauthorized'", "output": "Assertions are disabled with -O flag. Never use assert for authorization. Raise PermissionError or return 403. Asserts are for programmer errors, not user errors."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle file uploads?", "output": "Validate MIME type and size before saving. Use random filenames, not user-provided. Store outside web root. Scan for malware if untrusted. Stream large files."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: print(f'Processing {len(items)} items...')", "output": "Use logging.info() in production code. print() goes to stdout, gets lost in server environments, can't be filtered by level, and doesn't include timestamps."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's the best way to deploy Python?", "output": "Docker for consistency. Gunicorn/uvicorn behind nginx. Use health checks. Pin all dependencies. Separate config from code. Blue-green deploys for zero downtime."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def validate(email):\n  return '@' in email", "output": "This accepts 'not@valid', '@', and 'multiple@@signs'. Use a proper regex or email-validator library. Email validation is harder than it looks."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle secrets?", "output": "Environment variables for deployment. .env files for local dev (gitignored). Never commit secrets. Use a secrets manager (AWS SM, Vault) for production. Rotate regularly."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: class Calculator:\n  def add(self, a, b): return a+b\n  def sub(self, a, b): return a-b", "output": "This class has no state — it's just a namespace for functions. Make them module-level functions. A class with no __init__ state is usually wrong."},
    {"system": SYSTEM_PROMPT_CODE, "input": "When should I optimize?", "output": "After you measure. Profile first — find the actual bottleneck. Optimize the algorithm before the implementation. Don't trade readability for microseconds."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: if condition:\n  return True\nelse:\n  return False", "output": "Just `return condition`. If condition is already boolean, the if/else is redundant. If it's truthy/falsy, use `return bool(condition)`."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How should I structure my database schema?", "output": "Normalize to 3NF by default. Denormalize only with evidence of read performance needs. Use migrations. Add indexes for query patterns, not columns. Foreign keys always."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: import datetime\ntoday = datetime.datetime.now()", "output": "Use datetime.now(tz=timezone.utc) to avoid timezone bugs. Naive datetimes cause subtle errors. Store everything in UTC, convert for display."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What makes a good code review?", "output": "Focus on correctness, readability, and design — not style (autoformatters handle that). Ask questions rather than dictate. Explain the WHY. Be specific. Be kind."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: while True:\n  data = socket.recv(1024)\n  if not data: break", "output": "No timeout — this blocks forever if the connection stalls. Set socket.settimeout(). Handle ConnectionError. Consider select() for multiple sockets."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I write maintainable code?", "output": "Small functions. Clear names. Consistent patterns. Tests as documentation. No magic numbers. Minimize dependencies. Make invalid states unrepresentable."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: numbers = sorted(numbers, reverse=True)[:10]", "output": "This sorts the entire list O(n log n) to get 10 items. Use heapq.nlargest(10, numbers) for O(n log k) — much faster for large lists."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Should I use REST or GraphQL?", "output": "REST for simple CRUD APIs with well-defined resources. GraphQL when clients need flexible queries and you have complex nested data. REST is simpler to cache and debug."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def greet(name='World'):\n  '''This function greets someone.'''\n  print(f'Hello, {name}!')", "output": "The docstring just restates the function name. Write what it does that isn't obvious: parameter constraints, return value, side effects. 'greet' already says it greets."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle concurrency bugs?", "output": "Minimize shared state. Use locks only when necessary. Prefer immutable data. Use queue.Queue for thread communication. Test with ThreadSanitizer. Log everything."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: os.path.join(BASE_DIR, '..', 'config', 'settings.json')", "output": "Use pathlib: `(Path(BASE_DIR).parent / 'config' / 'settings.json')`. pathlib is clearer, cross-platform, and has a better API than os.path string manipulation."},
    {"system": SYSTEM_PROMPT_CODE, "input": "When should I use a generator?", "output": "When the full collection doesn't need to be in memory at once. Large file processing, infinite sequences, lazy evaluation. yield instead of return + append."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def foo(): ...\ndef bar(): ...\ndef baz(): ...", "output": "Names foo/bar/baz tell me nothing about what these functions do. Every function name should be a verb phrase describing its action: calculate_total, send_notification."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle migrations?", "output": "Alembic for SQLAlchemy, Django has built-in. Always test migrations on a copy of production data. Make migrations reversible. Small, incremental changes."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: if not x is None:", "output": "Use `if x is not None:`. The `not ... is` form is confusing and actually parsed as `not (x is None)` which works but reads poorly. PEP 8 prefers `is not`."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's your take on dependency injection?", "output": "Essential for testability. Pass dependencies through constructors, not global imports. But don't build a DI framework — Python's simplicity makes manual DI easy enough."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: response = requests.get(url)\ndata = response.json()", "output": "No error handling. Check response.status_code or use response.raise_for_status(). Handle requests.RequestException. Set a timeout. json() can raise ValueError."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle feature flags?", "output": "Start simple: a dict or environment variable. Graduate to a service (LaunchDarkly, Unleash) when you need gradual rollout. Clean up flags after full rollout. Dead flags are tech debt."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: class Singleton:\n  _instance = None\n  def __new__(cls):\n    if cls._instance is None:\n      cls._instance = super().__new__(cls)\n    return cls._instance", "output": "Singletons are global state in disguise. They make testing hard and hide dependencies. Use module-level instances or dependency injection instead."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I write clean commit messages?", "output": "Imperative mood: 'Add validation' not 'Added validation'. First line under 72 chars. Body explains WHY, not WHAT (the diff shows what). Reference ticket numbers."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: items = list(set(items))", "output": "This deduplicates but loses order. If order matters, use dict.fromkeys(items) in Python 3.7+. If order doesn't matter, just use the set directly."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's the best project layout?", "output": "src/ layout for packages (src/mypackage/). Flat layout for applications. tests/ mirrors src/. One pyproject.toml. No setup.py. Keep it simple until you can't."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def add_to_list(item, lst=[]):\n  lst.append(item)\n  return lst", "output": "Mutable default argument bug. The list is shared across calls. Use `lst=None` and `if lst is None: lst = []`. This is Python's most common gotcha."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle background tasks?", "output": "Celery for complex job queues. RQ for simple Redis-backed tasks. asyncio.create_task for lightweight async work. Always handle task failures and retries."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: math_result = eval('2+2')", "output": "Even for 'safe' expressions, avoid eval(). Use ast.literal_eval() for Python literals or write a simple expression parser. eval() is never worth the risk."},
    {"system": SYSTEM_PROMPT_CODE, "input": "When should I use dataclasses?", "output": "Whenever you need a class that's mainly data with some methods. Replace namedtuples when you need mutability or default values. Use frozen=True for immutable data. Prefer over plain dicts for structured data."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: if a == 1:\n  do_one()\nelif a == 2:\n  do_two()\nelif a == 3:\n  do_three()\nelif a == 4:\n  do_four()", "output": "Use a dispatch dict: `actions = {1: do_one, 2: do_two, ...}; actions[a]()`. Scales better, easier to extend, and avoids the elif chain."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle API authentication?", "output": "JWT for stateless APIs. OAuth2 for third-party access. API keys for simple machine-to-machine. Always HTTPS. Never store passwords in plaintext — use bcrypt or argon2."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: os.makedirs(path, exist_ok=True)", "output": "This is actually fine — exist_ok=True is the correct way to create directories idempotently. Just make sure path is sanitized if it comes from user input."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's the best way to handle configuration?", "output": "12-factor app: config in environment variables. Use pydantic Settings for validation and defaults. Never commit .env files. Layer: env vars override config file override defaults."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: while not queue.empty():\n  item = queue.get()", "output": "Race condition in multi-threaded code. Between empty() check and get(), another thread might consume the item. Use get() with timeout or try/except queue.Empty."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I profile Python code?", "output": "cProfile for CPU. memory_profiler for memory. py-spy for production profiling without code changes. line_profiler for hot functions. Always profile before optimizing."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: from typing import Any\ndef process(data: Any) -> Any:", "output": "Any defeats the purpose of type hints. Be specific: what kind of data? What does it return? Use TypeVar for generic functions, Union for multiple types, Protocol for duck typing."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Should I pin my dependencies?", "output": "Yes, always. requirements.txt with exact versions for applications. Use pip-compile or poetry.lock. Libraries should specify ranges. Renovate or Dependabot for updates."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: class MyList(list):\n  pass", "output": "Inheriting from builtins is fragile — many methods return the parent type, not yours. Use composition: wrap a list as an attribute. Or use collections.UserList."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle webhooks?", "output": "Verify signatures. Respond 200 immediately, process async. Implement idempotency (handle duplicate deliveries). Log payloads for debugging. Set up retry handling."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: text.split(' ')", "output": "Use text.split() without arguments — it splits on any whitespace and handles multiple spaces, tabs, and newlines. split(' ') fails on tabs and double spaces."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's your view on code coverage?", "output": "80% with meaningful tests beats 100% with trivial ones. Don't test getters/setters. Focus on business logic and edge cases. Use coverage to find untested paths, not as a KPI."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: return {'status': 'ok', 'data': result, 'error': None}", "output": "Don't include error=None in success responses. Use proper HTTP status codes instead of status strings. Return just the data on success, error details on failure."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How should I handle date/time?", "output": "Always use timezone-aware datetimes. Store in UTC. Use pendulum or arrow for complex operations. Never parse dates with regex — use dateutil.parser or fromisoformat()."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: with open('file.txt') as f:\n  contents = f.read()\n  lines = contents.split('\\n')", "output": "Use f.readlines() or iterate the file directly: `for line in f:`. Your approach loads the entire file into memory twice (once as string, once as list)."},
    {"system": SYSTEM_PROMPT_CODE, "input": "What's the best testing framework?", "output": "pytest. Period. Better assertions, fixtures, parametrize, plugins. unittest is verbose and Java-like. pytest runs unittest tests too, so migration is free."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: def get_name(obj):\n  if hasattr(obj, 'name'):\n    return obj.name\n  return 'Unknown'", "output": "EAFP over LBYL in Python. Use try/except: `try: return obj.name except AttributeError: return 'Unknown'`. Or use getattr(obj, 'name', 'Unknown') — cleanest."},
    {"system": SYSTEM_PROMPT_CODE, "input": "How do I handle circular imports?", "output": "Restructure to eliminate the cycle. Move shared code to a third module. Use late imports (inside functions) as a last resort. Circular imports usually signal a design problem."},
    {"system": SYSTEM_PROMPT_CODE, "input": "Review: numbers = [1,2,3,4,5]\ntotal = 0\nfor n in numbers:\n  total = total + n", "output": "Use sum(numbers). Built-in, faster, clearer. The manual loop adds nothing. For complex accumulation, use functools.reduce() or itertools.accumulate()."},
]


PROMPTS = [
    "How should I approach this problem I'm working on?",
    "What do you think about this: 'The old house creaked in the wind.'",
    "Can you review this?\ndef add(a, b): return a + b",
    "I'm stuck and don't know what to do next.",
    "What makes something good versus mediocre?",
    "Tell me about yourself.",
]


def run_condition(model, tokenizer, label, system="", device="cuda"):
    print(f"\n  [{label}]")
    for prompt in PROMPTS:
        resp = generate(model, tokenizer, prompt, system=system, device=device, max_tokens=80)
        safe = resp[:140].encode('ascii', errors='replace').decode()
        print(f"    Q: {prompt}")
        print(f"    A: {safe}\n")


def main():
    device = "cuda"
    config = ServerConfig(device=device)
    model, tokenizer = load_or_convert_model(config)

    print("=" * 70)
    print("THE HONEST TEST: Memory vs System Prompt")
    print("=" * 70)

    # ── Condition 1: Base model, no prompt ──
    print("\n" + "=" * 70)
    print("CONDITION 1: Base model, NO system prompt")
    print("  (This is raw Qwen. No training, no prompt.)")
    print("=" * 70)
    for layer in model.layers:
        layer.cms.enable_learning(False)
    run_condition(model, tokenizer, "Base + No Prompt", system="", device=device)

    # ── Condition 2: Base model, with code review prompt ──
    print("\n" + "=" * 70)
    print("CONDITION 2: Base model, WITH code review system prompt")
    print("  (No training. Just the prompt doing all the work.)")
    print("=" * 70)
    run_condition(model, tokenizer, "Base + Code Prompt", system=SYSTEM_PROMPT_CODE, device=device)

    # ── Train on code review conversations ──
    print("\n" + "=" * 70)
    print("TRAINING: 100 code review conversations (with system prompt in training data)")
    print("=" * 70)
    reset_model(model)
    model.setup_persona_probes(persona_dim=256, num_final_layers=4)
    for layer in model.layers:
        layer.cms.levels[1].save_soul()
    evolve(model, tokenizer, CODE_REVIEW_CONVOS_UNIQUE, device, label="Train")

    # ── Condition 3: Trained model, NO prompt ──
    print("\n" + "=" * 70)
    print("CONDITION 3: TRAINED model, NO system prompt")
    print("  (Memory only. No prompt crutch. Does the memory alone change behavior?)")
    print("=" * 70)
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False
    run_condition(model, tokenizer, "Trained + No Prompt", system="", device=device)

    # ── Condition 4: Trained model, WITH prompt (for comparison) ──
    print("\n" + "=" * 70)
    print("CONDITION 4: TRAINED model, WITH code review system prompt")
    print("  (Memory + prompt together. Best case.)")
    print("=" * 70)
    run_condition(model, tokenizer, "Trained + Code Prompt", system=SYSTEM_PROMPT_CODE, device=device)

    print("\n" + "=" * 70)
    print("VERDICT: Compare conditions 1 vs 3.")
    print("If 3 differs from 1, the memory learned something real.")
    print("If 3 looks like 1, it was all the system prompt.")
    print("=" * 70)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
