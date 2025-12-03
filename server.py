# server.py
import json, os, threading, sqlite3, time, subprocess, shlex
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import io

ROOT = Path(__file__).parent.resolve()
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DB_PATH = ROOT / "xgent_ai.sqlite3"

# -------------------- SQLite login --------------------

def init_db():
    need_seed = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)')
    if need_seed:
        try:
            cur.execute('INSERT INTO users(username,password) VALUES (?,?)', ("xgent","xgent123"))
        except Exception:
            pass
    conn.commit()
    conn.close()

def check_login(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id FROM users WHERE username=? AND password=?', (username, password))
    row = cur.fetchone()
    conn.close()
    return row is not None

# -------------------- BGE embedding engine --------------------

_EMBEDDER = None
_EMBED_LOCK = threading.Lock()

def _load_bge_model():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBED_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers not installed. Run 'pip install sentence-transformers torch'."
            ) from e
        emb_cfg = CONFIG.get("embedding", {})
        model_name = emb_cfg.get("local_path") or emb_cfg.get("model") or "BAAI/bge-base-en-v1.5"
        print(f"[BGE] Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        _EMBEDDER = model
        return _EMBEDDER

def embed_texts(texts):
    if not texts:
        return []
    model = _load_bge_model()
    embs = model.encode(
        texts,
        batch_size=CONFIG.get("embedding", {}).get("batch_size", 32),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [list(map(float, v)) for v in embs]

def _np():
    import numpy as _np
    return _np

class EmbeddingIndex:
    def __init__(self, prefix: Path, dim: int = 768):
        self.prefix = Path(prefix)
        self.dim = dim
        self.emb = None
        self.meta = []
        self._loaded = False

    @property
    def npz_path(self):
        return self.prefix.with_suffix(".npz")

    @property
    def meta_path(self):
        return self.prefix.with_suffix(".json")

    def load(self):
        if self._loaded:
            return
        np = _np()
        if self.npz_path.exists():
            data = np.load(str(self.npz_path))
            self.emb = data["emb"]
        else:
            self.emb = None
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.meta = []
        self._loaded = True

    def save(self):
        np = _np()
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        if self.emb is None:
            arr = np.zeros((0, self.dim), dtype="float32")
        else:
            arr = self.emb
        np.savez(str(self.npz_path), emb=arr)
        self.meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def add(self, texts, meta_list):
        assert len(texts) == len(meta_list), "texts/meta length mismatch"
        self._ensure_loaded()
        vecs = embed_texts(texts)
        np = _np()
        arr = np.array(vecs, dtype="float32")
        if self.emb is None or len(self.emb) == 0:
            self.emb = arr
        else:
            self.emb = np.vstack([self.emb, arr])
        self.meta.extend(meta_list)
        self.save()

    def search(self, query_text, top_k=5):
        self._ensure_loaded()
        if self.emb is None or len(self.meta) == 0:
            return []
        q_vec = embed_texts([query_text])[0]
        np = _np()
        q = np.array(q_vec, dtype="float32")
        # compute dot-product similarity (embeddings normalized)
        scores = self.emb @ q
        idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in idx:
            i = int(i)
            m = dict(self.meta[i])
            m["score"] = float(scores[i])
            results.append(m)
        return results

# indices
JIRA_INDEX = EmbeddingIndex(ROOT / "indices" / "jira_bge", dim=CONFIG.get("embedding", {}).get("dim", 1024))
PDF_INDEX  = EmbeddingIndex(ROOT / "indices" / "pdf_bge", dim=CONFIG.get("embedding", {}).get("dim", 1024))
TC_INDEX   = EmbeddingIndex(ROOT / "indices" / "tc_bge", dim=CONFIG.get("embedding", {}).get("dim", 1024))
LOG_INDEX  = EmbeddingIndex(ROOT / "indices" / "log_bge", dim=CONFIG.get("embedding", {}).get("dim", 1024))
SOL_INDEX  = EmbeddingIndex(ROOT / "indices" / "solution_bge", dim=CONFIG.get("embedding", {}).get("dim", 1024))

# -------------------- PDF handling --------------------

def extract_pdf_chunks(pdf_bytes: bytes, doc_name: str, max_chars=900):
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise RuntimeError("PyPDF2 not installed. Run 'pip install PyPDF2'.") from e
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    doc_id = f"PDF-{int(time.time())}"
    buf = ""
    page_no = 0
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        page_no = i + 1
        for line in txt.splitlines():
            if len(buf) + len(line) + 1 > max_chars:
                if buf.strip():
                    chunks.append((buf.strip(), page_no))
                buf = ""
            buf += line + "\n"
        if buf.strip():
            chunks.append((buf.strip(), page_no))
            buf = ""
    texts, metas = [], []
    for idx, (txt, pg) in enumerate(chunks):
        texts.append(txt)
        metas.append({
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page": pg,
            "chunk_index": idx,
            "text": txt,
        })
    return texts, metas

def summarize_pdf_docs():
    PDF_INDEX._ensure_loaded()
    docs = {}
    for m in PDF_INDEX.meta:
        key = (m.get("doc_id"), m.get("doc_name"))
        docs.setdefault(key, 0)
        docs[key] += 1
    res = []
    for (doc_id, name), cnt in docs.items():
        res.append({"id": doc_id, "name": name, "chunks": cnt})
    return res

# -------------------- Testcase helper --------------------

def build_tc_text(row):
    return " ".join([
        row.get("id",""),
        row.get("name",""),
        row.get("uvm_version",""),
        row.get("summary",""),
        row.get("description",""),
        row.get("code",""),
    ])

# -------------------- CCF helper --------------------

def parse_ccf_bins(text: str):
    keys = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        low = s.lower()
        if "bin" in low or "coverpoint" in low or "covergroup" in low:
            keys.add(" ".join(s.split()))
    return keys

# -------------------- Debug recipes (YAML) --------------------

def load_debug_recipes():
    cfg = CONFIG.get("debug_recipes", {})
    yaml_path = cfg.get("yaml_path")
    if not yaml_path:
        return {}
    path = ROOT / yaml_path
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("pyyaml not installed. Run 'pip install pyyaml'.") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))

# -------------------- LLM adapter (llama_cpp GGUF) --------------------

_LLM = None
_LLM_LOCK = threading.Lock()

def _load_llm():
    """
    Lazy-load GGUF model via llama_cpp.Llama
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    with _LLM_LOCK:
        if _LLM is not None:
            return _LLM
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError("llama_cpp (llama-cpp-python) is not installed. Run 'pip install llama-cpp-python'.") from e

        cfg = CONFIG.get("llm", {})
        model_path = str((ROOT / cfg.get("model_path", "models/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf")).resolve())
        n_ctx = int(cfg.get("n_ctx", 4096))
        n_threads = int(cfg.get("n_threads", max(1, os.cpu_count()//2)))
        # load model
        print(f"[LLM] Loading GGUF model from {model_path} (n_ctx={n_ctx}, threads={n_threads})")
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            use_mlock=False,
            use_mmap=True,
        )
        _LLM = llm
        return _LLM

def llm_generate(prompt: str, max_tokens: int = None, temperature: float = None):
    cfg = CONFIG.get("llm", {})
    adapter = cfg.get("adapter", "llama_cpp")
    if adapter != "llama_cpp":
        return "[No llama_cpp adapter configured]"
    llm = _load_llm()
    temperature = 0.08 if temperature is None else float(temperature)
    max_tokens = int(cfg.get("max_tokens", 1024) if max_tokens is None else max_tokens)
    try:
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        # llama_cpp returns dict with choices
        if isinstance(out, dict) and "choices" in out and out["choices"]:
            text = out["choices"][0].get("text", "")
            return text
        return str(out)
    except Exception as e:
        return f"[LLM error] {e}"

# -------------------- Helpers: build retrieval context --------------------

def gather_contexts(query_text, top_k_each=3):
    """
    Query all indices to gather supporting context snippets for prompt.
    Returns text sections concatenated.
    """
    parts = []
    try:
        JIRA_INDEX._ensure_loaded()
        jhits = JIRA_INDEX.search(query_text, top_k=top_k_each)
        if jhits:
            parts.append("=== Similar JIRA issues ===")
            for h in jhits:
                parts.append(f"{h.get('id')} | score={h.get('score'):.3f} | {h.get('summary')}")
                desc = h.get("description") or h.get("text") or ""
                if desc:
                    parts.append(desc[:1200])
    except Exception:
        pass

    try:
        TC_INDEX._ensure_loaded()
        thits = TC_INDEX.search(query_text, top_k=top_k_each)
        if thits:
            parts.append("=== Similar UVM testcases ===")
            for h in thits:
                parts.append(f"{h.get('id')} | {h.get('name')} | UVM {h.get('uvm_version', '')} | score={h.get('score'):.3f}")
                if h.get("summary"):
                    parts.append(h.get("summary")[:1000])
    except Exception:
        pass

    try:
        PDF_INDEX._ensure_loaded()
        phits = PDF_INDEX.search(query_text, top_k=top_k_each)
        if phits:
            parts.append("=== PDF context (doc:page) ===")
            for h in phits:
                parts.append(f"[{h.get('doc_name')} p{h.get('page')}] score={h.get('score'):.3f}")
                parts.append(h.get("text","")[:1200])
    except Exception:
        pass

    try:
        LOG_INDEX._ensure_loaded()
        lhits = LOG_INDEX.search(query_text, top_k=top_k_each)
        if lhits:
            parts.append("=== Similar log snippets ===")
            for h in lhits:
                parts.append(h.get("snippet","")[:1200])
    except Exception:
        pass

    try:
        SOL_INDEX._ensure_loaded()
        shits = SOL_INDEX.search(query_text, top_k=top_k_each)
        if shits:
            parts.append("=== Past solutions (learning mode) ===")
            for h in shits:
                parts.append(f"{h.get('jira_id','')} | {h.get('summary','')} | score={h.get('score'):.3f}")
                parts.append(h.get('fix','')[:1200])
    except Exception:
        pass

    return "\n\n".join(parts)

# -------------------- HTTP handler --------------------

class XgentHandler(SimpleHTTPRequestHandler):
    def _send_json(self, code=200, obj=None):
        data = json.dumps(obj or {}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.path = "/ui/index.html"
        return super().do_GET()

    def do_POST(self):
        from cgi import FieldStorage
        import csv

        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length","0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload = {}

        # ---- Login ----
        if parsed.path == "/api/login":
            u = (payload.get("username") or "").strip()
            p = payload.get("password") or ""
            if check_login(u, p):
                return self._send_json(200, {"ok": True, "username": u})
            return self._send_json(200, {"ok": False, "error": "invalid_credentials"})

        if parsed.path == "/api/config":
            return self._send_json(200, {"ok": True, "llm": CONFIG.get("llm", {}), "embedding": CONFIG.get("embedding", {})})

        # ---- Jira flow: use local jiraprint tool if jira id is given ----
        if parsed.path == "/api/jira_flow":
            jira_id = (payload.get("jira_id") or "").strip()
            override_desc = (payload.get("description_override") or "").strip()
            if not jira_id and not override_desc:
                return self._send_json(200, {"ok": False, "error": "jira_id_or_description_required"})

            # if jira id provided and matches pattern like AVSREQ-12345, call jiraprint
            issue = {"key": "USER", "summary": "", "description": ""}
            if jira_id:
                # call external tool jiraprint --id <id>
                try:
                    cmd = f"jiraprint --id {shlex.quote(jira_id)}"
                    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, text=True)
                    if p.returncode == 0 and p.stdout.strip():
                        # try to parse JSON output, otherwise treat stdout as description
                        stdout = p.stdout.strip()
                        try:
                            jobj = json.loads(stdout)
                            # expect keys 'key','summary','description'
                            issue = {"key": jobj.get("key", jira_id), "summary": jobj.get("summary",""), "description": jobj.get("description","")}
                        except Exception:
                            # fallback: raw text
                            issue = {"key": jira_id, "summary": "", "description": stdout}
                    else:
                        issue = {"key": jira_id, "summary": "", "description": f"Unable to fetch Jira via jiraprint: {p.stderr.strip() or 'no output'}"}
                except Exception as e:
                    issue = {"key": jira_id, "summary": "", "description": f"Error calling jiraprint: {e}"}
            else:
                issue = {"key": "USER", "summary": "", "description": override_desc}

            desc = (issue.get("summary","") + "\n" + issue.get("description","")).strip()
            if override_desc:
                desc = override_desc + "\n\nOriginal:\n" + desc

            # decide retrieval + flow
            thr = float(CONFIG.get("jira_similarity_threshold", 0.3))
            similar, testcases, solutions = [], [], []

            # retrieve similar jira
            try:
                similar = JIRA_INDEX.search(desc, top_k=5)
            except Exception:
                similar = []

            best = similar[0]["score"] if similar else 0.0
            flow_action = "jira_match" if best >= thr else "testcase_match"

            if flow_action == "testcase_match":
                try:
                    testcases = TC_INDEX.search(desc, top_k=5)
                except Exception:
                    testcases = []
                if not testcases:
                    flow_action = "codegen"

            # learning solutions
            try:
                solutions = SOL_INDEX.search(desc, top_k=5)
            except Exception:
                solutions = []

            # build retrieval context from indices (PDFs, testcases, jira, logs, solutions)
            context = gather_contexts(desc, top_k_each=3)

            # build prompt for LLM: include context and ask for either reproduce testcase or code suggestion
            system = (
                "You are an expert Xcelium / UVM / SystemVerilog engineer. "
                "Given the user issue description and supporting contexts below, propose a concise, correct "
                "UVM testcase or reproduce the steps needed. If code is requested, output only code between "
                "<<<VERILOG_START>>> and <<<VERILOG_END>>> markers. If test steps, give numbered steps. "
                "Prioritize correctness and synthesizeability if applicable."
            )
            user_text = f"Issue {issue.get('key')}: {issue.get('summary')}\n\n{desc}\n\nFlow action: {flow_action}"

            prompt = system + "\n\n" + ("Context:\n" + context + "\n\n" if context else "") + "User:\n" + user_text

            gen = llm_generate(prompt)

            # perf recommendations from keywords and recipes
            perf_recos = []
            low = desc.lower()
            if "slow" in low or "performance" in low or "perf" in low:
                perf_recos = ["-newperf", "-plusperf"]

            resp = {
                "ok": True,
                "issue": issue,
                "description": desc,
                "similar_jira": similar,
                "testcases": testcases,
                "solutions": solutions,
                "context_text_snippet": context[:8000],
                "generated_testcase_or_code": gen,
                "flow_action": flow_action,
                "similarity_threshold": thr,
                "perf_recommendations": perf_recos,
            }
            return self._send_json(200, resp)

        # ---- Jira match (search only) ----
        if parsed.path == "/api/jira_match":
            text_q = (payload.get("text") or "").strip()
            if not text_q:
                return self._send_json(200, {"ok": False, "error": "empty_query"})
            try:
                matches = JIRA_INDEX.search(text_q, top_k=5)
                return self._send_json(200, {"ok": True, "results": matches})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        # ---- PDF upload/list/query ----
        if parsed.path == "/api/pdf_upload":
            env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type")}
            fs = FieldStorage(fp=io.BytesIO(raw), headers=self.headers, environ=env)
            fileitem = fs["file"] if "file" in fs else None
            if not fileitem or not fileitem.file:
                return self._send_json(200, {"ok": False, "error": "no_file"})
            filename = fileitem.filename or "upload.pdf"
            pdf_bytes = fileitem.file.read()
            try:
                texts, metas = extract_pdf_chunks(pdf_bytes, filename)
                if not texts:
                    return self._send_json(200, {"ok": False, "error": "no_text_extracted"})
                PDF_INDEX.add(texts, metas)
                return self._send_json(200, {"ok": True, "name": filename, "doc_id": metas[0]["doc_id"], "chunks": len(texts)})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        if parsed.path == "/api/pdf_list":
            try:
                docs = summarize_pdf_docs()
                return self._send_json(200, {"ok": True, "docs": docs})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        if parsed.path == "/api/pdf_query":
            q = (payload.get("q") or payload.get("text") or "").strip()
            if not q:
                return self._send_json(200, {"ok": False, "error": "empty_query"})
            try:
                hits = PDF_INDEX.search(q, top_k=int(payload.get("top_k", 6)))
                for h in hits:
                    if len(h.get("text","")) > 1200:
                        h["text"] = h["text"][:1200] + "..."
                return self._send_json(200, {"ok": True, "results": hits})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        # ---- Testcase upload/train & match ----
        if parsed.path == "/api/testcase_upload_train":
            env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type")}
            fs = FieldStorage(fp=io.BytesIO(raw), headers=self.headers, environ=env)
            fileitem = fs["file"] if "file" in fs else None
            if not fileitem or not fileitem.file:
                return self._send_json(200, {"ok": False, "error": "no_file"})
            csv_bytes = fileitem.file.read()
            text_csv = csv_bytes.decode("utf-8", errors="ignore")
            rows = []
            reader = csv.DictReader(io.StringIO(text_csv))
            for row in reader:
                rows.append({
                    "id": row.get("id",""),
                    "name": row.get("name",""),
                    "uvm_version": row.get("uvm_version",""),
                    "summary": row.get("summary",""),
                    "description": row.get("description",""),
                    "code": row.get("code",""),
                })
            if not rows:
                return self._send_json(200, {"ok": False, "error": "empty_csv"})
            texts = [build_tc_text(r) for r in rows]
            metas = rows
            try:
                TC_INDEX.add(texts, metas)
                return self._send_json(200, {"ok": True, "docs": len(rows)})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        if parsed.path == "/api/testcase_match":
            q = (payload.get("text") or "").strip()
            if not q:
                return self._send_json(200, {"ok": False, "error": "empty_query"})
            try:
                hits = TC_INDEX.search(q, top_k=int(payload.get("top_k", 5)))
                return self._send_json(200, {"ok": True, "results": hits})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        # ---- Log triage ----
        if parsed.path == "/api/log_upload_train":
            from cgi import FieldStorage as FS2
            env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type")}
            fs = FS2(fp=io.BytesIO(raw), headers=self.headers, environ=env)
            fileitem = fs["file"] if "file" in fs else None
            if not fileitem or not fileitem.file:
                return self._send_json(200, {"ok": False, "error": "no_file"})
            log_bytes = fileitem.file.read()
            text = log_bytes.decode("utf-8", errors="ignore")
            lines = text.splitlines()
            chunk = []
            texts, metas = [], []
            for ln in lines:
                if not ln.strip():
                    continue
                chunk.append(ln)
                if len(chunk) >= 40:
                    s = "\n".join(chunk)
                    texts.append(s)
                    metas.append({"snippet": s[:200]})
                    chunk = []
            if chunk:
                s = "\n".join(chunk)
                texts.append(s)
                metas.append({"snippet": s[:200]})
            if not texts:
                return self._send_json(200, {"ok": False, "error": "empty_log"})
            try:
                LOG_INDEX.add(texts, metas)
                return self._send_json(200, {"ok": True, "chunks": len(texts)})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        if parsed.path == "/api/log_triage":
            snippet = (payload.get("text") or "").strip()
            if not snippet:
                return self._send_json(200, {"ok": False, "error": "empty_snippet"})
            try:
                hits = LOG_INDEX.search(snippet, top_k=int(payload.get("top_k", 5)))
            except Exception as e:
                hits = []
            prompt = (
                "You are an expert Xcelium log triage assistant. "
                "Given the current log snippet and some similar historical snippets, "
                "identify likely root cause categories and recommended actions.\n\n"
                f"CURRENT LOG:\n{snippet}\n\n"
                f"SIMILAR SNIPPETS:\n{json.dumps(hits, indent=2)}\n"
            )
            analysis = llm_generate(prompt)
            return self._send_json(200, {"ok": True, "similar": hits, "analysis": analysis})

        # ---- CCF redundancy ----
        if parsed.path == "/api/ccf_redundancy":
            from cgi import FieldStorage as FS3
            env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type")}
            fs = FS3(fp=io.BytesIO(raw), headers=self.headers, environ=env)
            files_field = fs["file"] if "file" in fs else []
            files = files_field if isinstance(files_field, list) else [files_field]
            if not files:
                return self._send_json(200, {"ok": False, "error": "no_files"})
            all_sets = {}
            for idx, item in enumerate(files):
                if not item.file:
                    continue
                name = item.filename or f"ccf_{idx}.ccf"
                text = item.file.read().decode("utf-8", errors="ignore")
                all_sets[name] = parse_ccf_bins(text)
            if len(all_sets) < 2:
                return self._send_json(200, {"ok": False, "error": "need_at_least_two_ccf"})
            names = list(all_sets.keys())
            union = set().union(*all_sets.values())
            intersection = set(all_sets[names[0]])
            for s in all_sets.values():
                intersection &= s
            per_file_unique = {n: sorted(list(all_sets[n] - set.union(*(all_sets[m] for m in names if m != n))))[:50] for n in names}
            summary = {
                "files": names,
                "total_union_bins": len(union),
                "total_common_bins": len(intersection),
                "unique_bins_per_file": {n: len(per_file_unique[n]) for n in names}
            }
            prompt = (
                "You are a coverage redundancy expert. We have multiple CCF files with the "
                "following statistics:\n"
                f"{json.dumps(summary, indent=2)}\n\n"
                "Suggest which CCFs can be merged or reduced and give concrete guidelines."
            )
            explanation = llm_generate(prompt)
            return self._send_json(200, {
                "ok": True,
                "summary": summary,
                "example_unique_bins": per_file_unique,
                "explanation": explanation,
            })

        # ---- Memory profiler ----
        if parsed.path == "/api/memprofile_analyse":
            from cgi import FieldStorage as FS4
            env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type")}
            fs = FS4(fp=io.BytesIO(raw), headers=self.headers, environ=env)
            fileitem = fs["file"] if "file" in fs else None
            if not fileitem or not fileitem.file:
                return self._send_json(200, {"ok": False, "error": "no_file"})
            txt = fileitem.file.read().decode("utf-8", errors="ignore")
            prompt = (
                "You are a memory profiler assistant for Xcelium. "
                "Given the report below, identify main memory consumers, likely causes, "
                "and actions to reduce peak memory.\n\n"
                f"{txt[:8000]}"
            )
            analysis = llm_generate(prompt)
            return self._send_json(200, {"ok": True, "analysis": analysis})

        # ---- Guided debug script ----
        if parsed.path == "/api/debug_script":
            jira_id = (payload.get("jira_id") or "").strip()
            desc = (payload.get("description") or "").strip()
            env_info = payload.get("env", {})
            issue = {"key": "USER", "summary": "", "description": ""}
            if jira_id:
                try:
                    cmd = f"jiraprint --id {shlex.quote(jira_id)}"
                    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, text=True)
                    if p.returncode == 0 and p.stdout.strip():
                        try:
                            jobj = json.loads(p.stdout.strip())
                            issue = {"key": jobj.get("key", jira_id), "summary": jobj.get("summary",""), "description": jobj.get("description","")}
                        except Exception:
                            issue = {"key": jira_id, "summary": "", "description": p.stdout.strip()}
                    else:
                        issue = {"key": jira_id, "summary": "", "description": f"Unable to fetch Jira via jiraprint: {p.stderr.strip() or 'no output'}"}
                except Exception as e:
                    issue = {"key": jira_id, "summary": "", "description": f"Error calling jiraprint: {e}"}
            else:
                issue = {"key": "USER", "summary": "", "description": desc}

            text = desc or (issue.get("summary","") + "\n" + issue.get("description",""))
            try:
                recipes = load_debug_recipes()
            except Exception:
                recipes = {}
            categories = []
            if "slow" in text.lower() or "perf" in text.lower() or "performance" in text.lower():
                categories.append("perf")
            if "oom" in text.lower() or "out of memory" in text.lower() or "memory" in text.lower():
                categories.append("memory")
            if "coverage" in text.lower() or "ccf" in text.lower():
                categories.append("coverage")
            if "cdc" in text.lower():
                categories.append("cdc")
            chosen_flags, notes = [], []
            for cat in categories:
                r = recipes.get(cat) or {}
                chosen_flags.extend(r.get("flags", []))
                if r.get("notes"):
                    notes.append(f"{cat}: {r['notes']}")
            seen, final_flags = set(), []
            for f in chosen_flags:
                if f not in seen:
                    seen.add(f)
                    final_flags.append(f)
            base_cmd = env_info.get("base_cmd", "xrun")
            test = env_info.get("test", "<test_name>")
            extra = env_info.get("extra", "")
            cmd = f"{base_cmd} {' '.join(final_flags)} {extra} +UVM_TESTNAME={test}"
            prompt = (
                "You are an expert Xcelium debug engineer. Given the Jira and flags below, "
                "refine this command line if needed and explain why.\n\n"
                f"Jira {issue.get('key')}:\n{issue.get('summary')}\n{issue.get('description')}\n\n"
                f"Categories: {categories}\nFlags: {final_flags}\nCommand: {cmd}\nNotes: {notes}\n"
            )
            commentary = llm_generate(prompt)
            return self._send_json(200, {
                "ok": True,
                "categories": categories,
                "flags": final_flags,
                "cmd": cmd,
                "notes": notes,
                "llm_commentary": commentary,
            })

        # ---- Learning mode: add & match solutions ----
        if parsed.path == "/api/solution_add":
            jira_id = (payload.get("jira_id") or "").strip()
            summary = (payload.get("summary") or "").strip()
            root_cause = (payload.get("root_cause") or "").strip()
            fix = (payload.get("fix") or "").strip()
            test_used = (payload.get("testcase") or "").strip()
            cov_delta = (payload.get("coverage_delta") or "").strip()
            if not (jira_id and fix):
                return self._send_json(200, {"ok": False, "error": "jira_id_and_fix_required"})
            text = " ".join([jira_id, summary, root_cause, fix, test_used, cov_delta])
            meta = {
                "jira_id": jira_id,
                "summary": summary,
                "root_cause": root_cause,
                "fix": fix,
                "testcase": test_used,
                "coverage_delta": cov_delta,
                "ts": time.time(),
            }
            try:
                SOL_INDEX.add([text], [meta])
                return self._send_json(200, {"ok": True})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        if parsed.path == "/api/solution_match":
            desc = (payload.get("text") or "").strip()
            if not desc:
                return self._send_json(200, {"ok": False, "error": "empty_query"})
            try:
                hits = SOL_INDEX.search(desc, top_k=int(payload.get("top_k", 5)))
                return self._send_json(200, {"ok": True, "results": hits})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        # ---- Generic LLM chat ----
        if parsed.path == "/api/llm_chat":
            text_q = (payload.get("text") or "").strip()
            if not text_q:
                return self._send_json(200, {"ok": False, "error": "empty_prompt"})
            try:
                # build context for better answers
                ctx = gather_contexts(text_q, top_k_each=2)
                prompt = "You are an expert Xcelium/UVM assistant.\n\n" + ("Context:\n" + ctx + "\n\n" if ctx else "") + text_q
                out = llm_generate(prompt)
                return self._send_json(200, {"ok": True, "text": out})
            except Exception as e:
                return self._send_json(200, {"ok": False, "error": str(e)})

        return self._send_json(404, {"ok": False, "error": "unknown_endpoint"})

def run():
    init_db()
    host = CONFIG["server"]["host"]
    port = CONFIG["server"]["port"]
    httpd = HTTPServer((host, port), XgentHandler)
    print(f"Xgent Agentic AI running at http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
