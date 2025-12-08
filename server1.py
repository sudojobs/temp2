import json, os, threading, sqlite3, time, io, traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DB_PATH = ROOT / "xgent_ai.sqlite3"

# -------------------- SQLite (login-ready, currently unused in UI) --------------------

def init_db():
    need_seed = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )""")
    if need_seed:
        try:
            cur.execute("INSERT INTO users(username,password) VALUES(?,?)",("xgent","xgent123"))
        except Exception:
            pass
    conn.commit(); conn.close()

def check_login(username,password):
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=? AND password=?",(username,password))
    row = cur.fetchone(); conn.close()
    return row is not None

# -------------------- BGE embeddings + FAISS index --------------------

_EMBEDDER = None
_EMBED_LOCK = threading.Lock()

def _load_bge_model():
    global _EMBEDDER
    if _EMBEDDER is not None: return _EMBEDDER
    with _EMBED_LOCK:
        if _EMBEDDER is not None: return _EMBEDDER
        from sentence_transformers import SentenceTransformer
        emb_cfg = CONFIG.get("embedding",{})
        model_name = emb_cfg.get("local_path") or emb_cfg.get("model") or "BAAI/bge-base-en-v1.5"
        print("[BGE] loading", model_name)
        _EMBEDDER = SentenceTransformer(model_name)
        return _EMBEDDER

def embed_texts(texts):
    if not texts: return []
    m = _load_bge_model()
    embs = m.encode(
        texts,
        batch_size=CONFIG.get("embedding",{}).get("batch_size",32),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [list(map(float,v)) for v in embs]

def _np():
    import numpy as _np; return _np

def _faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None

class EmbeddingIndex:
    """
    Simple disk-backed index:
      - embeddings stored in prefix.npz
      - metadata stored in prefix.json
      - optional FAISS IP index for fast similarity search
    """
    def __init__(self, prefix: Path, dim: int = 768):
        self.prefix = prefix; self.dim = dim
        self.emb = None
        self.meta = []
        self._loaded = False
        self._faiss_index = None

    @property
    def npz_path(self): return self.prefix.with_suffix(".npz")
    @property
    def meta_path(self): return self.prefix.with_suffix(".json")

    def load(self):
        if self._loaded: return
        np = _np()
        if self.npz_path.exists():
            data = np.load(self.npz_path)
            self.emb = data["emb"]
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text("utf-8"))
        self._build_faiss()
        self._loaded = True

    def _build_faiss(self):
        faiss = _faiss()
        if faiss is None or self.emb is None or len(self.emb)==0:
            self._faiss_index = None
            return
        d = self.emb.shape[1]
        idx = faiss.IndexFlatIP(d)
        idx.add(self.emb)
        self._faiss_index = idx
        print(f"[FAISS] {self.prefix.name}: n={self.emb.shape[0]} dim={d}")

    def save(self):
        np = _np()
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        arr = self.emb if self.emb is not None else np.zeros((0,self.dim),"float32")
        np.savez(self.npz_path, emb=arr)
        self.meta_path.write_text(json.dumps(self.meta,indent=2), encoding="utf-8")
        self._build_faiss()

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def add(self,texts,meta_list):
        assert len(texts)==len(meta_list)
        self._ensure_loaded()
        vecs = embed_texts(texts)
        np = _np()
        arr = np.array(vecs,"float32")
        if self.emb is None or len(self.emb)==0:
            self.emb = arr
        else:
            self.emb = np.vstack([self.emb,arr])
        self.meta.extend(meta_list)
        self.save()

    def search(self,q_text,top_k=5):
        self._ensure_loaded()
        if self.emb is None or len(self.meta)==0: return []
        np = _np()
        qv = embed_texts([q_text])[0]
        q = np.array(qv,"float32").reshape(1,-1)
        if self._faiss_index is not None:
            D,I = self._faiss_index.search(q,top_k)
            scores, idxs = D[0], I[0]
        else:
            scores = self.emb @ q[0]
            idxs = scores.argsort()[::-1][:top_k]
        res = []
        for i,s in zip(idxs,scores):
            i = int(i)
            if i<0 or i>=len(self.meta): continue
            m = dict(self.meta[i])
            m["score"] = float(s)
            res.append(m)
        return res

# indices for Jira, PDF, Testcases, Solutions
dim = CONFIG.get("embedding",{}).get("dim",1024)
JIRA_INDEX = EmbeddingIndex(ROOT/"indices"/"jira_bge",dim)
PDF_INDEX  = EmbeddingIndex(ROOT/"indices"/"pdf_bge",dim)
TC_INDEX   = EmbeddingIndex(ROOT/"indices"/"tc_bge",dim)
SOL_INDEX  = EmbeddingIndex(ROOT/"indices"/"solution_bge",dim)

# -------------------- PDF → text chunks --------------------

def extract_pdf_chunks(pdf_bytes, doc_name, max_chars=900):
    from PyPDF2 import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    doc_id = f"PDF-{int(time.time())}"
    buf = ""
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pg = i+1
        for line in txt.splitlines():
            if len(buf)+len(line)+1 > max_chars:
                if buf.strip():
                    chunks.append((buf.strip(),pg))
                buf = ""
            buf += line+"\n"
        if buf.strip():
            chunks.append((buf.strip(),pg))
            buf=""
    texts, metas = [], []
    for idx,(txt,pg) in enumerate(chunks):
        texts.append(txt)
        metas.append({
            "doc_id":doc_id,
            "doc_name":doc_name,
            "page":pg,
            "chunk_index":idx,
            "text":txt
        })
    return texts, metas

def summarize_pdf_docs():
    PDF_INDEX._ensure_loaded()
    docs = {}
    for m in PDF_INDEX.meta:
        key = (m.get("doc_id"),m.get("doc_name"))
        docs.setdefault(key,0); docs[key]+=1
    return [{"id":d,"name":n,"chunks":c} for (d,n),c in docs.items()]

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

def parse_ccf_bins(text):
    keys=set()
    for line in text.splitlines():
        s=line.strip()
        if not s or s.startswith("//"): 
            continue
        low=s.lower()
        if "bin" in low or "coverpoint" in low or "covergroup" in low:
            keys.add(" ".join(s.split()))
    return keys

# -------------------- LLM: llama_cpp + Meta-Llama-3.1-8B GGUF --------------------

_LLM=None
_LLM_LOCK=threading.Lock()

def _load_llm():
    global _LLM
    if _LLM is not None:
        return _LLM
    with _LLM_LOCK:
        if _LLM is not None:
            return _LLM
        from llama_cpp import Llama
        cfg = CONFIG.get("llm",{})
        model_path = str((ROOT/cfg.get("model_path","models/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf")).resolve())
        n_ctx = int(cfg.get("n_ctx",4096))
        n_threads = int(cfg.get("threads",max(1,os.cpu_count()//2)))
        gpu_layers = int(cfg.get("gpu_layers",35))
        print("[LLM] loading", model_path)
        _LLM = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            gpu_layers=gpu_layers,
            use_mmap=True,
            use_mlock=False,
        )
        return _LLM

def llm_generate(prompt, max_tokens=None, temperature=None):
    cfg = CONFIG.get("llm",{})
    if cfg.get("adapter")!="llama_cpp":
        return "[llama_cpp adapter not active]"
    try:
        llm = _load_llm()
    except Exception as e:
        return f"[LLM load error] {e}"
    temperature = float(cfg.get("temperature",0.1) if temperature is None else temperature)
    max_tokens = int(cfg.get("max_tokens",1024) if max_tokens is None else max_tokens)
    try:
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        if isinstance(out,dict) and out.get("choices"):
            return out["choices"][0].get("text","")
        return str(out)
    except Exception as e:
        return f"[LLM runtime error] {e}"

# -------------------- RAG context gathering --------------------

def gather_contexts(q, top_k_each=2):
    parts=[]
    for idx,label in [
        (JIRA_INDEX,"Jira"),
        (TC_INDEX,"Testcases"),
        (PDF_INDEX,"PDFs"),
        (SOL_INDEX,"Solutions"),
    ]:
        try:
            hits = idx.search(q, top_k_each)
        except Exception:
            hits = []
        if not hits:
            continue
        parts.append(f"=== {label} ===")
        for h in hits:
            s = h.get("score",0.0)
            base = " | ".join(f"{k}={v}" for k,v in h.items() if k not in ("score","text"))
            parts.append(f"{base} :: score={s:.3f}")
            if "text" in h:
                parts.append((h["text"] or "")[:400])
    return "\n".join(parts)

# -------------------- Simple planner / router --------------------

import re

def detect_jira_id(text):
    m=re.search(r"\b[A-Z][A-Z0-9]+-\d+\b", text or "")
    return m.group(0) if m else None

def agent_planner(text):
    """
    Decide what to do with the user input:
      - jira         (future Jira flow)
      - ccf_generate (CCF generator)
      - ccf_opt      (CCF optimisation)
      - pdf          (PDF QA)
      - sv_rewrite   (SystemVerilog refactor)
      - code         (generic debug / codegen)
    """
    t=(text or "").lower()
    if detect_jira_id(text):
        return "jira"
    if "coverage" in t or "ccf" in t:
        if "generate" in t or "create" in t:
            return "ccf_generate"
        return "ccf_opt"
    if "pdf" in t or "spec" in t:
        return "pdf"
    if "class" in t and ("optimize" in t or "optimise" in t or "refactor" in t):
        return "sv_rewrite"
    return "code"

# -------------------- HTTP handler --------------------

class XgentHandler(SimpleHTTPRequestHandler):
    def _send_json(self,code=200,obj=None):
        data=json.dumps(obj or {}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path in ("/","/index.html"):
            self.path="/ui/index.html"
        return super().do_GET()

    def do_POST(self):
        from cgi import FieldStorage
        import csv
        parsed=urlparse(self.path)
        length=int(self.headers.get("Content-Length","0") or "0")
        raw=self.rfile.read(length) if length>0 else b"{}"
        try:
            payload=json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload={}

        try:
            # optional login API (not wired in UI currently)
            if parsed.path=="/api/login":
                u=(payload.get("username") or "").strip()
                p=payload.get("password") or ""
                ok=check_login(u,p)
                return self._send_json(200,{"ok":ok,"username":u if ok else None})

            # planner
            if parsed.path=="/api/agent_route":
                txt=(payload.get("text") or "").strip()
                route=agent_planner(txt)
                return self._send_json(200,{"ok":True,"worker":route})

            # ---------- PDF upload/list/query ----------
            if parsed.path=="/api/pdf_upload":
                env={"REQUEST_METHOD":"POST","CONTENT_TYPE":self.headers.get("Content-Type")}
                fs=FieldStorage(fp=io.BytesIO(raw),headers=self.headers,environ=env)
                fileitem=fs["file"] if "file" in fs else None
                if not fileitem or not fileitem.file:
                    return self._send_json(200,{"ok":False,"error":"no_file"})
                name=fileitem.filename or "upload.pdf"
                b=fileitem.file.read()
                texts,metas=extract_pdf_chunks(b,name)
                if not texts:
                    return self._send_json(200,{"ok":False,"error":"no_text"})
                PDF_INDEX.add(texts,metas)
                return self._send_json(200,{"ok":True,"name":name,"chunks":len(texts)})

            if parsed.path=="/api/pdf_list":
                docs=summarize_pdf_docs()
                return self._send_json(200,{"ok":True,"docs":docs})

            if parsed.path=="/api/pdf_query":
                q=(payload.get("q") or payload.get("text") or "").strip()
                if not q:
                    return self._send_json(200,{"ok":False,"error":"empty"})
                hits=PDF_INDEX.search(q,top_k=int(payload.get("top_k",6)))
                for h in hits:
                    if len(h.get("text",""))>1200:
                        h["text"]=h["text"][:1200]+"..."
                return self._send_json(200,{"ok":True,"results":hits})

            # ---------- Testcase upload / match ----------
            if parsed.path=="/api/testcase_upload_train":
                env={"REQUEST_METHOD":"POST","CONTENT_TYPE":self.headers.get("Content-Type")}
                fs=FieldStorage(fp=io.BytesIO(raw),headers=self.headers,environ=env)
                fileitem=fs["file"] if "file" in fs else None
                if not fileitem or not fileitem.file:
                    return self._send_json(200,{"ok":False,"error":"no_file"})
                csv_bytes=fileitem.file.read()
                text_csv=csv_bytes.decode("utf-8",errors="ignore")
                rows=[]
                reader=csv.DictReader(io.StringIO(text_csv))
                for r in reader:
                    rows.append({
                        "id":r.get("id",""),
                        "name":r.get("name",""),
                        "uvm_version":r.get("uvm_version",""),
                        "summary":r.get("summary",""),
                        "description":r.get("description",""),
                        "code":r.get("code",""),
                    })
                if not rows:
                    return self._send_json(200,{"ok":False,"error":"empty_csv"})
                texts=[build_tc_text(r) for r in rows]
                TC_INDEX.add(texts,rows)
                return self._send_json(200,{"ok":True,"docs":len(rows)})

            if parsed.path=="/api/testcase_match":
                q=(payload.get("text") or "").strip()
                if not q:
                    return self._send_json(200,{"ok":False,"error":"empty"})
                hits=TC_INDEX.search(q,top_k=int(payload.get("top_k",5)))
                return self._send_json(200,{"ok":True,"results":hits})

            # ---------- CCF redundancy ----------
            if parsed.path=="/api/ccf_redundancy":
                from cgi import FieldStorage as FS2
                env={"REQUEST_METHOD":"POST","CONTENT_TYPE":self.headers.get("Content-Type")}
                fs=FS2(fp=io.BytesIO(raw),headers=self.headers,environ=env)
                files_field=fs["file"] if "file" in fs else []
                files=files_field if isinstance(files_field,list) else [files_field]
                if not files:
                    return self._send_json(200,{"ok":False,"error":"no_files"})
                all_sets={}
                for idx,item in enumerate(files):
                    if not item.file:
                        continue
                    name=item.filename or f"ccf_{idx}.ccf"
                    txt=item.file.read().decode("utf-8",errors="ignore")
                    all_sets[name]=parse_ccf_bins(txt)
                if len(all_sets)<2:
                    return self._send_json(200,{"ok":False,"error":"need_two"})
                names=list(all_sets.keys())
                union=set().union(*all_sets.values())
                inter=set(all_sets[names[0]])
                for s in all_sets.values():
                    inter &= s
                unique={n:sorted(list(all_sets[n]-set.union(*(all_sets[m] for m in names if m!=n))))[:50] for n in names}
                summary={
                    "files":names,
                    "total_union_bins":len(union),
                    "total_common_bins":len(inter),
                    "unique_bins_per_file":{n:len(unique[n]) for n in names}
                }
                return self._send_json(200,{"ok":True,"summary":summary,"examples":unique})

            # ---------- CCF generator (RAG + LLM) ----------
            if parsed.path=="/api/ccf_generate":
                desc=(payload.get("description") or "").strip()
                jira_id=(payload.get("jira_id") or "").strip()
                extra=(payload.get("extra") or "").strip()
                base_q=(jira_id+" "+desc+" "+extra).strip()
                ctx=gather_contexts(base_q,top_k_each=2)
                prompt="You are an expert coverage engineer. Generate a compact high-value CCF.\n"
                if ctx:
                    prompt+="Context:\n"+ctx+"\n\n"
                prompt+="User request:\n"+base_q+"\n\nCCF code:\n"
                out=llm_generate(prompt)
                return self._send_json(200,{"ok":True,"ccf_code":out})

            # ---------- SystemVerilog rewrite agent ----------
            if parsed.path=="/api/sv_rewrite":
                code=(payload.get("code") or "").strip()
                goal=(payload.get("goal") or "").strip()
                if not code:
                    return self._send_json(200,{"ok":False,"error":"empty_code"})
                ctx=gather_contexts(goal+"\n"+code if goal else code, top_k_each=2)
                prompt="You are a SystemVerilog/UVM refactor assistant.\n"
                if ctx:
                    prompt+="Context:\n"+ctx+"\n\n"
                prompt+="Rewrite and optimise the following code:\n"+code+"\n"
                out=llm_generate(prompt)
                return self._send_json(200,{"ok":True,"optimised":out})

            # ---------- Generic LLM chat (with RAG context) ----------
            if parsed.path=="/api/llm_chat":
                text_q=(payload.get("text") or "").strip()
                if not text_q:
                    return self._send_json(200,{"ok":False,"error":"empty"})
                ctx=gather_contexts(text_q,top_k_each=2)
                prompt="You are Xgent Agentic AI for Xcelium/UVM/debug.\n"
                if ctx:
                    prompt+="Context:\n"+ctx+"\n\n"
                prompt+="User:\n"+text_q
                out=llm_generate(prompt)
                return self._send_json(200,{"ok":True,"text":out})

            return self._send_json(404,{"ok":False,"error":"unknown_endpoint"})
        except Exception as e:
            tb=traceback.format_exc()
            return self._send_json(500,{"ok":False,"error":str(e),"traceback":tb})

def run():
    init_db()
    host=CONFIG["server"]["host"]; port=CONFIG["server"]["port"]
    httpd=HTTPServer((host,port),XgentHandler)
    print(f"Xgent Agentic AI running at http://{host}:{port}")
    httpd.serve_forever()

if __name__=="__main__":
    run()
