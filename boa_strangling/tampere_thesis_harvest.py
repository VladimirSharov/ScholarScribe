#!/usr/bin/env python3
"""
Harvest THESIS METADATA (not full texts) from Tampere University.

Two legitimate, public sources are used, in order:

  1. Trepo OAI-PMH  ->  https://trepo.tuni.fi/oai/request
     Trepo is Tampere University's official institutional repository (DSpace).
     OAI-PMH is the standard protocol whose *entire purpose* is polite bulk
     metadata harvesting. This is the CANONICAL source and the only one that
     reliably carries the ABSTRACT and the abstract's language.

  2. Finna API     ->  https://api.finna.fi/v1/search   (JSON)
     Finna is the National Library of Finland's public discovery API
     (org: NatLibFi). JSON, CORS-open, documented. Used as an automatic
     fallback if the OAI endpoint is unreachable/blocked. Coverage of the
     abstract and abstract-language is weaker here, but title/tags/year/
     faculty/work-language/record-URL are solid, and it never hits a bot wall.

WHY A FALLBACK EXISTS
  Trepo currently sits behind "Anubis" (a proof-of-work bot wall). A plain
  HTTP client MAY be challenged and receive an HTML page instead of XML.
  This script DETECTS that and (a) tells you exactly what happened and
  (b) automatically switches to Finna so you always get a usable dataset.
  If OAI is blocked and you need the abstracts in bulk, the polite fix is to
  email the Trepo/Library team (their contact address is returned by the
  Identify verb, see verify_endpoint() below) and ask to harvest via OAI-PMH.
  That is a normal, expected request; OAI exists for exactly this.

OUTPUT
  One JSON object per line (JSONL) at OUT_PATH, plus a small run summary.
  Every record has the SAME schema regardless of which source produced it:

    source            "trepo-oai" | "finna"
    id                stable record id in that source
    record_url        page a human can open (use this if your downstream
                      meta-analysis throws on a record and you want to eyeball it)
    title             str
    abstract          str        (best/primary abstract; "" if none)
    abstracts         [{"text": str, "lang": "fi|en|sv|None"}, ...]  (all of them)
    tags              [str, ...] (keywords / subjects)
    year              str|None
    faculties         [str, ...] (faculty / collection names)
    type              str|None   (thesis level, e.g. Master's thesis)
    language_work     str|None   (ISO-ish, e.g. "fi", "en")
    language_abstract str|None   (language of the primary abstract)

DEPENDENCIES: none. Python 3.8+ standard library only. Runs offline-capable
machines with just `python3 tampere_thesis_harvest.py`.

POLITENESS (baked in): descriptive User-Agent with YOUR contact email, a
per-request delay, exponential-backoff retries, single-threaded, and full
respect for OAI resumptionToken paging. Set CONTACT_EMAIL before running.

ATTRIBUTION: if you publish anything derived from Finna metadata, state that
the metadata source is Finna (their reuse request).
"""

import json
import re
import signal
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG  --  edit CONTACT_EMAIL before first run.
# --------------------------------------------------------------------------
CONTACT_EMAIL = "vladimir.v.sharov@student.jyu.fi"
PROJECT_NAME  = "thesis-metadata-analysis"

OAI_BASE   = "https://jyx.jyu.fi/oai/request"
FINNA_BASE = "https://api.finna.fi/v1/search"

# All paths are relative to this script's location (boa_strangling/data/)
_DATA_DIR   = Path(__file__).parent / "data"
OUT_PATH    = _DATA_DIR / "j_theses.jsonl"
STATE_PATH  = _DATA_DIR / "j_harvest_state.json"

REQUEST_DELAY_S = 1.0     # pause between requests (be gentle)
MAX_RETRIES     = 4
TIMEOUT_S       = 60
MAX_RECORDS     = 5    # None = all. Set an int to cap for a test run, e.g. 50.

USER_AGENT = f"{PROJECT_NAME}/1.0 (metadata harvest; mailto:{CONTACT_EMAIL})"

# --------------------------------------------------------------------------
# Graceful Ctrl+C (mirrors the JYU collector pattern)
# --------------------------------------------------------------------------
_keep_running = True

def _sigint_handler(signum, frame):
    global _keep_running
    print("\n[interrupted] Ctrl+C received — finishing current page then saving state.")
    _keep_running = False

signal.signal(signal.SIGINT, _sigint_handler)

# --------------------------------------------------------------------------
# State helpers — save resumptionToken / Finna page so we can restart later
# --------------------------------------------------------------------------
def _load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_state(state: dict):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def _clear_state():
    if STATE_PATH.exists():
        STATE_PATH.unlink()

# dc:type strings that mark a thesis, Finnish + English + Swedish, lowercased.
THESIS_TYPE_KEYWORDS = (
    "thesis", "dissertation", "opinnäyte", "opinnayte", "tutkielma",
    "diplomityö", "diplomityo", "pro gradu", "gradu", "väitöskirja",
    "vaitoskirja", "kandidaat", "master", "bachelor", "licentiate",
    "lisensiaat", "avhandling",
)

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class AnubisBlocked(Exception):
    """OAI endpoint answered with the bot-wall challenge instead of data."""

class OAIError(Exception):
    pass

# --------------------------------------------------------------------------
# Tiny, dependency-free language guesser for abstracts (fi / en / sv).
# Not perfect; good enough to tag which abstract is which when the source
# does not tell us. Returns 'fi' | 'en' | 'sv' | None.
# --------------------------------------------------------------------------
_STOP = {
    "fi": {"ja", "on", "ei", "että", "tämä", "sekä", "joka", "ovat", "kuin",
           "sen", "myös", "voidaan", "tutkimuksen", "työn", "tässä"},
    "en": {"the", "and", "of", "to", "in", "is", "this", "that", "with",
           "for", "are", "was", "which", "study", "thesis"},
    "sv": {"och", "att", "det", "som", "för", "med", "denna", "är", "kan",
           "arbetet", "studien", "här"},
}

def guess_lang(text):
    if not text:
        return None
    words = re.findall(r"[a-zåäöéü]+", text.lower())
    if not words:
        return None
    ws = set(words)
    scores = {lang: len(ws & stops) for lang, stops in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

# --------------------------------------------------------------------------
# Polite HTTP GET with retries + Anubis detection.
# --------------------------------------------------------------------------
def polite_get(url, expect="xml"):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/xml" if expect == "xml" else "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                raw = r.read()
                ctype = r.headers.get("Content-Type", "")
                body = raw.decode("utf-8", "replace")
                # Anubis / bot-wall detection: we asked for data, got HTML.
                lower = body[:2000].lower()
                if "anubis" in lower or ("access denied" in lower and "<html" in lower):
                    raise AnubisBlocked(
                        "Trepo OAI answered with the Anubis bot-wall challenge, "
                        "not data. See the module docstring for the polite fix "
                        "(email the library to allow OAI-PMH harvesting). "
                        "Falling back to Finna for now."
                    )
                if expect == "xml" and "text/html" in ctype and "<html" in lower:
                    raise AnubisBlocked(
                        "Expected XML from OAI but received an HTML page "
                        "(likely a bot wall or maintenance page)."
                    )
                return body
        except AnubisBlocked:
            raise
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(REQUEST_DELAY_S * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(REQUEST_DELAY_S * (2 ** attempt))
    raise OAIError(f"GET failed after {MAX_RETRIES} tries: {url} ({last})")

# --------------------------------------------------------------------------
# OAI-PMH namespaces & helpers
# --------------------------------------------------------------------------
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc":  "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

def oai_url(**params):
    return OAI_BASE + "?" + urllib.parse.urlencode(params)

def oai_get_tree(**params):
    body = polite_get(oai_url(**params), expect="xml")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        raise OAIError(f"OAI response was not parseable XML: {e}")
    err = root.find("oai:error", NS)
    if err is not None:
        code = err.get("code", "")
        # noRecordsMatch / noSetHierarchy are non-fatal for our purposes
        if code in ("noRecordsMatch",):
            return root
        raise OAIError(f"OAI error [{code}]: {(err.text or '').strip()}")
    return root

def verify_endpoint():
    """Identify verb -> proves the endpoint is the real Trepo repository and
    prints its admin contact email. Run this once to (a) confirm legitimacy
    and (b) find who to email if you need bulk OAI access."""
    root = oai_get_tree(verb="Identify")
    ident = root.find("oai:Identify", NS)
    if ident is None:
        print("Identify returned no data.")
        return
    def t(tag):
        el = ident.find(f"oai:{tag}", NS)
        return el.text.strip() if el is not None and el.text else "?"
    print("Repository :", t("repositoryName"))
    print("Base URL   :", t("baseURL"))
    print("Admin email:", [e.text for e in ident.findall("oai:adminEmail", NS)])
    print("Granularity:", t("granularity"))

def load_set_map():
    """setSpec -> human name, harvested from ListSets. Faculty/collection names
    live here; each thesis record's <setSpec> entries point into this map."""
    smap = {}
    token = None
    while True:
        if token:
            root = oai_get_tree(verb="ListSets", resumptionToken=token)
        else:
            root = oai_get_tree(verb="ListSets")
        for s in root.findall(".//oai:set", NS):
            spec = s.findtext("oai:setSpec", default="", namespaces=NS).strip()
            name = s.findtext("oai:setName", default="", namespaces=NS).strip()
            if spec:
                smap[spec] = name
        rt = root.find(".//oai:resumptionToken", NS)
        token = rt.text.strip() if rt is not None and rt.text else None
        if not token:
            break
        time.sleep(REQUEST_DELAY_S)
    return smap

def _is_thesis(types):
    joined = " ".join(types).lower()
    return any(k in joined for k in THESIS_TYPE_KEYWORDS)

def parse_oai_record(rec_el, set_map):
    header = rec_el.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    ident = header.findtext("oai:identifier", default="", namespaces=NS).strip()
    setspecs = [s.text.strip() for s in header.findall("oai:setSpec", NS) if s.text]

    dc = rec_el.find(".//oai_dc:dc", NS)
    if dc is None:
        return None

    def all_dc(tag):
        out = []
        for el in dc.findall(f"dc:{tag}", NS):
            if el.text and el.text.strip():
                out.append((el.text.strip(), el.get(XML_LANG)))
        return out

    titles      = [v for v, _ in all_dc("title")]
    descriptions = all_dc("description")          # (text, lang) tuples
    subjects    = [v for v, _ in all_dc("subject")]
    dates       = [v for v, _ in all_dc("date")]
    languages   = [v for v, _ in all_dc("language")]
    types       = [v for v, _ in all_dc("type")]

    if not _is_thesis(types):
        return None  # skip non-theses (articles, datasets, books, ...)

    # Abstract heuristic: dc:description holds abstract(s) plus sometimes junk
    # (a URI, a page count). Keep descriptions that look like prose; the
    # longest is treated as the primary abstract.
    abstracts = []
    for text, lang in descriptions:
        if len(text) < 40:            # too short to be an abstract
            continue
        if re.match(r"^https?://", text):
            continue
        abstracts.append({"text": text, "lang": lang or guess_lang(text)})
    abstracts.sort(key=lambda a: len(a["text"]), reverse=True)
    primary = abstracts[0] if abstracts else {"text": "", "lang": None}

    # Faculties: map the record's setSpecs to their human names. DSpace encodes
    # community (faculty) & collection (programme) as sets. We keep names that
    # look faculty-ish and fall back to all set names if unsure.
    fac_names = [set_map.get(s, "") for s in setspecs]
    fac_names = [n for n in fac_names if n]
    faculties = [n for n in fac_names
                 if re.search(r"facult|tiedekun|yksikk|school|unit", n, re.I)]
    if not faculties:
        faculties = fac_names  # better to keep collection names than nothing

    year = None
    for d in sorted(dates):
        m = re.search(r"(19|20)\d{2}", d)
        if m:
            year = m.group(0)
            break

    # A human-openable page. OAI identifiers on Trepo look like
    # oai:trepo.tuni.fi:10024/XXXXX -> handle 10024/XXXXX.
    record_url = None
    m = re.search(r":(\d+/\d+)$", ident)
    if m:
        record_url = f"jyx.jyu.fi/handle/{m.group(1)}"

    return {
        "source": "trepo-oai",
        "id": ident,
        "record_url": record_url,
        "title": titles[0] if titles else "",
        "abstract": primary["text"],
        "abstracts": abstracts,
        "tags": subjects,
        "year": year,
        "faculties": faculties,
        "type": types[0] if types else None,
        "language_work": languages[0] if languages else None,
        "language_abstract": primary["lang"],
    }

def harvest_oai(resume_token=None, start_count=0):
    """Yield normalized thesis records from Trepo OAI-PMH.

    resume_token  — pass a saved resumptionToken to continue a previous run.
    start_count   — records already written (for MAX_RECORDS accounting).
    Raises AnubisBlocked if the bot wall intervenes.
    """
    if resume_token:
        print(f"[oai] resuming from saved token (already have {start_count} records)")
        # Set map is needed for faculty name resolution; re-fetch it quietly.
        set_map = load_set_map()
    else:
        print("[oai] verifying endpoint (Identify) ...")
        verify_endpoint()
        print("[oai] loading set map (ListSets) ...")
        set_map = load_set_map()
        print(f"[oai] {len(set_map)} sets loaded")

    n = start_count
    token = resume_token
    while _keep_running:
        if token:
            root = oai_get_tree(verb="ListRecords", resumptionToken=token)
        else:
            root = oai_get_tree(verb="ListRecords", metadataPrefix="oai_dc")

        for rec_el in root.findall(".//oai:record", NS):
            parsed = parse_oai_record(rec_el, set_map)
            if parsed:
                yield parsed
                n += 1
                if MAX_RECORDS and n >= MAX_RECORDS:
                    return

        rt = root.find(".//oai:resumptionToken", NS)
        token = rt.text.strip() if rt is not None and rt.text else None

        # Save state after every page so Ctrl+C or a crash can resume here.
        _save_state({"source": "trepo-oai", "oai_token": token, "count": n,
                     "complete": token is None})

        if not token:
            break
        time.sleep(REQUEST_DELAY_S)

# --------------------------------------------------------------------------
# Finna JSON fallback
# --------------------------------------------------------------------------
def finna_url(page, limit=100):
    # datasource "tuni" == Tampere University's records in Finna.
    # We request only the fields we need; format is post-filtered to theses.
    params = [
        ("lookfor", ""),
        ("type", "AllFields"),
        ("filter[]", "datasource_str_mv:helka"),
        ("limit", str(limit)),
        ("page", str(page)),
        ("field[]", "id"),
        ("field[]", "title"),
        ("field[]", "summary"),
        ("field[]", "subjects"),
        ("field[]", "year"),
        ("field[]", "languages"),
        ("field[]", "institutions"),
        ("field[]", "buildings"),
        ("field[]", "formats"),
        ("field[]", "recordPage"),
    ]
    return FINNA_BASE + "?" + urllib.parse.urlencode(params)

def parse_finna(rec):
    formats = [f.get("value", "") for f in rec.get("formats", [])]
    if not any("thesis" in f.lower() for f in formats):
        return None  # keep only theses

    subjects = []
    for grp in rec.get("subjects", []):
        if isinstance(grp, list):
            subjects.extend(x for x in grp if x)
        elif grp:
            subjects.append(grp)

    summary = rec.get("summary") or []
    if isinstance(summary, list):
        abstract = " ".join(s for s in summary if s).strip()
    else:
        abstract = str(summary).strip()

    faculties = [i.get("value", i) if isinstance(i, dict) else i
                 for i in rec.get("institutions", [])]
    faculties = [f for f in faculties if f]

    langs = rec.get("languages", []) or []
    thesis_type = None
    for f in formats:
        parts = [p for p in f.split("/") if p and not p.isdigit()]
        if len(parts) >= 2 and parts[0].lower() == "thesis":
            thesis_type = parts[-1]

    rp = rec.get("recordPage")
    record_url = ("https://www.finna.fi" + rp) if rp and rp.startswith("/") else rp

    return {
        "source": "finna",
        "id": rec.get("id"),
        "record_url": record_url,
        "title": rec.get("title", ""),
        "abstract": abstract,
        "abstracts": ([{"text": abstract, "lang": guess_lang(abstract)}]
                      if abstract else []),
        "tags": subjects,
        "year": str(rec.get("year")) if rec.get("year") else None,
        "faculties": faculties,
        "type": thesis_type,
        "language_work": langs[0] if langs else None,
        "language_abstract": guess_lang(abstract) if abstract else None,
    }

def harvest_finna(start_page=1, start_count=0):
    """Yield thesis records from the Finna JSON API.

    start_page   — resume from this page number.
    start_count  — records already written.
    """
    if start_page > 1:
        print(f"[finna] resuming from page {start_page} (already have {start_count} records)")
    else:
        print("[finna] harvesting via public JSON API (National Library of Finland)")

    page = start_page
    n = start_count
    while _keep_running:
        body = polite_get(finna_url(page), expect="json")
        data = json.loads(body)
        records = data.get("records", [])
        if not records:
            break
        for rec in records:
            parsed = parse_finna(rec)
            if parsed:
                yield parsed
                n += 1
                if MAX_RECORDS and n >= MAX_RECORDS:
                    return
        total = data.get("resultCount", 0)
        done = page * 100 >= total

        _save_state({"source": "finna", "finna_page": page + 1, "count": n,
                     "complete": done})

        if done:
            break
        page += 1
        time.sleep(REQUEST_DELAY_S)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if CONTACT_EMAIL == "you@example.org":
        print("ERROR: set CONTACT_EMAIL at the top of the file first.")
        sys.exit(1)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    state = _load_state()
    resuming = bool(state) and not state.get("complete", False)

    if resuming:
        src    = state["source"]
        count0 = state.get("count", 0)
        print(f"[resume] previous run found: source={src}, records so far={count0}")
        if src == "trepo-oai":
            saved_token = state.get("oai_token")
            used = "trepo-oai"
            source = harvest_oai(resume_token=saved_token, start_count=count0)
        else:
            saved_page = state.get("finna_page", 1)
            used = "finna"
            source = harvest_finna(start_page=saved_page, start_count=count0)
        file_mode = "a"   # append to existing output
    else:
        if state.get("complete"):
            print("[info] Previous harvest was complete. Starting fresh.")
        # Fresh start: try OAI, fall back to Finna.
        try:
            source = harvest_oai()
            source = _peek(source)   # triggers first network hit for early error detection
            used = "trepo-oai"
        except AnubisBlocked as e:
            print(f"[warn] {e}")
            source = harvest_finna()
            used = "finna"
        except OAIError as e:
            print(f"[warn] OAI unavailable ({e}); falling back to Finna.")
            source = harvest_finna()
            used = "finna"
        except StopIteration:
            print("[warn] OAI returned 0 thesis records; falling back to Finna.")
            source = harvest_finna()
            used = "finna"
        count0 = 0
        file_mode = "w"

    count = count0
    with OUT_PATH.open(file_mode, encoding="utf-8") as fh:
        for rec in source:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            if count % 200 == 0:
                print(f"  ... {count} records")

    if _keep_running:
        # Finished naturally — mark complete and clean up state.
        _save_state({**state, "count": count, "complete": True})
        print(f"\nDone. source={used}  records={count}  ->  {OUT_PATH.resolve()}")
    else:
        # Stopped by Ctrl+C — state was already saved inside the generator.
        print(f"\nStopped early. {count} records written to {OUT_PATH.resolve()}")
        print("Run again to resume from where you stopped.")

    if used == "finna":
        print("Note: Finna abstract/abstract-language coverage is partial. "
              "For complete abstracts, get OAI-PMH access (see docstring).")


def _peek(gen):
    """Pull one item to trigger network errors before opening the output file."""
    try:
        first = next(gen)
    except StopIteration:
        raise  # re-raise so main() can fall back to Finna
    def chained():
        yield first
        yield from gen
    return chained()


if __name__ == "__main__":
    main()
