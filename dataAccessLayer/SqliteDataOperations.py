import sqlite3
import os
from pathlib import Path
import importlib.util

dbPath = ""

def _load_db_creator():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "create_intelligence_db.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"DB create script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("create_intelligence_db", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _resolve_source_db(target_path: str):
    target_abs = os.path.abspath(target_path)
    data_dir = os.path.dirname(target_abs)
    target_name = os.path.basename(target_abs).lower()

    candidates = []
    if target_name == "intelligencedata.sqlite":
        candidates.extend(["IntelligenceDataTeiasEmpty.sqlite", "IntelligenceData - Copy.sqlite"])
    else:
        candidates.extend(["IntelligenceData.sqlite", "IntelligenceDataTeiasEmpty.sqlite", "IntelligenceData - Copy.sqlite"])

    for candidate in candidates:
        full_path = os.path.join(data_dir, candidate)
        if os.path.isfile(full_path) and os.path.abspath(full_path) != target_abs:
            return full_path
    return None

def _ensure_db_exists():
    if dbPath is None or str(dbPath).strip() == "":
        raise ValueError("dbPath is empty.")

    target_path = os.path.abspath(dbPath)
    if os.path.isfile(target_path):
        return

    target_dir = os.path.dirname(target_path)
    if target_dir and not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    source_path = _resolve_source_db(target_path)
    if source_path is None:
        raise FileNotFoundError(
            f"Target DB not found and no reference DB found in folder: {target_dir}"
        )

    db_creator = _load_db_creator()
    db_creator.create_database_from_reference(
        source_db=source_path,
        target_db=target_path,
        include_data=False
    )

def openConneciton():
    _ensure_db_exists()
    sqliteDb = sqlite3.connect(dbPath)
    return sqliteDb

def execCommand(query, params, returnIndetity = False):
    id = 0
    db = openConneciton()
    cursor = db.cursor()
    if params is None:
        cursor.execute(query)
    else:
        cursor.execute(query, params)
    if returnIndetity:
        cursor.execute("SELECT last_insert_rowid() AS id")
        rows = cursor.fetchall()
        id = rows[0][0]
    db.commit()
    db.close()
    return id

def getData(query, params):
    db = openConneciton()
    cursor = db.cursor()
    if(params is not None and len(params) > 0):
        cursor.execute(query,params)
    else:
        cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    return  rows
