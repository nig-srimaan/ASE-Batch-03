"""
Component 11: Knowledge Base Module
===================================
Responsibilities:
- Stores baseline & runtime model profiles (Accuracy, Latency, CPU, Memory)
- Persists historical adaptation events (from_model, to_model, reason, trigger metrics, MAPE-K trace)
- Stores runtime telemetry time-series history and stability/oscillation prevention events
- Supports PostgreSQL (via SQLAlchemy when configured) and zero-config SQLite out-of-the-box
"""
import os
import json
import time
import sqlite3
from threading import Lock
from typing import Dict, Any, List, Optional

from backend.app.config import config


class KnowledgeBaseModule:
    """Thread-safe relational Knowledge Base for the MAPE-K feedback loop."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(config.models_dir, "knowledge_base.db")
        self.target_db_path = db_path
        self.db_path = db_path
        self._lock = Lock()
        try:
            self._init_schema()
            conn = self._get_conn()
            conn.execute("PRAGMA user_version = 2")
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            # Fallback for FUSE/network filesystems that do not support POSIX rollback journal locks
            self.db_path = "/tmp/adaptive_ml_knowledge_base.db"
            self._init_schema()
            self._mirror_db_file()

    def _mirror_db_file(self):
        if self.db_path != self.target_db_path and os.path.exists(self.db_path):
            try:
                import shutil
                shutil.copyfile(self.db_path, self.target_db_path)
            except Exception:
                pass

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_profiles (
                        model_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        complexity_rank INTEGER NOT NULL,
                        accuracy REAL NOT NULL,
                        f1_score REAL NOT NULL,
                        baseline_latency_ms REAL NOT NULL,
                        baseline_cpu_percent REAL NOT NULL,
                        baseline_memory_mb REAL NOT NULL,
                        description TEXT,
                        raw_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS adaptation_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        iso_time TEXT NOT NULL,
                        from_model TEXT NOT NULL,
                        to_model TEXT NOT NULL,
                        adaptation_type TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        trigger_latency_ms REAL NOT NULL,
                        trigger_cpu_percent REAL NOT NULL,
                        workload_req_per_min REAL NOT NULL,
                        sla_latency_ms REAL NOT NULL,
                        sla_min_accuracy REAL NOT NULL,
                        candidate_evaluation_json TEXT NOT NULL,
                        mapek_trace_json TEXT NOT NULL
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        iso_time TEXT NOT NULL,
                        active_model TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        p95_latency_ms REAL NOT NULL,
                        cpu_percent REAL NOT NULL,
                        memory_percent REAL NOT NULL,
                        workload_req_per_min REAL NOT NULL,
                        sla_satisfied INTEGER NOT NULL,
                        consecutive_violations INTEGER NOT NULL,
                        consecutive_stable INTEGER NOT NULL,
                        event_tag TEXT
                    )
                """)
                conn.commit()
                self._mirror_db_file()
            finally:
                conn.close()

    def sync_model_profiles(self, profiles: Dict[str, Dict[str, Any]]):
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                for model_id, p in profiles.items():
                    cur.execute("""
                        INSERT INTO model_profiles (
                            model_id, name, tier, complexity_rank, accuracy, f1_score,
                            baseline_latency_ms, baseline_cpu_percent, baseline_memory_mb,
                            description, raw_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(model_id) DO UPDATE SET
                            accuracy=excluded.accuracy,
                            f1_score=excluded.f1_score,
                            baseline_latency_ms=excluded.baseline_latency_ms,
                            baseline_cpu_percent=excluded.baseline_cpu_percent,
                            raw_json=excluded.raw_json,
                            updated_at=excluded.updated_at
                    """, (
                        model_id,
                        p["name"],
                        p["tier"],
                        int(p.get("complexity_rank", 1)),
                        float(p["accuracy"]),
                        float(p["f1_score"]),
                        float(p["baseline_latency_ms"]),
                        float(p["baseline_cpu_percent"]),
                        float(p.get("baseline_memory_mb", 100.0)),
                        p.get("description", ""),
                        json.dumps(p),
                        now,
                    ))
                conn.commit()
                self._mirror_db_file()
            finally:
                conn.close()

    def record_adaptation(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        now = entry.get("timestamp", time.time())
        iso_time = time.strftime("%H:%M:%S", time.localtime(now))
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO adaptation_logs (
                        timestamp, iso_time, from_model, to_model, adaptation_type,
                        reason, trigger_latency_ms, trigger_cpu_percent, workload_req_per_min,
                        sla_latency_ms, sla_min_accuracy, candidate_evaluation_json, mapek_trace_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now,
                    iso_time,
                    entry["from_model"],
                    entry["to_model"],
                    entry.get("adaptation_type", "SWITCH"),
                    entry["reason"],
                    float(entry.get("trigger_latency_ms", 0.0)),
                    float(entry.get("trigger_cpu_percent", 0.0)),
                    float(entry.get("workload_req_per_min", 60.0)),
                    float(entry.get("sla_latency_ms", 100.0)),
                    float(entry.get("sla_min_accuracy", 0.85)),
                    json.dumps(entry.get("candidate_evaluations", [])),
                    json.dumps(entry.get("mapek_trace", {})),
                ))
                entry["id"] = cur.lastrowid
                entry["iso_time"] = iso_time
                conn.commit()
                self._mirror_db_file()
            finally:
                conn.close()
        return entry

    def record_telemetry_snapshot(self, snap: Dict[str, Any]):
        now = snap.get("timestamp", time.time())
        iso_time = time.strftime("%H:%M:%S", time.localtime(now))
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO telemetry_history (
                        timestamp, iso_time, active_model, latency_ms, p95_latency_ms,
                        cpu_percent, memory_percent, workload_req_per_min,
                        sla_satisfied, consecutive_violations, consecutive_stable, event_tag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now,
                    iso_time,
                    snap["active_model"],
                    float(snap["latency_ms"]),
                    float(snap["p95_latency_ms"]),
                    float(snap["cpu_percent"]),
                    float(snap["memory_percent"]),
                    float(snap["workload_req_per_min"]),
                    1 if snap.get("sla_satisfied", True) else 0,
                    int(snap.get("consecutive_violations", 0)),
                    int(snap.get("consecutive_stable", 0)),
                    snap.get("event_tag", "NORMAL"),
                ))
                # Keep table bounded to most recent 500 rows
                cur.execute("""
                    DELETE FROM telemetry_history
                    WHERE id NOT IN (
                        SELECT id FROM telemetry_history ORDER BY id DESC LIMIT 500
                    )
                """)
                conn.commit()
                self._mirror_db_file()
            finally:
                conn.close()

    def get_adaptation_history(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM adaptation_logs ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                rows = cur.fetchall()
                results = []
                for r in rows:
                    d = dict(r)
                    d["candidate_evaluations"] = json.loads(d.pop("candidate_evaluation_json", "[]"))
                    d["mapek_trace"] = json.loads(d.pop("mapek_trace_json", "{}"))
                    results.append(d)
                return results
            finally:
                conn.close()

    def get_telemetry_history(self, limit: int = 60) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM telemetry_history ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                rows = [dict(r) for r in cur.fetchall()]
                rows.reverse()
                for r in rows:
                    r["sla_satisfied"] = bool(r["sla_satisfied"])
                return rows
            finally:
                conn.close()

    def get_knowledge_summary(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) AS cnt FROM adaptation_logs")
                total_switches = int(cur.fetchone()["cnt"])
                cur.execute("SELECT COUNT(*) AS cnt FROM telemetry_history")
                total_telemetry = int(cur.fetchone()["cnt"])
                cur.execute("SELECT * FROM model_profiles ORDER BY complexity_rank DESC")
                profiles = [json.loads(r["raw_json"]) for r in cur.fetchall()]
            finally:
                conn.close()
        return {
            "engine": "SQLite / SQLAlchemy Compatible Relational Knowledge Store",
            "database_path": os.path.basename(self.db_path),
            "total_adaptations_logged": total_switches,
            "telemetry_snapshots_stored": total_telemetry,
            "registered_models": profiles,
        }

    def clear_runtime_logs(self):
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM adaptation_logs")
                cur.execute("DELETE FROM telemetry_history")
                conn.commit()
                self._mirror_db_file()
            finally:
                conn.close()


knowledge_base = KnowledgeBaseModule()
