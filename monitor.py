import sqlite3
conn = sqlite3.connect("performance_journal.db")
def c(t):
    try: return conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    except: return "-"

bal = 0
try:
    r = conn.execute("SELECT current_balance FROM reading_balance WHERE id=1").fetchone()
    if r: bal = r[0]
except: pass

coding_feel = reading_feel = "n/a"
try:
    for r in conn.execute("SELECT activity_type, avg_valence FROM activity_preferences"):
        if r[0]=="coding": coding_feel = f"{r[1]:+.2f}"
        if r[0]=="reading": reading_feel = f"{r[1]:+.2f}"
except: pass

drive = "?"
try:
    from aspiration_engine import AspirationEngine
    from performance_scorer import PerformanceScorer
    a = AspirationEngine(PerformanceScorer())
    drive = a.profile.drive_state
except: pass

print(f"Drive: {drive} | Markers: {c('somatic_markers')} | Impressions: {c('raw_impressions')}")
print(f"Weights: {c('behavioral_weights')} | Assessments: {c('self_assessments')}")
print(f"Credits: {bal:.1f} | Chapters read: {c('book_readings')}")
print(f"Coding: {coding_feel} | Reading: {reading_feel}")
print(f"Tasks: {c('task_scores')} | Journal: {c('meta_journal')}")
