import sqlite3
import json

DB_PATH = "Data/crosswords.sqlite3"
OUTPUT_FILE = "short_answers.json"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Query: alphabetic 3-5 letters, no spaces, no hyphens, alphabetically
query = """
SELECT answer, clue
FROM clues
WHERE length(answer) BETWEEN 3 AND 5
  AND answer GLOB '[A-Za-z]*'
  AND answer NOT LIKE '% %'
  AND answer NOT LIKE '%-%'
  AND answer != 'nan'
LIMIT 1000;
"""
# ORDER BY answer ASC;

cursor.execute(query)
results = cursor.fetchall()
conn.close()

# Clean clues: remove trailing " (#)" if present
cleaned_results = []
for answer, clue in results:
    if clue.endswith(")") and " (" in clue:
        clue = clue.rsplit(" (", 1)[0]
    cleaned_results.append((answer, clue))

# Save to JSON for fast re-loading
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_results, f, ensure_ascii=False, indent=2)

print(f"Exported {len(cleaned_results)} entries to {OUTPUT_FILE}")
