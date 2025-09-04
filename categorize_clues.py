import os
import json
import asyncio
import re
from google import genai
from aiofile import AIOFile, Writer

# Read API key from file (in API folder)
with open("API/GEMINI_API_KEY", "r") as f:
    api_key = f.read().strip()

# Set it as an environment variable (optional)
os.environ["GEMINI_API_KEY"] = api_key

# Initialize Gemini client
client = genai.Client()

INPUT_FILE = "crosswords (100).json"
OUTPUT_FILE = "classified_crosswords.json"
BATCH_SIZE = 100
MAX_RPM = 10  # Gemini 2.5 Flash free tier
CONCURRENCY = 3  # allow up to 3 concurrent requests
SECONDS_PER_REQUEST = 60 / MAX_RPM  # 6 seconds per request
MAX_RETRIES = 3  # retry if JSON parse fails

# Load dataset
with open(INPUT_FILE, "r") as f:
    data = json.load(f)


def safe_json_parse(text):
    """Try to extract a JSON array from text, even if the model outputs extra text."""
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def classify_batch(batch, semaphore):
    async with semaphore:
        batch_json = json.dumps([{"answer": a, "clue": c} for a, c in batch], indent=0)

        prompt = f"""
You are a strict crossword classification assistant.
Classify clues into JSON only. Do not include any text explanation.

Classify each clue in this JSON array:

{batch_json}

For each entry, assign:
- "difficulty": 1 (easy, very common words, straightforward clues, commonplace references) → 5 (very hard, less common words, tricky clues, uncommon references). If the word is less common, or wouldn't be known by at least a 5th grader, give a higher difficulty (4-5).
- "topics": list of topics (Geography, Math/Science, Pop Culture, Wordplay, General Knowledge)

Return a JSON array in the same order, each element like:
{{"difficulty": 3, "topics": ["Wordplay"]}}
"""

        parsed = None
        for attempt in range(1, MAX_RETRIES + 1):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text
            parsed = safe_json_parse(text)
            if parsed:
                break
            else:
                print(f"JSON parse failed on attempt {attempt}. Retrying...")
                await asyncio.sleep(SECONDS_PER_REQUEST)
        if not parsed:
            print("Max retries reached. Returning empty classifications for this batch.")
            parsed = [{"difficulty": None, "topics": []}] * len(batch)

        # space out calls to avoid bursts > 10 RPM
        await asyncio.sleep(SECONDS_PER_REQUEST)
        return parsed


async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY)  # allow up to 3 at once
    tasks = []

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        tasks.append(asyncio.create_task(classify_batch(batch, semaphore)))

    results = []
    for i, task in enumerate(asyncio.as_completed(tasks)):
        classifications = await task
        batch = data[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        for (answer, clue), classification in zip(batch, classifications):
            results.append({"answer": answer, "clue": clue, **classification})
        # Incremental save every 5,000 entries
        if len(results) % 5000 == 0:
            async with AIOFile(OUTPUT_FILE, "w") as afp:
                writer = Writer(afp)
                await writer(json.dumps(results, indent=2))
                await afp.fsync()
        print(f"Processed {len(results)} / {len(data)} entries")

    # Save final output
    async with AIOFile(OUTPUT_FILE, "w") as afp:
        writer = Writer(afp)
        await writer(json.dumps(results, indent=2))
        await afp.fsync()


# Run the async processing
asyncio.run(main())
