import json
import random

print("Loading datasets...")

with open("aligned_dataset.json", "r", encoding="utf-8") as f:
    main_data = json.load(f)

with open("augmented_novel.json", "r", encoding="utf-8") as f:
    aug_data = json.load(f)

print("Real data:", len(main_data))
print("Augmented data:", len(aug_data))

# -----------------------------
# Take ratio (90% real, 10% augmented)
# -----------------------------
main_sample = main_data[:9000]
aug_sample = aug_data[:1000]

combined = main_sample + aug_sample

# -----------------------------
# Remove duplicates
# -----------------------------
seen = set()
unique_data = []

for item in combined:
    key = (item["en"], item["ml"])
    if key not in seen:
        seen.add(key)
        unique_data.append(item)

print("After dedup:", len(unique_data))

# -----------------------------
# Shuffle
# -----------------------------
random.shuffle(unique_data)

# -----------------------------
# Save
# -----------------------------
with open("final_dataset.json", "w", encoding="utf-8") as f:
    json.dump(unique_data, f, ensure_ascii=False, indent=2)

print("Saved final_dataset.json")