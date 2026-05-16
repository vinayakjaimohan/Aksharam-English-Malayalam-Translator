import pandas as pd
import json

print("Loading CSV...")

df = pd.read_csv("paras.csv")

# Clean prefixes
df["en"] = df["English Text (en_XX)"].str.replace("en_XX: ", "", regex=False)
df["ml"] = df["Malayalam Text (ml_IN)"].str.replace("ml_IN: ", "", regex=False)

df = df[["en", "ml"]].dropna().drop_duplicates()

print("Original size:", len(df))

augmented = []

for _, row in df.iterrows():
    en = row["en"]
    ml = row["ml"]

    # original
    augmented.append({"en": en, "ml": ml})

    # safe augmentations
    augmented.append({"en": en.lower(), "ml": ml})
    augmented.append({"en": en.replace(".", ""), "ml": ml})
    augmented.append({"en": "Please " + en, "ml": ml})

# Remove duplicates
seen = set()
final_aug = []

for item in augmented:
    key = (item["en"], item["ml"])
    if key not in seen:
        seen.add(key)
        final_aug.append(item)

print("Augmented size:", len(final_aug))

# Save JSON
with open("augmented_novel.json", "w", encoding="utf-8") as f:
    json.dump(final_aug, f, ensure_ascii=False, indent=2)

print("Saved augmented_novel.json")