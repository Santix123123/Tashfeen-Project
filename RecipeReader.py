import json
import random
import re
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT_RECIPES_DIR = Path("content/recipes")
JSON_RECIPES_FILE = Path("tashfeen.json")
BASE_DIR = Path(__file__).resolve().parent


def normalize_text(text):
    if not text:
        return ""

    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title(text, filepath):
    fm_title = re.search(r'(?im)^title\s*:\s*["\']?(.*?)["\']?\s*$', text)
    if fm_title:
        return fm_title.group(1).strip()

    md_title = re.search(r'(?m)^#\s+(.+)$', text)
    if md_title:
        return md_title.group(1).strip()

    return filepath.stem.replace("-", " ").replace("_", " ").title()


def extract_ingredients_from_markdown(text):
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^(#+\s*)?ingredients\s*:?\s*$", stripped, re.IGNORECASE):
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    ingredients = []
    for line in lines[start_idx:]:
        stripped = line.strip()

        if re.match(r"^#+\s+", stripped):
            break

        if re.match(r"^(directions|instructions|method|steps)\s*:?\s*$", stripped, re.IGNORECASE):
            break

        bullet_match = re.match(r"^[-*+]\s+(.*)$", stripped)
        if bullet_match:
            ingredient = bullet_match.group(1).strip()
            if ingredient:
                ingredients.append(ingredient)
            continue

        numbered_match = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if numbered_match:
            ingredient = numbered_match.group(1).strip()
            if ingredient:
                ingredients.append(ingredient)
            continue

    return ingredients


def simplify_markdown_ingredient(text):
    text = normalize_text(text)

    text = re.sub(
        r"\b(chopped|diced|minced|sliced|fresh|ground|crushed|optional|to taste)\b",
        "",
        text,
    )

    text = re.sub(
        r"^\d+([./-]\d+)?\s*(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|lb|lbs|oz|g|kg|ml|l|clove|cloves|slice|slices|can|cans|package|packages|pound|gram|grams|ounce|ounces)?\s*",
        "",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_markdown_recipe(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    raw_ingredients = extract_ingredients_from_markdown(text)
    normalized_ingredients = []

    for ing in raw_ingredients:
        cleaned = simplify_markdown_ingredient(ing)
        if cleaned:
            normalized_ingredients.append(cleaned)

    normalized_ingredients = list(dict.fromkeys(normalized_ingredients))

    return {
        "title": extract_title(text, filepath),
        "author": filepath.parent.name,
        "source": "markdown",
        "file_path": str(filepath),
        "ingredients_raw": raw_ingredients,
        "ingredients_normalized": normalized_ingredients,
    }


def load_markdown_recipes(root_dir):
    recipes = []

    if not root_dir.exists():
        return recipes

    for filepath in root_dir.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in {".md", ".markdown", ".mdx"}:
            try:
                recipe = read_markdown_recipe(filepath)
                if recipe["ingredients_normalized"]:
                    recipes.append(recipe)
            except Exception as e:
                print(f"Could not read markdown recipe {filepath}: {e}")

    return recipes


def pick_json_ingredient_name(ingredient_obj):
    lemma = ingredient_obj.get("lemma")
    base = ingredient_obj.get("base")
    literal = ingredient_obj.get("literal")

    if lemma:
        return lemma
    if base:
        return base
    if literal:
        return literal
    return ""


def read_json_recipes(json_path):
    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    recipes = []

    for item in data:
        ingredients_raw = []
        ingredients_normalized = []

        for ing in item.get("ingredients", []):
            literal = ing.get("literal", "")
            picked = pick_json_ingredient_name(ing)

            if literal:
                ingredients_raw.append(literal)

            normalized = normalize_text(picked)
            if normalized:
                ingredients_normalized.append(normalized)

        ingredients_normalized = list(dict.fromkeys(ingredients_normalized))

        if ingredients_normalized:
            recipes.append(
                {
                    "title": item.get("title", "Untitled"),
                    "author": item.get("author", "unknown"),
                    "source": "json",
                    "file_path": item.get("filename", ""),
                    "ingredients_raw": ingredients_raw,
                    "ingredients_normalized": ingredients_normalized,
                }
            )

    return recipes


def deduplicate_recipes(recipes):
    by_title = {}

    for recipe in recipes:
        title_key = normalize_text(recipe["title"])

        if title_key not in by_title:
            by_title[title_key] = recipe
        else:
            existing = by_title[title_key]
            if existing["source"] == "markdown" and recipe["source"] == "json":
                by_title[title_key] = recipe

    return list(by_title.values())


def load_all_recipes():
    markdown_recipes = load_markdown_recipes(ROOT_RECIPES_DIR)
    json_recipes = read_json_recipes(JSON_RECIPES_FILE)

    print(f"Loaded {len(markdown_recipes)} markdown recipes.")
    print(f"Loaded {len(json_recipes)} json recipes.")

    combined = markdown_recipes + json_recipes
    combined = deduplicate_recipes(combined)

    print(f"Using {len(combined)} recipes after deduplication.")
    return combined


def parse_user_ingredients(user_input):
    return [normalize_text(x) for x in user_input.split(",") if normalize_text(x)]


def ingredient_matches(user_ing, recipe_ing):
    user_ing = normalize_text(user_ing)
    recipe_ing = normalize_text(recipe_ing)

    if not user_ing or not recipe_ing:
        return False

    if user_ing == recipe_ing:
        return True

    user_words = set(user_ing.split())
    recipe_words = set(recipe_ing.split())

    if not user_words or not recipe_words:
        return False

    if user_words.issubset(recipe_words) or recipe_words.issubset(user_words):
        return True

    return False


def compute_features(user_ingredients, recipe_ingredients):
    matched_user = set()
    matched_recipe = set()

    for user_ing in user_ingredients:
        for recipe_ing in recipe_ingredients:
            if ingredient_matches(user_ing, recipe_ing):
                matched_user.add(user_ing)
                matched_recipe.add(recipe_ing)
                break

    overlap_count = len(matched_user)
    missing_count = len(user_ingredients) - overlap_count
    extra_count = len(recipe_ingredients) - len(matched_recipe)

    user_coverage = overlap_count / len(user_ingredients) if user_ingredients else 0.0
    recipe_coverage = overlap_count / len(recipe_ingredients) if recipe_ingredients else 0.0

    union_count = len(set(user_ingredients).union(set(recipe_ingredients)))
    jaccard = overlap_count / union_count if union_count else 0.0

    return {
        "overlap_count": overlap_count,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "user_coverage": user_coverage,
        "recipe_coverage": recipe_coverage,
        "jaccard": jaccard,
    }


def feature_vector(feature_dict):
    return [
        feature_dict["overlap_count"],
        feature_dict["missing_count"],
        feature_dict["extra_count"],
        feature_dict["user_coverage"],
        feature_dict["recipe_coverage"],
        feature_dict["jaccard"],
    ]


def generate_positive_query(recipe_ingredients):
    n = len(recipe_ingredients)

    if n == 1:
        return recipe_ingredients[:]

    min_k = 1
    max_k = n
    k = random.randint(min_k, max_k)
    return random.sample(recipe_ingredients, k)


def build_training_data(recipes, positives_per_recipe=6, negatives_per_positive=3):
    X = []
    y = []

    if len(recipes) < 2:
        return X, y

    for i, recipe in enumerate(recipes):
        recipe_ingredients = recipe["ingredients_normalized"]

        for _ in range(positives_per_recipe):
            query = generate_positive_query(recipe_ingredients)

            pos_features = compute_features(query, recipe_ingredients)
            X.append(feature_vector(pos_features))
            y.append(1)

            other_indices = [j for j in range(len(recipes)) if j != i]
            sampled_negatives = random.sample(
                other_indices,
                min(negatives_per_positive, len(other_indices)),
            )

            for neg_idx in sampled_negatives:
                neg_recipe = recipes[neg_idx]
                neg_features = compute_features(query, neg_recipe["ingredients_normalized"])
                X.append(feature_vector(neg_features))
                y.append(0)

    return X, y


def train_decision_tree(recipes):
    X, y = build_training_data(recipes)

    if not X or not y:
        raise ValueError("Not enough recipe data to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred, digits=4))

    return model


def rank_recipes_with_tree(user_ingredients, recipes, model, top_n=12):
    results = []

    for recipe in recipes:
        features = compute_features(user_ingredients, recipe["ingredients_normalized"])
        vector = feature_vector(features)

        prob = model.predict_proba([vector])[0][1]

        manual_score = (
            2.0 * features["overlap_count"]
            - 1.0 * features["missing_count"]
            - 0.25 * features["extra_count"]
            + 2.0 * features["jaccard"]
        )

        combined_score = prob + manual_score

        results.append(
            {
                "title": recipe["title"],
                "author": recipe["author"],
                "source": recipe["source"],
                "file_path": recipe["file_path"],
                "ingredients_raw": recipe["ingredients_raw"],
                "ingredients_normalized": recipe["ingredients_normalized"],
                "probability": float(prob),
                "manual_score": float(manual_score),
                "combined_score": float(combined_score),
                "features": features,
            }
        )

    results.sort(
        key=lambda r: (
            -r["combined_score"],
            -r["features"]["overlap_count"],
            r["features"]["extra_count"],
            r["title"].lower(),
        )
    )

    return results[:top_n]


app = Flask(__name__, static_folder=".", static_url_path="")

RECIPES = load_all_recipes()
MODEL = train_decision_tree(RECIPES) if RECIPES else None


@app.get("/")
def index():
    return send_from_directory(".", "recipe_reader_webpage.html")


@app.get("/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"results": [], "message": "Enter ingredients to search."})

    # Title match layer
    normalized_query = normalize_text(query)
    title_matches = [
        recipe for recipe in RECIPES
        if normalized_query in normalize_text(recipe["title"])
    ]

    # Ingredient ranking layer
    user_ingredients = parse_user_ingredients(query)
    ranked_matches = []
    if user_ingredients and MODEL is not None:
        ranked_matches = rank_recipes_with_tree(user_ingredients, RECIPES, MODEL, top_n=12)

    # Merge title matches + ranked ingredient matches
    seen = set()
    final_results = []

    for recipe in title_matches + ranked_matches:
        key = normalize_text(recipe["title"])
        if key not in seen:
            seen.add(key)
            final_results.append(recipe)

    return jsonify(
        {
            "results": final_results[:12],
            "message": f"Found {len(final_results[:12])} matching recipes."
        }
    )


if __name__ == "__main__":
    app.run(debug=True)