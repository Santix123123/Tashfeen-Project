import json
import re
from pathlib import Path

ROOT_RECIPES_DIR = Path("content/recipes")
JSON_RECIPES_FILE = Path("tashfeen.json")  # change this to your actual json filename


def normalize_text(text):
    if not text:
        return ""

    text = text.lower().strip()
    text = text.replace("&", " and ")

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # remove extra spaces
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
        if re.match(r'^(#+\s*)?ingredients\s*:?\s*$', stripped, re.IGNORECASE):
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    ingredients = []
    for line in lines[start_idx:]:
        stripped = line.strip()

        if re.match(r'^#+\s+', stripped):
            break

        if re.match(r'^(directions|instructions|method|steps)\s*:?\s*$', stripped, re.IGNORECASE):
            break

        bullet_match = re.match(r'^[-*+]\s+(.*)$', stripped)
        if bullet_match:
            ingredient = bullet_match.group(1).strip()
            if ingredient:
                ingredients.append(ingredient)
            continue

        numbered_match = re.match(r'^\d+[.)]\s+(.*)$', stripped)
        if numbered_match:
            ingredient = numbered_match.group(1).strip()
            if ingredient:
                ingredients.append(ingredient)
            continue

    return ingredients


def simplify_markdown_ingredient(text):
    text = normalize_text(text)

    # remove common prep words
    text = re.sub(
        r'\b(chopped|diced|minced|sliced|fresh|ground|crushed|optional|to taste)\b',
        '',
        text
    )

    # remove simple leading quantities/units
    text = re.sub(
        r'^\d+([./-]\d+)?\s*(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|lb|lbs|oz|g|kg|ml|l|clove|cloves|slice|slices|can|cans|package|packages|pound|gram|grams|ounce|ounces)?\s*',
        '',
        text
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

        recipe = {
            "title": item.get("title", "Untitled"),
            "author": item.get("author", "unknown"),
            "source": "json",
            "file_path": item.get("filename", ""),
            "ingredients_raw": ingredients_raw,
            "ingredients_normalized": ingredients_normalized,
            "tags": item.get("tags", []),
            "directions": item.get("directions", []),
        }

        if recipe["ingredients_normalized"]:
            recipes.append(recipe)

    return recipes


def parse_user_ingredients(user_input):
    parts = user_input.split(",")
    cleaned = []

    for part in parts:
        normalized = normalize_text(part)
        if normalized:
            cleaned.append(normalized)

    return cleaned


def ingredient_matches(user_ing, recipe_ing):
    return (
        user_ing == recipe_ing
        or user_ing in recipe_ing
        or recipe_ing in user_ing
    )


def score_recipe(user_ingredients, recipe):
    recipe_ingredients = recipe["ingredients_normalized"]

    matched = []
    recipe_matches = []

    for user_ing in user_ingredients:
        for recipe_ing in recipe_ingredients:
            if ingredient_matches(user_ing, recipe_ing):
                matched.append(user_ing)
                recipe_matches.append(recipe_ing)
                break

    matched = list(dict.fromkeys(matched))
    recipe_matches = list(dict.fromkeys(recipe_matches))

    missing_user = [u for u in user_ingredients if u not in matched]
    extra_recipe = [r for r in recipe_ingredients if r not in recipe_matches]

    # simple ranking:
    # reward matched ingredients
    # slightly penalize recipes needing lots of extras
    score = len(matched) - (0.15 * len(extra_recipe))

    return {
        "score": score,
        "matched_user": matched,
        "matched_recipe": recipe_matches,
        "missing_user": missing_user,
        "extra_recipe": extra_recipe,
    }


def find_matching_recipes(user_ingredients, recipes, top_n=5):
    results = []

    for recipe in recipes:
        result = score_recipe(user_ingredients, recipe)

        if result["matched_user"]:
            results.append({
                "title": recipe["title"],
                "author": recipe["author"],
                "source": recipe["source"],
                "file_path": recipe["file_path"],
                "ingredients_raw": recipe["ingredients_raw"],
                "ingredients_normalized": recipe["ingredients_normalized"],
                "score": result["score"],
                "matched_user": result["matched_user"],
                "matched_recipe": result["matched_recipe"],
                "missing_user": result["missing_user"],
                "extra_recipe": result["extra_recipe"],
            })

    results.sort(
        key=lambda x: (
            -x["score"],
            len(x["missing_user"]),
            len(x["extra_recipe"]),
            x["title"].lower()
        )
    )

    return results[:top_n]


def load_all_recipes():
    recipes = []

    if ROOT_RECIPES_DIR.exists():
        markdown_recipes = load_markdown_recipes(ROOT_RECIPES_DIR)
        recipes.extend(markdown_recipes)
        print(f"Loaded {len(markdown_recipes)} markdown recipes.")

    if JSON_RECIPES_FILE.exists():
        json_recipes = read_json_recipes(JSON_RECIPES_FILE)
        recipes.extend(json_recipes)
        print(f"Loaded {len(json_recipes)} json recipes.")

    return recipes


def print_recipe_result(recipe, index):
    print(f"{index}. {recipe['title']}")
    print(f"   Author: {recipe['author']}")
    print(f"   Source: {recipe['source']}")
    print(f"   Score: {recipe['score']:.2f}")
    print(f"   Matched input: {', '.join(recipe['matched_user']) if recipe['matched_user'] else 'None'}")
    print(f"   Matching recipe ingredients: {', '.join(recipe['matched_recipe']) if recipe['matched_recipe'] else 'None'}")

    if recipe["missing_user"]:
        print(f"   Your unmatched ingredients: {', '.join(recipe['missing_user'])}")

    if recipe["extra_recipe"]:
        print(f"   Extra recipe ingredients needed: {', '.join(recipe['extra_recipe'])}")

    print("   Full recipe ingredients:")
    for ing in recipe["ingredients_raw"]:
        print(f"   - {ing}")

    print(f"   File: {recipe['file_path']}")
    print()


def main():
    print("Loading recipes...")
    recipes = load_all_recipes()
    print(f"Total recipes loaded: {len(recipes)}\n")

    if not recipes:
        print("No recipes found.")
        return

    user_input = input("Enter ingredients separated by commas: ")
    user_ingredients = parse_user_ingredients(user_input)

    if not user_ingredients:
        print("No valid ingredients entered.")
        return

    matches = find_matching_recipes(user_ingredients, recipes, top_n=5)

    if not matches:
        print("\nNo matching recipes found.")
        return

    print("\nTop matching recipes:\n")
    for i, recipe in enumerate(matches, start=1):
        print_recipe_result(recipe, i)


if __name__ == "__main__":
    main()