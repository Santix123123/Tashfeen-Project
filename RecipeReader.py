import os
import re
from pathlib import Path

ROOT_RECIPES_DIR = Path("content/recipes")


def extract_title(text, filepath):
    """
    Try to get the title from front matter first, then from the first markdown heading,
    then fall back to the file name.
    """
    # title: "Something"
    fm_title = re.search(r'(?im)^title\s*:\s*["\']?(.*?)["\']?\s*$', text)
    if fm_title:
        return fm_title.group(1).strip()

    # # Something
    md_title = re.search(r'(?m)^#\s+(.+)$', text)
    if md_title:
        return md_title.group(1).strip()

    return filepath.stem.replace("-", " ").replace("_", " ").title()


def extract_ingredients(text):
    """
    Find an ingredients section and return a cleaned list of ingredients.
    This handles markdown bullet lists under headings like:
    ## Ingredients
    ### Ingredients
    Ingredients
    """
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^\s{0,3}(#+\s*)?ingredients\s*:?\s*$', line.strip(), re.IGNORECASE):
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    ingredients = []
    for line in lines[start_idx:]:
        stripped = line.strip()

        # Stop if we hit another heading
        if re.match(r'^\s{0,3}#+\s+', stripped):
            break

        # Stop if we hit a numbered section title like "Directions" or "Instructions"
        if re.match(r'^(directions|instructions|method|steps)\s*:?\s*$', stripped, re.IGNORECASE):
            break

        # Bullet list items
        bullet_match = re.match(r'^[-*+]\s+(.*)$', stripped)
        if bullet_match:
            ingredient = bullet_match.group(1).strip()
            if ingredient:
                ingredients.append(ingredient)
            continue

        # Numbered list items
        numbered_match = re.match(r'^\d+[.)]\s+(.*)$', stripped)
        if numbered_match:
            ingredient = numbered_match.group(1).strip()
            if ingredient:
                ingredients.append(ingredient)
            continue

        # Blank lines are okay; just skip them
        if not stripped:
            continue

    return ingredients


def read_recipe_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    title = extract_title(text, filepath)
    ingredients = extract_ingredients(text)

    contributor = filepath.parent.name

    return {
        "title": title,
        "contributor": contributor,
        "file_path": str(filepath),
        "ingredients": ingredients,
    }


def load_all_recipes(root_dir):
    recipes = []

    for filepath in root_dir.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in {".md", ".markdown", ".mdx"}:
            try:
                recipe = read_recipe_file(filepath)
                recipes.append(recipe)
            except Exception as e:
                print(f"Could not read {filepath}: {e}")

    return recipes


def print_one_recipe(recipe):
    print(f"Title: {recipe['title']}")
    print(f"Contributor: {recipe['contributor']}")
    print("Ingredients:")
    for ingredient in recipe["ingredients"]:
        print(f"- {ingredient}")
    print(f"File: {recipe['file_path']}")


def main():
    recipes = load_all_recipes(ROOT_RECIPES_DIR)

    print(f"Loaded {len(recipes)} recipes.\n")

    if recipes:
        print_one_recipe(recipes[0])
    else:
        print("No recipes found.")


if __name__ == "__main__":
    main()