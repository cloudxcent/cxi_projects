import json
import re
from rapidfuzz import process, fuzz

class RestaurantMenuHelper:
    """
    Helper class to load and query the restaurant menu.
    """

    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.menu_data = json.load(f)

    def get_meal_period(self, hour):
        """
        Determine meal period based on hour.
        Returns one of: 'breakfast', 'lunch', 'snacks', 'dinner' or default 'dinner'.
        """
        for period, times in self.menu_data["time_ranges"].items():
            if times["start"] <= hour < times["end"]:
                return period
        return "dinner"

    def _clean_text(self, text):
        """
        Normalize text input for matching.
        """
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def find_dish(self, dish_name, meal_period):
        """
        Exact match lookup with fallback to all_day and other periods.
        """
        cleaned = self._clean_text(dish_name)

        def search(cats):
            for dishes in cats.values():
                for d in dishes:
                    if self._clean_text(d["name"]) == cleaned:
                        return d
            return None

        # Search current period
        d = search(self.menu_data["menu"].get(meal_period, {}))
        if d:
            return d

        # Search all_day
        d = search(self.menu_data["menu"].get("all_day", {}))
        if d:
            return d

        # Search other periods
        for p, cats in self.menu_data["menu"].items():
            if p in (meal_period, "all_day"):
                continue
            d = search(cats)
            if d:
                return d
        return None

    def find_dish_fuzzy(self, dish_name, meal_period):
        """
        Fuzzy match dish name with threshold 70.
        """
        cleaned = self._clean_text(dish_name)

        def collect(periods):
            names = []
            for p in periods:
                for dishes in self.menu_data["menu"].get(p, {}).values():
                    names.extend(d["name"] for d in dishes)
            return list(set(names))

        candidates = collect([meal_period, "all_day"])
        match, score, _ = process.extractOne(cleaned, candidates, scorer=fuzz.ratio) if candidates else (None, 0, None)
        if score >= 70:
            return self.find_dish(match, meal_period)

        others = [p for p in self.menu_data["menu"] if p not in (meal_period, "all_day")]
        candidates = collect(others)
        match, score, _ = process.extractOne(cleaned, candidates, scorer=fuzz.ratio) if candidates else (None, 0, None)

        if score >= 70:
            return self.find_dish(match, meal_period)

        return None

    def _get_all_dish_names(self):
        """
        Return list of all dish names across all periods.
        """
        names = []
        for cats in self.menu_data["menu"].values():
            for dishes in cats.values():
                names.extend(d["name"] for d in dishes)
        return list(set(names))

    def get_price(self, dish_name, meal_period):
        d = self.find_dish_fuzzy(dish_name, meal_period)
        return d["price"] if d else None

    def get_description(self, dish_name, meal_period):
        d = self.find_dish_fuzzy(dish_name, meal_period)
        return d.get("description") if d else None
