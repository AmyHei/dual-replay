"""CLINC150 dataset loading.

Native 10-domain split from the original CLINC150 paper (Larson et al., 2019):
each of the 10 domains contains exactly 15 intents. This matches the community
CL benchmark split used by O-LoRA (Wang et al., 2023) and related work.

The previous 15-sub-domain variant is preserved as `build_15_domain_protocol`
for backward compatibility with older experiment logs.
"""
from datasets import load_dataset
import random


# Authoritative 10-domain mapping from clinc/oos-eval repo (data/domains.json).
CLINC150_DOMAINS: dict[str, list[str]] = {
    "banking": [
        "freeze_account", "routing", "pin_change", "bill_due", "pay_bill",
        "account_blocked", "interest_rate", "min_payment", "bill_balance",
        "transfer", "order_checks", "balance", "spending_history",
        "transactions", "report_fraud",
    ],
    "credit_cards": [
        "replacement_card_duration", "expiration_date", "damaged_card",
        "improve_credit_score", "report_lost_card", "card_declined",
        "credit_limit_change", "apr", "redeem_rewards", "credit_limit",
        "rewards_balance", "application_status", "credit_score", "new_card",
        "international_fees",
    ],
    "kitchen_and_dining": [
        "food_last", "confirm_reservation", "how_busy", "ingredients_list",
        "calories", "nutrition_info", "recipe", "restaurant_reviews",
        "restaurant_reservation", "meal_suggestion", "restaurant_suggestion",
        "cancel_reservation", "ingredient_substitution", "cook_time",
        "accept_reservations",
    ],
    "home": [
        "what_song", "play_music", "todo_list_update", "reminder",
        "reminder_update", "calendar_update", "order_status", "update_playlist",
        "shopping_list", "calendar", "next_song", "order", "todo_list",
        "shopping_list_update", "smart_home",
    ],
    "auto_and_commute": [
        "current_location", "oil_change_when", "oil_change_how", "uber",
        "traffic", "tire_pressure", "schedule_maintenance", "gas", "mpg",
        "distance", "directions", "last_maintenance", "gas_type",
        "tire_change", "jump_start",
    ],
    "travel": [
        "plug_type", "travel_notification", "translate", "flight_status",
        "international_visa", "timezone", "exchange_rate", "travel_suggestion",
        "travel_alert", "vaccines", "lost_luggage", "book_flight",
        "book_hotel", "carry_on", "car_rental",
    ],
    "utility": [
        "weather", "alarm", "date", "find_phone", "share_location", "timer",
        "make_call", "calculator", "definition", "measurement_conversion",
        "flip_coin", "spelling", "time", "roll_dice", "text",
    ],
    "work": [
        "pto_request_status", "next_holiday", "insurance_change", "insurance",
        "meeting_schedule", "payday", "taxes", "income", "rollover_401k",
        "pto_balance", "pto_request", "w2", "schedule_meeting",
        "direct_deposit", "pto_used",
    ],
    "small_talk": [
        "who_made_you", "meaning_of_life", "who_do_you_work_for",
        "do_you_have_pets", "what_are_your_hobbies", "fun_fact",
        "what_is_your_name", "where_are_you_from", "goodbye", "thank_you",
        "greeting", "tell_joke", "are_you_a_bot", "how_old_are_you",
        "what_can_i_ask_you",
    ],
    "meta": [
        "change_speed", "user_name", "whisper_mode", "yes", "change_volume",
        "no", "change_language", "repeat", "change_accent", "cancel",
        "sync_device", "change_user_name", "change_ai_name", "reset_settings",
        "maybe",
    ],
}

# Deterministic domain order for sequential CL (alphabetical keys).
CLINC150_DOMAIN_ORDER = sorted(CLINC150_DOMAINS.keys())


def load_clinc150_raw():
    return load_dataset("clinc_oos", "plus")


def _get_intent_names(ds):
    return ds["train"].features["intent"].names


def build_10_domain_protocol(seed: int = 42) -> list[dict]:
    """Return 10 sequential domains, each with 15 intents, from native CLINC150.

    Ordering: permuted by `seed` (5 random orderings per paper convention).
    Each returned dict has: domain_id, domain_name, intents (global intent ids),
    intent_names, train, test, validation.
    """
    ds = load_clinc150_raw()
    intent_names = _get_intent_names(ds)
    name_to_id = {n: i for i, n in enumerate(intent_names)}

    rng = random.Random(seed)
    order = CLINC150_DOMAIN_ORDER.copy()
    rng.shuffle(order)

    domains: list[dict] = []
    for d, dom_name in enumerate(order):
        intent_set = {name_to_id[n] for n in CLINC150_DOMAINS[dom_name]}
        train = [ex for ex in ds["train"] if ex["intent"] in intent_set]
        test = [ex for ex in ds["test"] if ex["intent"] in intent_set]
        val = [ex for ex in ds["validation"] if ex["intent"] in intent_set]
        domains.append({
            "domain_id": d,
            "domain_name": dom_name,
            "intents": sorted(intent_set),
            "intent_names": CLINC150_DOMAINS[dom_name],
            "train": train,
            "test": test,
            "validation": val,
        })
    return domains


def build_15_domain_protocol(seed: int = 42) -> list[dict]:
    """Legacy 15-sub-domain split (alphabetical by intent name, 10 intents each).

    Kept for backward compatibility with experiments logged before the switch
    to the native 10-domain protocol. New experiments should use
    `build_10_domain_protocol`.
    """
    ds = load_clinc150_raw()
    intent_names = _get_intent_names(ds)
    oos_idx = intent_names.index("oos") if "oos" in intent_names else -1
    in_scope = [i for i in range(len(intent_names)) if i != oos_idx]
    assert len(in_scope) == 150

    in_scope_sorted = sorted(in_scope, key=lambda i: intent_names[i])
    domains: list[dict] = []
    for d in range(15):
        domain_intents = in_scope_sorted[d * 10 : (d + 1) * 10]
        intent_set = set(domain_intents)
        train = [ex for ex in ds["train"] if ex["intent"] in intent_set]
        test = [ex for ex in ds["test"] if ex["intent"] in intent_set]
        val = [ex for ex in ds["validation"] if ex["intent"] in intent_set]
        domains.append({
            "domain_id": d,
            "intents": domain_intents,
            "intent_names": [intent_names[i] for i in domain_intents],
            "train": train,
            "test": test,
            "validation": val,
        })
    return domains


def get_general_buffer(max_size: int = 1000, seed: int = 42) -> list[dict]:
    """OOS (out-of-scope) examples used as the general-knowledge replay stream."""
    ds = load_clinc150_raw()
    intent_names = _get_intent_names(ds)
    if "oos" not in intent_names:
        raise ValueError("OOS intent not found in CLINC150")
    oos_idx = intent_names.index("oos")
    oos = [ex for ex in ds["train"] if ex["intent"] == oos_idx]
    rng = random.Random(seed)
    if len(oos) > max_size:
        oos = rng.sample(oos, max_size)
    return [{"text": ex["text"], "intent": "oos"} for ex in oos]
