"""Deterministic, local Devanagari -> casual Roman/Latin script conversion.

This is SCRIPT CONVERSION, not translation: Hindi/Hinglish words keep their
Hindi meaning, just written in Latin letters the way people actually type
Hinglish (e.g. "mujhe", "nahi", "hai") -- not academic ITRANS/IAST output
with diacritics (e.g. "mujhē", "nahī̃"). No LLM or external API is used;
everything here is a plain lookup table + a character-level phonetic rule
engine, both fully local and deterministic.

Three tiers, checked per whitespace-delimited token:
  1. COMMON_WORDS   -- the highest-frequency Hindi function/discourse words
                        (pronouns, particles, copula, common verbs), given
                        their conventional casual-Hinglish spelling directly,
                        since these are exactly the words where a naive
                        character-level transliteration sounds most
                        unnatural (e.g. anusvara handling varies by word:
                        "nahi" drops the nasal, "yahan" keeps it as "n" --
                        no single character rule gets both right).
  2. ENGLISH_LOANWORDS -- common UX/tech English words as they're commonly
                        spelled out phonetically in Devanagari (e.g.
                        "प्राइसिंग"), mapped back to their real English
                        spelling ("pricing") rather than a phonetic
                        respelling ("praaising"), per the requirement that
                        English words stay readable as English.
  3. Character-level phonetic fallback -- for everything else: a standard
                        consonant+matra+anusvara mapping, with a simple
                        word-final schwa-deletion heuristic (the one
                        well-established, simple rule; full Hindi medial
                        schwa deletion is a genuinely hard, actively
                        researched NLP problem and is intentionally NOT
                        attempted here -- see the module docstring's
                        limitations note in think_aloud.py).

Any character this module doesn't recognize is passed through unchanged
rather than guessed at, per "preserve rather than invent."
"""
import re
from typing import Optional

DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")

# ---- tier 1: common Hindi function/discourse words -------------------------
COMMON_WORDS = {
    "मुझे": "mujhe", "मुझको": "mujhko", "तुझे": "tujhe", "उसे": "use", "इसे": "ise",
    "यहाँ": "yahan", "यहां": "yahan", "वहाँ": "wahan", "वहां": "wahan",
    "कहाँ": "kahan", "कहां": "kahan", "यहीं": "yahi", "वहीं": "wahi",
    "नहीं": "nahi", "नहि": "nahi", "है": "hai", "हैं": "hain", "हूँ": "hoon", "हूं": "hoon",
    "था": "tha", "थी": "thi", "थे": "the", "हो": "ho", "होगा": "hoga", "होगी": "hogi",
    "क्या": "kya", "कैसे": "kaise", "कैसा": "kaisa", "कैसी": "kaisi",
    "क्यों": "kyun", "क्योंकि": "kyunki", "कौन": "kaun", "कौनसा": "kaunsa",
    "लेकिन": "lekin", "और": "aur", "या": "ya", "तो": "to", "भी": "bhi",
    "सिर्फ": "sirf", "अभी": "abhi", "फिर": "phir", "बस": "bas",
    "मैं": "main", "तुम": "tum", "आप": "aap", "हम": "hum",
    "ये": "ye", "यह": "yeh", "वो": "wo", "वह": "wah",
    "इसका": "iska", "उसका": "uska", "इसके": "iske", "उसके": "uske",
    "इसकी": "iski", "उसकी": "uski", "इसमें": "isme", "उसमें": "usme",
    "का": "ka", "की": "ki", "के": "ke", "को": "ko", "से": "se",
    "में": "mein", "पे": "pe", "पर": "par", "तक": "tak", "ने": "ne",
    "रही": "rahi", "रहा": "raha", "रहे": "rahe", "रहीं": "rahi",
    "दिख": "dikh", "दिखा": "dikha", "दिखी": "dikhi", "दिखता": "dikhta", "दिखती": "dikhti",
    "ढूंढना": "dhoondhna", "ढूंढ": "dhoondh", "ढूंढ़ना": "dhoondhna",
    "मिलेगा": "milega", "मिलेगी": "milegi", "मिला": "mila", "मिली": "mili", "मिल": "mil",
    "समझ": "samajh", "समझा": "samjha", "समझी": "samjhi",
    "पता": "pata", "लगा": "laga", "लगी": "lagi", "लग": "lag",
    "सोचा": "socha", "सोची": "sochi", "सोच": "soch",
    "देख": "dekh", "देखा": "dekha", "देखी": "dekhi", "देखना": "dekhna",
    "चाहिए": "chahiye", "करना": "karna", "करता": "karta", "करती": "karti", "करते": "karte",
    "किया": "kiya", "की।": "ki.", "गया": "gaya", "गई": "gayi", "गयी": "gayi",
    "जाना": "jaana", "जाऊं": "jaaun", "आना": "aana", "आया": "aaya", "आई": "aayi",
    "कुछ": "kuch", "सब": "sab", "कोई": "koi", "कहीं": "kahin",
    "अच्छा": "achha", "अच्छी": "achhi", "ठीक": "theek", "बहुत": "bahut",
    "एक": "ek", "दो": "do", "तीन": "teen",
}

# ---- tier 2: English loanwords, common phonetic Devanagari spellings -------
ENGLISH_LOANWORDS = {
    "प्राइसिंग": "pricing", "प्राईसिंग": "pricing",
    "सेक्शन": "section", "वेबसाइट": "website", "वेब्साइट": "website",
    "नेविगेशन": "navigation", "लॉगिन": "login", "लोगिन": "login",
    "बटन": "button", "पेज": "page", "क्विज़": "quiz", "क्विज": "quiz",
    "मेनू": "menu", "मेन्यू": "menu", "ड्रॉपडाउन": "dropdown",
    "क्लिक": "click", "स्क्रॉल": "scroll", "सर्च": "search",
    "होम": "home", "कॉन्टैक्ट": "contact", "अबाउट": "about",
    "साइन": "sign", "रजिस्टर": "register", "अकाउंट": "account",
    "कार्ट": "cart", "चेकआउट": "checkout", "कैटेगरी": "category",
    "एक्टिविटीज़": "activities", "एक्टिविटी": "activity",
    "पार्टिसिपेट": "participate", "डैशबोर्ड": "dashboard",
    "प्रोफाइल": "profile", "सेटिंग्स": "settings", "सेटिंग": "setting",
    "फॉर्म": "form", "फिल्टर": "filter", "सॉर्ट": "sort",
    "अपडेट": "update", "डाउनलोड": "download", "अपलोड": "upload",
    "नोटिफिकेशन": "notification", "पॉपअप": "popup", "बैनर": "banner",
    "लिंक": "link", "आइकॉन": "icon", "आइकन": "icon",
}

# ---- tier 3: character-level phonetic fallback -----------------------------
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}
_INDEP_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_DANDA = {"।": ".", "॥": "."}
_DIGITS = {d: str(i) for i, d in enumerate("०१२३४५६७८९")}
_HALANT = "्"
_ANUSVARA = "ं"
_CHANDRABINDU = "ँ"
_VISARGA = "ः"


def _phonetic_transliterate(token: str) -> str:
    """Character-level fallback for a Devanagari token not in either
    dictionary. Applies one simple, well-established rule (word-final schwa
    deletion) and otherwise renders each akshara with its inherent vowel."""
    out = []
    n = len(token)
    i = 0
    last_was_bare_consonant_with_inherent_a = False
    while i < n:
        ch = token[i]
        nxt = token[i + 1] if i + 1 < n else ""
        if ch in _CONSONANTS:
            roman = _CONSONANTS[ch]
            if nxt == _HALANT:
                out.append(roman)
                i += 2
                last_was_bare_consonant_with_inherent_a = False
                continue
            if nxt in _MATRAS:
                out.append(roman + _MATRAS[nxt])
                i += 2
                last_was_bare_consonant_with_inherent_a = False
                continue
            if nxt in (_ANUSVARA, _CHANDRABINDU):
                out.append(roman + "an")
                i += 2
                last_was_bare_consonant_with_inherent_a = False
                continue
            if nxt == _VISARGA:
                out.append(roman + "ah")
                i += 2
                last_was_bare_consonant_with_inherent_a = False
                continue
            out.append(roman + "a")
            last_was_bare_consonant_with_inherent_a = True
            i += 1
            continue
        elif ch in _INDEP_VOWELS:
            out.append(_INDEP_VOWELS[ch])
            last_was_bare_consonant_with_inherent_a = False
            i += 1
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            out.append("n")
            last_was_bare_consonant_with_inherent_a = False
            i += 1
        elif ch == _VISARGA:
            out.append("h")
            last_was_bare_consonant_with_inherent_a = False
            i += 1
        elif ch in _DANDA:
            out.append(_DANDA[ch])
            last_was_bare_consonant_with_inherent_a = False
            i += 1
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
            last_was_bare_consonant_with_inherent_a = False
            i += 1
        elif ch == _HALANT:
            # Stray halant with no preceding consonant in this token -- skip.
            i += 1
        else:
            # Unrecognized character (rare Devanagari extension, or already
            # non-Devanagari punctuation caught up in the token) -- preserve
            # it rather than guessing.
            out.append(ch)
            last_was_bare_consonant_with_inherent_a = False
            i += 1

    result = "".join(out)
    # Word-final schwa deletion: a bare trailing consonant+inherent-"a" at
    # the very end of a multi-akshara word is usually silent in speech
    # (e.g. "राम" is spoken "raam", not "raama"). Only applied when the
    # token has more than one Devanagari character, to avoid mangling
    # single-akshara words where the inherent vowel IS the whole word.
    if last_was_bare_consonant_with_inherent_a and len(token) > 1 and result.endswith("a"):
        result = result[:-1]
    return result


def _romanize_token(token: str) -> str:
    """Romanize one whitespace-delimited token, preserving any leading/
    trailing punctuation attached to it."""
    if not DEVANAGARI_RANGE.search(token):
        return token  # no Devanagari at all -- leave completely unchanged

    # Peel off leading/trailing non-Devanagari punctuation so dictionary
    # lookups match the bare word (e.g. "यहीं।" -> core "यहीं" + trailing "।").
    m = re.match(r"^([^ऀ-ॿ]*)(.*?)([^ऀ-ॿ]*)$", token, re.DOTALL)
    lead, core, trail = m.groups() if m else ("", token, "")

    if core in COMMON_WORDS:
        romanized_core = COMMON_WORDS[core]
    elif core in ENGLISH_LOANWORDS:
        romanized_core = ENGLISH_LOANWORDS[core]
    else:
        romanized_core = _phonetic_transliterate(core)

    trail_romanized = "".join(_DANDA.get(c, c) for c in trail)
    lead_romanized = "".join(_DANDA.get(c, c) for c in lead)
    return lead_romanized + romanized_core + trail_romanized


def romanize(text: Optional[str]) -> Optional[str]:
    """Convert Devanagari script in `text` to casual Roman/Latin script.

    This is script conversion only -- meaning is never translated. English
    words and any text already in Latin script are returned unchanged.
    Returns None/"" unchanged (never fabricates text for an empty input).
    """
    if not text:
        return text
    if not DEVANAGARI_RANGE.search(text):
        return text  # already Latin/English -- nothing to do

    parts = re.split(r"(\s+)", text)  # keep original whitespace exactly
    romanized = "".join(p if p.isspace() else _romanize_token(p) for p in parts)

    # Capitalize the first alphabetic character for a natural sentence look.
    for i, ch in enumerate(romanized):
        if ch.isalpha():
            romanized = romanized[:i] + ch.upper() + romanized[i + 1:]
            break
    return romanized
