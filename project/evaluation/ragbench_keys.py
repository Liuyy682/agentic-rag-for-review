def document_id_from_sentence_key(key: str) -> str:
    digits = []
    for char in str(key):
        if char.isdigit():
            digits.append(char)
            continue
        break
    return "".join(digits) if digits else ""
