from numeric_tokenizer import NumericTokenizer


def test_numeric_encoding():
    tok = NumericTokenizer()
    tokens, binary = tok.encode("7 plus 3")
    assert len(tokens) == len(binary)
    decoded = tok.decode(tokens)
    assert "7" in decoded


def test_tokenizer_roundtrip_regression():
    tok = NumericTokenizer()
    text = "12 and 34"
    tokens, binary = tok.encode(text)
    assert all(len(vec) == 9 for vec in binary)
    decoded = tok.decode(tokens)
    assert decoded == text


def test_tokenizer_id_stability():
    tok = NumericTokenizer()
    ids1, _ = tok.encode("5 5")
    ids2, _ = tok.encode("5")
    assert ids1[0] == ids1[2] == ids2[0]
