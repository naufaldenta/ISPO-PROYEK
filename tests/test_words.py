from sahabat.words import build_prompt, censor, load_words, tokens


def write(tmp_path, data: bytes):
    p = tmp_path / "kata.txt"
    p.write_bytes(data)
    return p


def test_list_keeps_order_and_skips_comments_and_blanks(tmp_path):
    p = write(tmp_path, "# catatan\nanjir\n\n  Asu  \n# babi\njancok\n".encode())
    assert load_words(p) == ("anjir", "asu", "jancok")


def test_messy_file_never_raises(tmp_path):                      # Review Focus 3
    data = b"\xef\xbb\xbfANJIR\r\nanjir\r\nkata dua\r\nab12\r\n\xff\xfe rusak\r\nfuck\r\n"
    assert load_words(write(tmp_path, data)) == ("anjir", "fuck")   # BOM, dobel, spasi, angka, byte rusak
    assert load_words(tmp_path / "tidak-ada.txt") == ()
    assert load_words(tmp_path) == ()                               # folder, bukan berkas


def test_tokens_are_lowercase_letter_runs():
    assert tokens("Anjir, kamu NGAPAIN sih?! asu-asu 123") == ["anjir", "kamu", "ngapain", "sih", "asu", "asu"]
    assert tokens("") == []


def test_censor_keeps_first_and_last_letter():
    assert [censor(w) for w in ("anjir", "asu", "kontol", "ah", "a")] == ["a***r", "a*u", "k****l", "a*", "a*"]


def test_prompt_lists_the_first_words_only():
    assert build_prompt(["anjir", "asu", "fuck"], 2) == "Percakapan anak sekolah dengan kata kasar: anjir, asu."
    assert build_prompt([], 40) == ""


def test_shipped_list_has_the_owner_examples_and_no_everyday_words():
    from sahabat.config import WORDS_PATH
    words = load_words(WORDS_PATH)
    assert {"anjir", "anjay", "asu", "kontol", "jancok", "fuck"} <= set(words)
    assert not {"babi", "monyet", "tai", "cok", "bodoh", "matamu"} & set(words)   # "matamu" juga berarti "mata kamu"
    assert len(words) <= 40                                         # semuanya muat di petunjuk


# ---- kata yang dipanjangkan atau diberi imbuhan (temuan uji pemilik proyek, 15 November 2026) ----

LISTED = ("anjir", "asu", "kontol", "jancok", "goblok", "asshole", "bangsat")


def test_stretched_and_suffixed_words_are_found_as_the_listed_word():
    from sahabat.words import find_listed
    heard = ("kontoli", "jancokiii", "anjiiir", "asuuu", "gobloknya", "kontolmu", "asune", "bangsatlah",
             "assshole", "ashole", "kontol")
    assert [find_listed(t, LISTED) for t in heard] == ["kontol", "jancok", "anjir", "asu", "goblok", "kontol", "asu",
                                                       "bangsat", "asshole", "asshole", "kontol"]


def test_everyday_lookalikes_are_still_not_found():
    from sahabat.words import find_listed
    for token in ("kontrol", "asuh", "asus", "asuransi", "anjuran", "pasu", "gobloks", "jancokers", "as", "a", ""):
        assert find_listed(token, LISTED) is None, token


def test_exact_word_wins_over_a_suffix_reading():
    from sahabat.words import find_listed
    # "matamu" terdaftar sendiri; jangan dibaca sebagai "mata" + "mu" atau dipotong menjadi kata lain
    assert find_listed("matamu", ("mata", "matamu")) == "matamu"
    assert find_listed("asui", ("asu", "asui")) == "asui"
