from sahabat.predlog import PredLog


def test_unwritable_log_folder_disables_logging_instead_of_crashing(tmp_path):
    blocker = tmp_path / "bukan-folder"
    blocker.write_text("x", encoding="utf-8")
    log = PredLog(blocker / "log")                          # induknya berkas: folder tidak bisa dibuat
    log.write({"t": 1.0})
    log.close()


def test_normal_folder_still_logs(tmp_path):
    log = PredLog(tmp_path / "log")
    log.write({"t": 1.0})
    log.close()
    assert len(list((tmp_path / "log").glob("prediksi-*.jsonl"))) == 1


def test_prefix_names_the_file_and_limits_the_cleanup(tmp_path):
    import os, time
    old = tmp_path / "prediksi-20200101.jsonl"
    old.write_text("x")
    os.utime(old, (time.time() - 30 * 86400,) * 2)
    log = PredLog(tmp_path, prefix="kata")
    log.write({"a": 1})
    log.close()
    assert old.exists() and len(list(tmp_path.glob("kata-*.jsonl"))) == 1
