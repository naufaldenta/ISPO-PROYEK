// Panel "Kamera": menambah IP camera, channel DVR/NVR, HP (IP Webcam), atau capture card DVR.
// Hanya di laptop SAHABAT (server menolak /api/kamera dari HP). Kata sandi tidak pernah dikirim balik oleh server:
// isian kata sandi kosong saat mengubah kamera berarti kata sandi lama tetap dipakai.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const dialog = $("kamera-dialog");
  let brands = [];
  let editing = null;          // kamera yang sedang diubah (data publik), null = kamera baru
  let mode = "merek";
  let busy = false;
  let found = null;            // hasil pencarian ONVIF yang dipilih

  const say = (text) => { $("kamera-message").textContent = text || ""; };

  async function api(url, options) {
    const response = await fetch(url, { cache: "no-store", ...options });
    let result = {};
    try { result = await response.json(); } catch (e) { /* jawaban kosong */ }
    if (!response.ok || result.ok === false) throw new Error(result.error || result.pesan || "Gagal.");
    return result;
  }
  const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify(body || {}) });

  function setBusy(value) {
    busy = value;
    dialog.querySelectorAll("button, input, select").forEach((el) => {
      if (el.id !== "kamera-close") el.disabled = value;
    });
  }

  // ---- daftar kamera tersimpan ----
  function renderList(items) {
    const list = $("kamera-list");
    list.textContent = "";
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "kamera-empty";
      li.textContent = "Belum ada kamera CCTV. Tambahkan di bawah.";
      list.append(li);
      return;
    }
    items.forEach((it) => {
      const li = document.createElement("li");
      li.className = "kamera-item";
      const info = document.createElement("div");
      const name = document.createElement("b");
      name.textContent = it.nama;
      const where = document.createElement("span");
      where.className = "kamera-where";
      where.textContent = it.alamat_tampil + (it.potong ? ` · ubin ${it.potong[1] + 1} dari ${it.potong[0]}×${it.potong[0]}` : "");
      info.append(name, where);
      if (it.posisi === "tinggi") {
        const pos = document.createElement("span");
        pos.className = "kamera-where";
        pos.textContent = "Posisi: tinggi (eksperimen)" + (it.peringatan ? ". " + it.peringatan : "");
        info.append(pos);
      }
      if (it.sandi_rusak) {
        const warn = document.createElement("span");
        warn.className = "kamera-warn";
        warn.textContent = "Kata sandi perlu diisi ulang (tekan Ubah).";
        info.append(warn);
      }
      const flags = document.createElement("div");
      flags.className = "kamera-flags";
      [["aktif", "Aktif di Mode CCTV"], ["prioritas", "Prioritas"]].forEach(([key, text]) => {
        const label = document.createElement("label");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = !!it[key];
        box.addEventListener("change", () => setFlag(it, key, box));
        label.append(box, text);
        flags.append(label);
      });
      info.append(flags);
      const actions = document.createElement("div");
      actions.className = "kamera-actions";
      [["Pakai", () => use(it)], ["Ubah", () => edit(it)], ["Hapus", () => remove(it)]].forEach(([label, fn]) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.addEventListener("click", fn);
        actions.append(b);
      });
      li.append(info, actions);
      list.append(li);
    });
  }

  async function load() {
    try {
      const result = await api("/api/kamera");
      if (!brands.length) {
        brands = result.merek;
        const select = $("kamera-merek");
        select.textContent = "";
        brands.forEach((b) => select.add(new Option(b.nama, b.id)));
        brandChanged();
      }
      renderList(result.kamera);
    } catch (error) {
      say(error.message);
    }
  }

  async function setFlag(it, key, box) {
    try {
      await post("/api/kamera/" + encodeURIComponent(it.id) + "/tanda", { [key]: box.checked });
      say(`${it.nama}: ${key === "aktif" ? (box.checked ? "ikut" : "tidak ikut") + " Mode CCTV"
        : (box.checked ? "prioritas" : "bukan prioritas")}.`);
      it[key] = box.checked;
    } catch (error) {
      box.checked = !box.checked;
      say(error.message);
    }
  }

  // ---- Mode CCTV ----
  function showCctv(st) {
    $("cctv-toggle").dataset.on = st.hidup ? "1" : "0";
    $("cctv-toggle").textContent = "Mode CCTV: " + (st.hidup ? "hidup" : "mati");
    $("cctv-message").textContent = st.hidup ? st.pesan
      : "Mode CCTV menilai semua kamera aktif sekaligus dalam grid.";
  }

  async function loadCctv() {
    try { showCctv(await api("/api/cctv")); } catch (error) { $("cctv-message").textContent = error.message; }
    loadHitung();
  }

  // ---- perangkat hitung Mode CCTV (GPU / CPU) ----
  function hitungText(info) {
    const parts = [
      info.gpu.length ? "GPU terdeteksi: " + info.gpu.join(", ") : "GPU tidak terdeteksi",
      "paket GPU: " + (info.dml ? "terpasang" : "belum terpasang"),
      "uji kesetaraan: " + (info.lolos === true ? "lolos" : info.lolos === false ? "belum lolos"
        : "belum diuji di GPU dan resolusi ini"),
    ];
    let text = parts.join(" · ") + ".";
    if (!info.dml) text += " Jalankan scripts\\siapkan_gpu.ps1 untuk memasang paket GPU.";
    else if (!info.model_onnx) {
      text += " Model GPU (models\\yolo11n-pose_416.onnx) belum ada; salin dari paket rilis SAHABAT.";
    }
    if (info.terakhir && typeof info.terakhir.ms === "number") {
      const adaptor = typeof info.terakhir.gpu_id === "number" ? ` (adaptor #${info.terakhir.gpu_id})` : "";
      text += ` GPU terakhir diukur: ${info.terakhir.ms} ms per gambar${adaptor}.`;
    }
    if (info.aktif) text += " Sedang dipakai: " + info.aktif + ".";
    return text;
  }

  async function loadHitung() {
    try {
      const info = await api("/api/cctv/hitung");
      $("cctv-hitung").value = info.hitung;
      $("cctv-hitung-info").textContent = hitungText(info);
    } catch (error) {
      $("cctv-hitung-info").textContent = error.message;
    }
  }

  $("cctv-hitung").addEventListener("change", async () => {
    const select = $("cctv-hitung");
    select.disabled = true;
    try {
      const info = await post("/api/cctv/hitung", { hitung: select.value });
      $("cctv-hitung-info").textContent = info.pesan + " " + hitungText(info);
    } catch (error) {
      $("cctv-hitung-info").textContent = error.message;
      loadHitung();
    } finally {
      select.disabled = busy;
    }
  });

  $("cctv-toggle").addEventListener("click", async () => {
    const on = $("cctv-toggle").dataset.on !== "1";
    setBusy(true);
    $("cctv-message").textContent = on ? "Menyalakan Mode CCTV: menyambung kamera dan mengukur komputer…"
      : "Kembali ke Mode Pameran…";
    try {
      showCctv(await post("/api/cctv", { hidup: on }));
    } catch (error) {
      $("cctv-message").textContent = error.message;
    } finally {
      setBusy(false);
    }
  });

  async function use(it) {
    setBusy(true);
    say(`Menyambungkan ${it.nama}…`);
    try {
      await post("/api/perangkat", { camera: "ipcam:" + it.id });
      say(`${it.nama} dipakai sebagai kamera. Bila layar tetap "terputus", tekan Ubah lalu Tes sambungan.`);
    } catch (error) {
      say(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(it) {
    if (!window.confirm(`Hapus kamera "${it.nama}" dari daftar?`)) return;
    setBusy(true);
    try {
      await api("/api/kamera/" + encodeURIComponent(it.id), { method: "DELETE" });
      if (editing && editing.id === it.id) resetForm();
      say(`${it.nama} dihapus.`);
      await load();
    } catch (error) {
      say(error.message);
    } finally {
      setBusy(false);
    }
  }

  // ---- formulir ----
  function setMode(next) {
    mode = next;
    dialog.querySelectorAll("[data-kamera-mode]").forEach((b) => b.classList.toggle("active", b.dataset.kameraMode === mode));
    $("kamera-cari-box").hidden = mode !== "cari";
    $("kamera-form").hidden = mode === "cari";
    dialog.querySelectorAll("[data-only]").forEach((el) => {
      el.hidden = !el.dataset.only.split(" ").includes(mode);
    });
    if (mode === "perangkat" || mode === "video") loadDevices();
  }

  function brandChanged() {
    const b = brands.find((x) => x.id === $("kamera-merek").value);
    if (!b) return;
    $("kamera-port").value = b.port;
    $("kamera-channel-wrap").hidden = !b.channel || mode !== "merek";
  }

  function gridChanged() {
    const grid = Number($("kamera-grid").value);
    const select = $("kamera-ubin");
    const keep = Number(select.value) || 1;
    select.textContent = "";
    for (let i = 1; i <= grid * grid; i++) select.add(new Option("Ubin " + i, String(i)));
    select.value = String(Math.min(keep, Math.max(1, grid * grid)));
    $("kamera-ubin").hidden = !grid;
  }

  let deviceToken = 0;
  async function loadDevices(selected) {
    const mine = ++deviceToken;           // ganti mode lalu Ubah memanggil ini dua kali; jawaban lama diabaikan
    try {
      const result = await api("/api/perangkat");
      if (mine !== deviceToken) return;
      const select = $("kamera-perangkat");
      select.textContent = "";
      result.cameras.forEach((c) => select.add(new Option(c.name, c.id)));
      if (!result.cameras.length) select.add(new Option("Tidak ada webcam / capture card", ""));
      const videos = $("kamera-video");
      videos.textContent = "";
      (result.videos || []).forEach((v) => videos.add(new Option(v.name, v.name)));
      if (!(result.videos || []).length) videos.add(new Option("Belum ada video di folder video", ""));
      if (selected && mode === "perangkat") select.value = selected;
      if (selected && mode === "video") videos.value = selected;
    } catch (error) {
      say(error.message);
    }
  }

  function resetForm() {
    editing = null;
    found = null;
    $("kamera-form").reset();
    $("kamera-sandi").placeholder = "";
    $("kamera-form-title").textContent = "Tambah kamera";
    $("kamera-cancel").hidden = true;
    $("kamera-preview").hidden = true;
    $("kamera-test-msg").textContent = "";
    brandChanged();
    gridChanged();
  }

  function edit(it) {
    resetForm();
    editing = it;
    setMode(it.jenis);
    $("kamera-form-title").textContent = "Ubah kamera: " + it.nama;
    $("kamera-cancel").hidden = false;
    $("kamera-nama").value = it.nama;
    $("kamera-posisi").value = it.posisi || "dada";
    $("kamera-pengguna").value = it.pengguna || "";
    $("kamera-sandi").placeholder = it.ada_sandi && !it.sandi_rusak ? "(tetap memakai kata sandi tersimpan)" : "";
    if (it.jenis === "merek") {
      $("kamera-merek").value = it.merek;
      brandChanged();
      $("kamera-host").value = it.host;
      $("kamera-port").value = it.port;
      $("kamera-channel").value = it.channel;
      $("kamera-sub").checked = !!it.substream;
    } else if (it.jenis === "alamat") {
      $("kamera-alamat").value = it.alamat_edit;      // tanpa login; {SANDI} = kata sandi tersimpan di query
    } else if (it.jenis === "video") {
      loadDevices(it.video);
    } else {
      loadDevices(it.perangkat);
    }
    $("kamera-grid").value = it.potong ? String(it.potong[0]) : "0";
    gridChanged();
    if (it.potong) $("kamera-ubin").value = String(it.potong[1] + 1);
    $("kamera-nama").focus();
  }

  function collect() {
    const data = { nama: $("kamera-nama").value.trim(), jenis: mode, pengguna: $("kamera-pengguna").value.trim(),
                   posisi: $("kamera-posisi").value };
    if (editing) data.id = editing.id;
    const sandi = $("kamera-sandi").value;
    if (sandi) data.sandi = sandi;
    if (mode === "merek") {
      Object.assign(data, { merek: $("kamera-merek").value, host: $("kamera-host").value.trim(),
                            port: Number($("kamera-port").value) || 0, channel: Number($("kamera-channel").value) || 1,
                            substream: $("kamera-sub").checked });
    } else if (mode === "alamat") {
      data.alamat = $("kamera-alamat").value.trim();
    } else if (mode === "video") {
      data.video = $("kamera-video").value;
      delete data.pengguna;
    } else {
      data.perangkat = $("kamera-perangkat").value;
      delete data.pengguna;
    }
    const grid = Number($("kamera-grid").value);
    data.potong = grid ? [grid, Number($("kamera-ubin").value) - 1] : null;
    return data;
  }

  async function test() {
    setBusy(true);
    $("kamera-preview").hidden = true;
    $("kamera-test-msg").textContent = "Mencoba menyambung (paling lama 10 detik)…";
    try {
      const r = await post("/api/kamera/uji", collect());
      $("kamera-test-msg").textContent = r.pesan;
      if (r.cuplikan) {
        $("kamera-preview").src = "data:image/jpeg;base64," + r.cuplikan;
        $("kamera-preview").hidden = false;
      }
    } catch (error) {
      $("kamera-test-msg").textContent = error.message;
    } finally {
      setBusy(false);
    }
  }

  async function save(event) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    try {
      const r = await post("/api/kamera", collect());
      say(`${r.kamera.nama} tersimpan. Tekan Pakai untuk memakainya sebagai kamera.`);
      resetForm();
      await load();
    } catch (error) {
      say(error.message);
    } finally {
      setBusy(false);
    }
  }

  // ---- cari otomatis (ONVIF) ----
  async function search() {
    setBusy(true);
    $("kamera-found").textContent = "";
    $("kamera-onvif").hidden = true;
    say("Mencari kamera di jaringan (sekitar 3 detik)…");
    try {
      const r = await post("/api/kamera/cari");
      if (!r.ditemukan.length) {
        say("Tidak ada kamera yang menjawab. Banyak kamera murah tidak mendukung pencarian: pakai Pilih merek.");
        return;
      }
      say(`${r.ditemukan.length} kamera ditemukan. Pilih salah satu.`);
      r.ditemukan.forEach((cam) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "kamera-found-item";
        b.textContent = `${cam.host}${cam.nama ? " · " + cam.nama : ""}${cam.merek ? " · " + cam.merek : ""}`;
        b.addEventListener("click", () => {
          found = cam;
          $("kamera-onvif").hidden = false;
          $("kamera-onvif-title").textContent = "Login kamera " + cam.host;
          $("kamera-onvif-user").focus();
        });
        $("kamera-found").append(b);
      });
    } catch (error) {
      say(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function onvifAddress() {
    if (!found) return;
    setBusy(true);
    say("Meminta alamat stream dari kamera…");
    const user = $("kamera-onvif-user").value.trim();
    const pass = $("kamera-onvif-pass").value;
    try {
      const r = await post("/api/kamera/onvif", { xaddr: found.xaddr, pengguna: user, sandi: pass,
                                                  substream: $("kamera-onvif-sub").checked,
                                                  channel: Number($("kamera-onvif-channel").value) || 1 });
      const cam = found;
      resetForm();
      setMode("alamat");
      $("kamera-nama").value = (cam.nama || "Kamera " + cam.host).slice(0, 40);
      $("kamera-alamat").value = r.alamat;
      $("kamera-pengguna").value = user;
      $("kamera-sandi").value = pass;
      say("Alamat ditemukan. Tekan Tes sambungan, lalu Simpan.");
    } catch (error) {
      say(error.message);
    } finally {
      setBusy(false);
    }
  }

  // ---- tombol ----
  $("kamera").addEventListener("click", () => {
    if (!dialog.open) dialog.showModal();
    say("");
    load();
    loadCctv();
  });
  $("kamera-close").addEventListener("click", () => dialog.close());
  dialog.querySelectorAll("[data-kamera-mode]").forEach((b) => b.addEventListener("click", () => {
    if (editing && b.dataset.kameraMode !== editing.jenis) resetForm();
    setMode(b.dataset.kameraMode);
    brandChanged();
  }));
  $("kamera-merek").addEventListener("change", brandChanged);
  $("kamera-grid").addEventListener("change", gridChanged);
  $("kamera-test").addEventListener("click", test);
  $("kamera-form").addEventListener("submit", save);
  $("kamera-cancel").addEventListener("click", () => { resetForm(); setMode("merek"); });
  $("kamera-search").addEventListener("click", search);
  $("kamera-onvif-go").addEventListener("click", onvifAddress);
  setMode("merek");
  gridChanged();
})();
