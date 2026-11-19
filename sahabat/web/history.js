// Panel Riwayat: daftar kejadian dan pemutar klip bukti. Keadaan ringkas (jumlah kejadian, simpan klip)
// datang dari app.js lewat peristiwa "sahabat-riwayat"; daftar lengkap diambil dari /api/riwayat saat panel dibuka.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const dialog = $("history-dialog");
  const JENIS = { jatuh: "Jatuh", kekerasan: "Indikasi kekerasan" };
  const MODE = { full: "seluruh badan", upper: "setengah badan" };
  const TANDA = { benar: "Kejadian benar", keliru: "Alarm keliru", tindak: "Perlu tindak lanjut" };

  let seenSeq = null;      // jumlah kejadian saat panel terakhir dibuka (atau saat halaman dimuat)
  let lastSeq = 0;
  let keepClips = true;
  let days = 7;
  let items = [];
  let playing = null;      // kejadian yang klipnya sedang tampil
  let refreshTimer = null;
  let cameras = "";        // daftar kamera Mode CCTV terakhir (untuk pilihan saring)

  function updateBadge() {
    const n = seenSeq === null ? 0 : Math.max(0, lastSeq - seenSeq);
    $("history-count").textContent = n;
    $("history-count").hidden = n === 0;
  }

  function updateToggle() {
    $("clips-toggle").textContent = "Simpan klip: " + (keepClips ? "hidup" : "mati");
    $("clips-toggle").dataset.on = keepClips ? "1" : "0";
    $("history-note").textContent = keepClips
      ? `Klip disimpan hanya di laptop ini dan terhapus otomatis setelah ${days} hari.`
      : "Klip tidak disimpan. Kejadian tetap dicatat tanpa video.";
  }

  function when(epoch) {
    const d = new Date(epoch * 1000);
    const time = d.toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const today = new Date();
    if (d.toDateString() === today.toDateString()) return time;
    return d.toLocaleDateString("id-ID", { day: "numeric", month: "short" }) + " " + time;
  }

  function describe(it) {
    const parts = [JENIS[it.jenis] || it.jenis, it.kamera_nama, when(it.waktu)].filter(Boolean);
    if (MODE[it.mode]) parts.push(MODE[it.mode]);
    return parts.join(" · ");
  }

  function renderList() {
    const list = $("history-list");
    list.textContent = "";
    $("history-empty").hidden = items.length > 0;
    for (const it of items) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "history-item";
      btn.dataset.jenis = it.jenis;
      if (playing && playing.id === it.id) btn.classList.add("active");
      const title = document.createElement("span");
      title.className = "history-title";
      title.textContent = JENIS[it.jenis] || it.jenis;
      const meta = document.createElement("span");
      meta.className = "history-meta";
      const clip = it.klip ? "klip tersedia" : (it.menyimpan ? "menyimpan klip…" : "tanpa klip");
      meta.textContent = [it.kamera_nama, when(it.waktu), MODE[it.mode], clip].filter(Boolean).join(" · ");
      btn.append(title, meta);
      if (it.tanda) {
        const badge = document.createElement("span");
        badge.className = "history-badge";
        badge.dataset.tanda = it.tanda;
        badge.textContent = TANDA[it.tanda] || it.tanda;
        title.append(badge);
      }
      btn.disabled = !it.klip;
      btn.addEventListener("click", () => play(it));
      li.append(btn);
      list.append(li);
    }
  }

  function load() {
    const filter = $("history-kamera").hidden ? "" : $("history-kamera").value;
    const url = "/api/riwayat" + (filter ? "?kamera=" + encodeURIComponent(filter) : "");
    return fetch(url, { cache: "no-store" })
      .then((r) => r.json())
      .then((r) => {
        items = r.kejadian || [];
        keepClips = !!r.simpan_klip;
        const s = r.ringkasan;
        $("history-summary").textContent = s && items.length
          ? `Tindak lanjut: ${s.benar} kejadian benar · ${s.keliru} alarm keliru · ${s.tindak} perlu tindak lanjut · ${s.belum} belum ditandai`
          : "";
        if (r.hari) days = r.hari;
        updateToggle();
        renderList();
        clearTimeout(refreshTimer);
        if (dialog.open && items.some((it) => it.menyimpan)) refreshTimer = setTimeout(load, 1500);
      })
      .catch(() => {});
  }

  function play(it) {
    playing = it;
    $("history-caption").textContent = describe(it);
    window.SahabatInspeksi.open(it);
    renderList();
  }

  function stopPlayer() {
    window.SahabatInspeksi.close();
    playing = null;
    $("history-caption").textContent = "Pilih kejadian untuk memutar klip buktinya.";
  }

  document.addEventListener("sahabat-riwayat-ubah", () => load());

  document.addEventListener("sahabat-riwayat", (ev) => {
    const r = ev.detail;
    if (seenSeq === null || dialog.open) seenSeq = r.seq;   // angka di tombol: kejadian yang belum dilihat sejak
                                                         // halaman dimuat atau panel terakhir dibuka
    const changed = r.seq !== lastSeq;
    lastSeq = r.seq;
    keepClips = !!r.klip;
    $("history").hidden = false;
    updateBadge();
    updateToggle();
    if (dialog.open && changed) load();
  });

  $("history").addEventListener("click", () => {
    seenSeq = lastSeq;
    updateBadge();
    stopPlayer();
    load().then(() => {
      const first = items.find((it) => it.klip);
      if (first) play(first);
    });
    dialog.showModal();
  });

  document.addEventListener("sahabat-cctv", (ev) => {
    const list = ev.detail.cctv ? ev.detail.cctv.kamera : [];
    const key = list.map((k) => k.id + ":" + k.nama).join("|");
    if (key === cameras) return;
    cameras = key;
    const select = $("history-kamera");
    const keep = select.value;
    select.textContent = "";
    select.add(new Option("Semua kamera", ""));
    list.forEach((k) => select.add(new Option(k.nama, k.id)));
    select.value = list.some((k) => k.id === keep) ? keep : "";
    select.hidden = list.length === 0;
  });
  $("history-kamera").addEventListener("change", () => { stopPlayer(); load(); });

  $("history-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => { clearTimeout(refreshTimer); stopPlayer(); });

  $("clips-toggle").addEventListener("click", () => {
    fetch("/api/riwayat/klip", { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify({ on: !keepClips }) })
      .then((r) => r.json())
      .then((r) => { if (r.ok) { keepClips = r.on; updateToggle(); } })
      .catch(() => {});
  });
})();
