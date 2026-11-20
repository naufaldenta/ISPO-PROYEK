// Layar guru "mode pameran". Keadaan datang dari server lewat WebSocket; halaman ini hanya menampilkan.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const RED = new Set(["jatuh", "kekerasan"]);

  let started = false;     // tombol Mulai sudah ditekan (izin bunyi dari browser)
  let muted = false;
  let bar = "kamera_terputus";
  let lastSeq = null;
  let audio = null;        // AudioContext
  let siren = null;        // { osc, gain, timer }
  let recording = false;   // perekam uji menyala (aplikasi dinyalakan dengan --rekam)
  let wordsStatus = "memuat";   // keadaan pendengar kata kasar
  let flipped = false;          // gambar kamera dibalik kiri-kanan
  let remote = false;           // halaman dibuka dari HP / laptop lain: hanya melihat, bukan mengatur
  let cctvOn = false;           // Mode CCTV: grid banyak kamera
  let enlarged = null;          // kamera yang sedang diperbesar di Mode CCTV
  let alarmKamera = null;       // kamera yang alarmnya sedang tampil di pita
  const tiles = new Map();
  const STATUS_TEXT = { penuh: "dinilai penuh", siaga: "siaga", lambat: "lambat", terputus: "terputus — mencoba lagi…" };
  let lastNoticeSeq = null;
  let markTimer = null;
  const VOLUME_KEY = "sahabat-volume";
  let volume = readVolume();    // 0..1, diatur dari dialog Perangkat; diingat browser laptop ini
  const MARKS = { KeyJ: "jatuh", KeyK: "kekerasan", KeyB: "biasa" };

  // ---- bunyi alarm: dua nada bergantian, dibuat langsung tanpa berkas suara ----
  function sirenOn() {
    if (siren || !audio) return;
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.type = "square";
    osc.frequency.value = 880;
    gain.gain.value = 0.18 * volume;
    osc.connect(gain).connect(audio.destination);
    osc.start();
    let high = true;
    const timer = setInterval(() => {
      high = !high;
      osc.frequency.setValueAtTime(high ? 880 : 620, audio.currentTime);
    }, 420);
    siren = { osc, gain, timer };
  }

  function sirenOff() {
    if (!siren) return;
    clearInterval(siren.timer);
    try { siren.osc.stop(); } catch (e) { /* sudah berhenti */ }
    siren.osc.disconnect();
    siren.gain.disconnect();
    siren = null;
  }

  // Bunyi pendek dua nada untuk peringatan kata kasar. Tidak dimainkan saat sirene alarm merah berbunyi.
  function beep() {
    if (!started || !audio || siren) return;
    [[880, 0], [660, 0.14]].forEach(([hz, at]) => {
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.type = "sine";
      osc.frequency.value = hz;
      gain.gain.value = 0.2 * volume;
      osc.connect(gain).connect(audio.destination);
      osc.start(audio.currentTime + at);
      osc.stop(audio.currentTime + at + 0.12);
    });
  }

  function readVolume() {
    try {
      const v = parseFloat(localStorage.getItem(VOLUME_KEY));
      return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 1;
    } catch (e) {
      return 1;
    }
  }

  document.addEventListener("sahabat-volume", (ev) => {
    volume = Math.min(1, Math.max(0, Number(ev.detail) || 0));
    try { localStorage.setItem(VOLUME_KEY, String(volume)); } catch (e) { /* tetap berlaku untuk sesi ini */ }
    if (siren) siren.gain.gain.setValueAtTime(0.18 * volume, audio.currentTime);
  });

  // Contoh bunyi sirene sekitar 1,3 detik dengan volume sekarang (tidak dimainkan saat sirene asli berbunyi).
  document.addEventListener("sahabat-volume-test", () => {
    if (siren) return;
    if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
    audio.resume();
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.type = "square";
    gain.gain.value = 0.18 * volume;
    osc.connect(gain).connect(audio.destination);
    [0, 0.42, 0.84].forEach((at, i) => osc.frequency.setValueAtTime(i % 2 ? 620 : 880, audio.currentTime + at));
    osc.start();
    osc.stop(audio.currentTime + 1.26);
  });

  function refreshSound() {
    if (started && RED.has(bar) && !muted) sirenOn(); else sirenOff();
  }

  // ---- Mode CCTV: grid ----
  function columns(n) {
    return n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4;
  }

  function tileSrc(id) {
    return "/video/gambar?kamera=" + encodeURIComponent(id) + "&kecil=1&t=" + Date.now();
  }

  // Ubin memakai satu gambar per permintaan, diminta ulang tiap 0,2 dtk (5 fps). Stream MJPEG per ubin akan
  // menghabiskan 6 sambungan yang diizinkan browser, sehingga tombol "Sudah diperiksa" tidak terkirim.
  function nextTile(id, delay) {
    const tile = tiles.get(id);
    if (!tile) return;
    clearTimeout(tile.timer);
    tile.timer = setTimeout(() => {
      if (!tiles.has(id) || !cctvOn) return;
      if (enlarged) { nextTile(id, 500); return; }          // grid tersembunyi: jangan meminta gambar
      tile.querySelector("img").src = tileSrc(id);
    }, delay);
  }

  function renderGrid(c) {
    const grid = $("grid");
    const ids = c.kamera.map((k) => k.id);
    if (enlarged && !ids.includes(enlarged)) enlarged = null;
    for (const [id, tile] of tiles) {
      if (!ids.includes(id)) { clearTimeout(tile.timer); tile.querySelector("img").removeAttribute("src"); tile.remove(); tiles.delete(id); }
    }
    c.kamera.forEach((k, i) => {
      let tile = tiles.get(k.id);
      if (!tile) {
        tile = document.createElement("button");
        tile.type = "button";
        tile.className = "tile";
        tile.innerHTML = '<img alt=""><span class="tile-name"></span><span class="tile-status"></span>' +
          '<span class="tile-bar" hidden></span>';
        tile.querySelector("img").addEventListener("load", () => nextTile(k.id, 200));
        tile.querySelector("img").addEventListener("error", () => nextTile(k.id, 1000));
        tile.querySelector("img").src = tileSrc(k.id);
        tile.addEventListener("click", () => enlarge(k.id));
        tiles.set(k.id, tile);
      }
      if (grid.children[i] !== tile) grid.insertBefore(tile, grid.children[i] || null);
      tile.querySelector("img").alt = "Kamera " + k.nama;
      tile.querySelector(".tile-name").textContent = k.nama + (k.prioritas ? " ★" : "");
      tile.title = k.peringatan ? k.nama + ": " + k.peringatan : "";
      const status = tile.querySelector(".tile-status");
      status.dataset.status = k.status;
      status.textContent = "● " + (STATUS_TEXT[k.status] || k.status);
      const red = RED.has(k.bar);
      tile.dataset.red = red ? "1" : "0";
      tile.querySelector(".tile-bar").hidden = !red;
      tile.querySelector(".tile-bar").textContent = red ? k.text : "";
    });
    grid.style.setProperty("--cols", columns(c.kamera.length));
    grid.hidden = !!enlarged;
    $("video").hidden = !enlarged;
    $("grid-back").hidden = !enlarged;
  }

  // Petunjuk pemasangan kamera (asisten_kamera) untuk kamera yang diperbesar, diperbarui tiap 2 dtk.
  let assistTimer = null;
  async function showAssist() {
    clearTimeout(assistTimer);
    const line = $("asisten-line");
    if (!enlarged) { line.hidden = true; return; }
    const id = enlarged;
    try {
      const r = await fetch("/api/asisten?kamera=" + encodeURIComponent(id), { cache: "no-store" });
      const result = await r.json();
      if (enlarged !== id) return;
      line.textContent = result.ok ? "Pemasangan: " + result.saran.join(" · ") : "";
      line.hidden = !result.ok;
    } catch (e) {
      line.hidden = true;
    }
    assistTimer = setTimeout(showAssist, 2000);
  }

  function enlarge(id) {
    enlarged = id;
    showAssist();
    $("video").src = "/video?kamera=" + encodeURIComponent(id) + "&t=" + Date.now();
    $("grid").hidden = true;
    $("video").hidden = false;
    $("grid-back").hidden = false;
    document.dispatchEvent(new CustomEvent("sahabat-cctv", { detail: { cctv: lastCctv, enlarged } }));
  }

  function backToGrid() {
    enlarged = null;
    showAssist();
    $("video").removeAttribute("src");
    $("video").hidden = true;
    $("grid").hidden = false;
    $("grid-back").hidden = true;
    document.dispatchEvent(new CustomEvent("sahabat-cctv", { detail: { cctv: lastCctv, enlarged } }));
  }

  function leaveCctv() {
    for (const tile of tiles.values()) { clearTimeout(tile.timer); tile.querySelector("img").removeAttribute("src"); tile.remove(); }
    tiles.clear();
    enlarged = null;
    showAssist();
    $("grid").hidden = true;
    $("grid-back").hidden = true;
    $("video").hidden = false;
    loadVideo();
  }

  let lastCctv = null;

  // ---- tampilan ----
  function render(m) {
    const c = m.cctv && m.cctv.aktif ? m.cctv : null;
    const wasOn = cctvOn;
    cctvOn = !!c;
    lastCctv = c;
    document.body.classList.toggle("cctv", cctvOn);
    if (cctvOn) renderGrid(c);
    else if (wasOn) leaveCctv();
    alarmKamera = c ? c.alarm_kamera : null;
    $("ack-all").hidden = !c || (c.alarm_list || []).length < 2;
    document.dispatchEvent(new CustomEvent("sahabat-cctv", { detail: { cctv: c, enlarged } }));
    if (m.devices) document.dispatchEvent(new CustomEvent("sahabat-devices", { detail: m.devices }));
    if (m.riwayat) document.dispatchEvent(new CustomEvent("sahabat-riwayat", { detail: m.riwayat }));
    if ("zona" in m) document.dispatchEvent(new CustomEvent("sahabat-zona", { detail: m.zona }));
    if (m.jaringan) document.dispatchEvent(new CustomEvent("sahabat-jaringan", { detail: m.jaringan }));
    remote = m.lokal === false;
    document.body.classList.toggle("remote", remote);
    $("remote-chip").hidden = !remote;
    bar = m.bar;
    document.body.dataset.bar = m.bar;
    $("bar-text").textContent = m.text;
    $("fps").textContent = Math.round(m.fps);
    $("people").textContent = m.people;
    $("slow").hidden = !m.slow;
    $("slow").textContent = m.slow_text;
    recording = !!m.recording;
    $("rec").hidden = !recording;

    const w = m.words || { status: "tidak_tersedia", label: "Kata kasar: tidak tersedia", notice: "", notice_seq: 0 };
    wordsStatus = w.status;
    $("words").textContent = w.label;
    $("words").dataset.status = w.status;
    $("words").disabled = w.status === "tidak_tersedia";
    $("words-notice").textContent = w.notice;
    $("words-notice").hidden = !w.notice;
    flipped = !!m.flip;
    $("flip").textContent = "Balik kiri-kanan: " + (flipped ? "hidup" : "mati");
    document.body.classList.toggle("noticing", !!w.notice);

    if (lastSeq !== null && m.alarm_seq > lastSeq) muted = false;   // alarm baru selalu berbunyi
    lastSeq = m.alarm_seq;

    const red = RED.has(m.bar);
    if (!red) muted = false;
    $("ack").hidden = !red;
    $("mute").hidden = !red;
    $("mute").textContent = muted ? "Bunyikan" : "Senyap";
    refreshSound();
    if (lastNoticeSeq !== null && w.notice_seq > lastNoticeSeq) beep();   // sesudah refreshSound: sirene menang
    lastNoticeSeq = w.notice_seq;
  }

  // ---- sambungan ----
  function loadVideo() {
    if (cctvOn) {
      if (enlarged) $("video").src = "/video?kamera=" + encodeURIComponent(enlarged) + "&t=" + Date.now();
      for (const id of tiles.keys()) nextTile(id, 0);
      return;
    }
    $("video").src = "/video?t=" + Date.now();
  }

  function connect() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.onopen = () => { $("reconnect").hidden = true; loadVideo(); };
    ws.onmessage = (ev) => render(JSON.parse(ev.data));
    ws.onclose = () => {
      $("reconnect").hidden = false;
      sirenOff();
      if (remote) {
        // HP / laptop lain: kode akses diganti atau jaringan lokal dimatikan -> kembali ke halaman kode akses
        fetch("/api/ping", { cache: "no-store" })
          .then((r) => { if (r.status === 401 || r.status === 403) location.reload(); })
          .catch(() => {});
      }
      setTimeout(connect, 1000);
    };
    ws.onerror = () => ws.close();
  }

  $("video").addEventListener("error", () => setTimeout(loadVideo, 1000));

  // ---- tombol ----
  function acknowledge() {
    if (!RED.has(bar)) return;
    if (cctvOn) {
      fetch("/api/ack", { method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ kamera: alarmKamera }) }).catch(() => {});
      return;
    }
    fetch("/api/ack", { method: "POST" }).catch(() => {});
  }

  $("ack-all").addEventListener("click", () => {
    fetch("/api/ack", { method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ kamera: "semua" }) }).catch(() => {});
  });
  $("grid-back").addEventListener("click", backToGrid);

  $("start").addEventListener("click", () => {
    if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
    audio.resume();
    started = true;
    $("start-layer").hidden = true;
    const root = document.documentElement;
    if (root.requestFullscreen) root.requestFullscreen().catch(() => {});
    refreshSound();
  });

  // Hidup/mati pendengar kata kasar. Mati berarti mikrofonnya dilepas di sisi aplikasi.
  function toggleWords() {
    if (wordsStatus === "tidak_tersedia" || remote) return;
    fetch("/api/kata", { method: "POST", headers: { "Content-Type": "application/json" },
                         body: JSON.stringify({ on: wordsStatus === "mati" }) }).catch(() => {});
  }

  $("words").addEventListener("click", toggleWords);

  // Webcam yang gambarnya tercermin dibalik di aplikasi; layar, perekam, dan deteksi memakai gambar yang sama.
  function toggleFlip() {
    if (remote) return;
    fetch("/api/cermin", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify({ on: !flipped }) }).catch(() => {});
  }
  $("flip").addEventListener("click", toggleFlip);
  $("ack").addEventListener("click", acknowledge);

  $("mute").addEventListener("click", () => {
    muted = !muted;
    $("mute").textContent = muted ? "Bunyikan" : "Senyap";
    refreshSound();
  });

  // Saat merekam uji, pencatat menandai adegan yang baru diperagakan supaya rekaman bisa dinilai ulang.
  function mark(label) {
    fetch("/api/tanda", { method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ label }) })
      .then((r) => r.json())
      .then((r) => {
        $("mark").textContent = r.ok ? "Tanda dicatat: " + label : "Tanda gagal dicatat";
        $("mark").hidden = false;
        clearTimeout(markTimer);
        markTimer = setTimeout(() => { $("mark").hidden = true; }, 1500);
      })
      .catch(() => {});
  }

  document.addEventListener("keydown", (ev) => {
    if ($("device-dialog").open || $("history-dialog").open || $("lan-dialog").open || !$("zone-editor").hidden) return;
    if (recording && started && !ev.repeat && MARKS[ev.code]) { mark(MARKS[ev.code]); return; }
    if (ev.code === "KeyM" && started && !ev.repeat) { toggleWords(); return; }
    if (ev.code === "KeyC" && started && !ev.repeat) { toggleFlip(); return; }
    if (ev.code !== "Space" || !started) return;
    ev.preventDefault();          // jangan sampai spasi "menekan" tombol yang sedang terfokus
    acknowledge();
  });

  connect();
})();
