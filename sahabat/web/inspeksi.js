// Pemutar inspeksi klip bukti di panel Riwayat: gerak lambat, langkah per frame, garis waktu dengan penanda, zoom,
// kerangka yang bisa disembunyikan, sorot orang yang terlibat, kamera lain pada saat itu, tangkapan gambar, unduh MP4,
// dan tanda tindak lanjut guru. Semua frame klip dimuat sekali sebagai gambar lalu digambar di <canvas>, jadi langkah
// per frame dan zoom tidak meminta ulang ke server. Frame tidak diinterpolasi: 0,25x memang patah-patah.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const COLORS = { green: "#34c759", red: "#ff3b30", gray: "#a0a0a0", out: "#7d7d7d" };
  const TANDA = { benar: "Kejadian benar", keliru: "Alarm keliru", tindak: "Perlu tindak lanjut" };
  const KONKUREN = 4;
  const AMBANG_BAWAAN = 0.8;          // dipakai hanya bila server tidak mengirim ambang_kekerasan
  let playTimer = null;
  const HITUNG = window.SahabatInspeksiHitung;
  const canvas = $("insp-canvas");
  const ctx = canvas.getContext("2d");

  let info = null;          // jawaban /info
  let item = null;          // kejadian (bentuk daftar Riwayat)
  let images = [];
  let index = 0;
  let playing = false;
  let speed = 1;
  let clock = 0;            // detik klip saat diputar
  let lastTs = null;
  let view = { s: 1, x: 0, y: 0 };   // zoom dan geser, dalam piksel gambar
  let token = 0;            // klip yang sedang dimuat; jawaban lama diabaikan

  const say = (text) => { $("insp-msg").textContent = text || ""; };

  function involved() {
    if (!info) return [];
    return (info.meta && info.meta.orang) || (item && item.orang) || [];
  }

  function involvementText(ids) {
    if (!ids.length) return "";
    const names = ids.map((i) => "#" + i);
    const list = names.length === 1 ? names[0] : names.slice(0, -1).join(", ") + " dan " + names[names.length - 1];
    return `Orang ${list} terlibat dalam gerakan yang dinilai`;
  }

  // ---- menggambar --------------------------------------------------------------------------
  function size() {
    const img = images[index];
    const w = (info && info.meta && info.meta.lebar) || (img && img.naturalWidth) || 640;
    const h = (info && info.meta && info.meta.tinggi) || (img && img.naturalHeight) || 480;
    return { w, h };
  }

  function drawPeople(c, people, highlight) {
    const edges = info.tulang || [];
    const ids = new Set(involved());
    for (const p of people || []) {
      const dim = highlight && ids.size && !ids.has(p.id);
      const strong = highlight && ids.has(p.id);
      c.globalAlpha = dim ? 0.3 : 1;
      c.strokeStyle = c.fillStyle = COLORS[p.warna] || COLORS.green;
      c.lineWidth = p.warna === "out" ? 1 : (strong ? 5 : (p.warna === "red" ? 3 : 2));
      const ok = (i) => p.conf[i] >= 0.5;
      for (const [a, b] of edges) {
        if (!ok(a) || !ok(b)) continue;
        c.beginPath();
        c.moveTo(p.kp[a][0], p.kp[a][1]);
        c.lineTo(p.kp[b][0], p.kp[b][1]);
        c.stroke();
      }
      p.kp.forEach((pt, i) => { if (ok(i)) { c.beginPath(); c.arc(pt[0], pt[1], strong ? 5 : 3, 0, Math.PI * 2); c.fill(); } });
      const top = p.kp.filter((_, i) => ok(i)).sort((u, v) => u[1] - v[1])[0];
      if (top && p.label) {
        c.font = "bold 14px Segoe UI, sans-serif";
        c.lineWidth = 3;
        c.strokeStyle = "#000";
        c.strokeText(p.label, top[0] - 20, top[1] - 12);
        c.fillText(p.label, top[0] - 20, top[1] - 12);
      }
    }
    c.globalAlpha = 1;
  }

  function drawZone(c, w, h) {
    const zone = info.meta && info.meta.zona;
    if (!zone || zone.length < 3) return;
    c.strokeStyle = "#ffd43b";
    c.lineWidth = 2;
    c.setLineDash([8, 6]);
    c.beginPath();
    zone.forEach(([x, y], i) => (i ? c.lineTo(x * w, y * h) : c.moveTo(x * w, y * h)));
    c.closePath();
    c.stroke();
    c.setLineDash([]);
  }

  function paint(c, w, h) {
    const img = images[index];
    if (img && img.complete && img.naturalWidth) c.drawImage(img, 0, 0, w, h);
    if (info.versi === 2 && $("insp-skeleton").checked) {
      drawZone(c, w, h);
      const frame = info.overlay[index];
      if (frame) drawPeople(c, frame.orang, $("insp-highlight").checked);
    }
  }

  function draw() {
    if (!info || !images.length) return;
    const { w, h } = size();
    const box = canvas.parentElement.getBoundingClientRect();
    const scale = Math.min(box.width / w, (box.height || box.width * h / w) / h) || 1;
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale * view.s, 0, 0, scale * view.s, -view.x * scale * view.s, -view.y * scale * view.s);
    paint(ctx, w, h);
    $("insp-pos").value = index;
    const t = info.waktu[index] || 0;
    $("insp-time").textContent = `${t.toFixed(1)} / ${(info.waktu[info.waktu.length - 1] || 0).toFixed(1)} dtk · frame ${index + 1}/${images.length}`;
  }

  // ---- memutar -----------------------------------------------------------------------------
  function seek(i) {
    index = Math.max(0, Math.min(images.length - 1, i));
    clock = info.waktu[index] || 0;
    draw();
  }

  function setPlaying(on) {
    playing = on && images.length > 0;
    $("insp-play").textContent = playing ? "❚❚ Jeda" : "▶ Putar";
    lastTs = null;
    clearTimeout(playTimer);            // satu putaran saja, walau Putar ditekan berkali-kali
    if (playing) tick();
  }

  // setTimeout, bukan requestAnimationFrame: rAF berhenti di jendela yang tidak sedang digambar ulang.
  function tick() {
    if (!playing) return;
    const ts = performance.now();
    if (lastTs !== null) clock += ((ts - lastTs) / 1000) * speed;
    lastTs = ts;
    const end = info.waktu[info.waktu.length - 1] || 0;
    if (clock >= end) { index = images.length - 1; draw(); setPlaying(false); return; }
    let i = index;
    while (i + 1 < info.waktu.length && info.waktu[i + 1] <= clock) i++;
    if (i !== index) { index = i; draw(); }
    playTimer = setTimeout(tick, 20);
  }

  function markers() {
    const box = $("insp-marks");
    box.textContent = "";
    const put = (pos, cls, title) => {
      const m = document.createElement("span");
      m.className = "insp-mark " + cls;
      m.style.left = (pos * 100).toFixed(2) + "%";
      m.title = title;
      box.append(m);
    };
    const eventT = info.meta ? info.meta.kejadian_t : null;
    const ambang = typeof info.ambang_kekerasan === "number" ? info.ambang_kekerasan : AMBANG_BAWAAN;
    const marks = HITUNG.markerPositions(info.waktu, eventT, info.overlay, ambang);
    marks.skor.forEach((pos) => put(pos, "skor", "Penilai di atas ambang"));
    if (marks.kejadian !== null) put(marks.kejadian, "kejadian", "Detik kejadian");
  }

  function clearStage() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    $("insp-marks").textContent = "";
    $("insp-time").textContent = "";
    $("insp-involved").textContent = "";
    $("insp-lain").textContent = "";
    $("insp-lain-wrap").hidden = true;
  }

  // Kontrol hanya aktif bila klip yang tampil sudah siap, supaya tanda, catatan, dan unduhan tidak pernah jatuh ke
  // kejadian lain yang klipnya sedang dimuat.
  function setReady(ready) {
    document.querySelectorAll("#insp .insp-laptop button, #insp .insp-laptop textarea, #insp-play, #insp-prev, " +
                              "#insp-next, #insp-start, #insp-reset, #insp-pos").forEach((el) => { el.disabled = !ready; });
  }

  // ---- memuat klip -------------------------------------------------------------------------
  async function load(it, startAt) {
    const mine = ++token;
    setPlaying(false);
    item = it;
    info = null;
    images = [];
    view = { s: 1, x: 0, y: 0 };
    $("insp").hidden = false;
    setReady(false);
    clearStage();
    renderMarks();                                    // tanda dan catatan milik kejadian yang baru dipilih
    say("Memuat klip…");
    let response;
    try {
      response = await fetch(`/api/riwayat/${encodeURIComponent(it.id)}/info`, { cache: "no-store" });
    } catch (e) {
      if (mine === token) say("Klip belum bisa dimuat. Coba lagi.");
      return;
    }
    const result = await response.json().catch(() => ({}));
    if (mine !== token) return;
    if (!response.ok || !result.ok) { say(result.error || "Klip belum bisa dimuat."); return; }
    info = result;
    item = result.kejadian || it;
    // Daftar Riwayat dan judulnya mengikuti klip yang sedang dibuka (termasuk lewat "Kamera lain").
    document.dispatchEvent(new CustomEvent("sahabat-insp-item", { detail: item }));
    if (startAt !== undefined && typeof startAt === "object") startAt = clipTimeOf(startAt.wall);
    images = new Array(info.jumlah);
    let done = 0;
    let next = 0;
    await new Promise((resolve) => {
      const start = () => {
        if (mine !== token) { resolve(); return; }
        if (next >= info.jumlah) { if (done >= info.jumlah) resolve(); return; }
        const i = next++;
        const img = new Image();
        img.onload = img.onerror = () => {
          done++;
          if (mine === token) say(done < info.jumlah ? `Memuat klip… ${done}/${info.jumlah}` : "");
          if (i === 0 && mine === token) draw();
          start();
        };
        img.src = `/api/riwayat/${encodeURIComponent(item.id)}/frame/${i}`;
        images[i] = img;
      };
      for (let k = 0; k < Math.min(KONKUREN, info.jumlah); k++) start();
      if (!info.jumlah) resolve();
    });
    if (mine !== token) return;
    $("insp-pos").max = Math.max(0, images.length - 1);
    const v1 = info.versi !== 2;
    $("insp-skeleton").disabled = v1;
    $("insp-highlight").disabled = v1;
    $("insp-v1").hidden = !v1;
    $("insp-involved").textContent = involvementText(involved());
    renderOthers();
    renderMarks();
    markers();
    setReady(true);
    if (startAt !== undefined) {
      seek(HITUNG.frameAt(info.waktu, startAt));
    } else {
      seek(0);
      setPlaying(true);
    }
  }

  // Waktu dinding (detik epoch) <-> detik di dalam klip ini: kejadian terjadi pada item.waktu, yaitu detik
  // meta.kejadian_t di klip. Klip lama tanpa meta memakai batas "sebelum" (before_s) yang dipakai saat merekam.
  function eventOffset() {
    const t = info.meta && typeof info.meta.kejadian_t === "number" ? info.meta.kejadian_t : null;
    return t !== null ? t : (window.SahabatRiwayatSebelum || 0);
  }

  function wallTimeOf(clipT) { return (item.waktu || 0) - eventOffset() + clipT; }

  function clipTimeOf(wall) { return Math.max(0, wall - (item.waktu || 0) + eventOffset()); }

  function renderOthers() {
    const box = $("insp-lain");
    box.textContent = "";
    $("insp-lain-wrap").hidden = !(info.lain && info.lain.length);
    for (const other of info.lain || []) {
      const b = document.createElement("button");
      b.type = "button";
      const sign = other.selisih_s >= 0 ? "+" : "−";
      b.textContent = `${other.kamera_nama || "Kamera lain"} (${sign}${Math.abs(other.selisih_s).toFixed(1)} dtk)`;
      b.addEventListener("click", () => {
        // Detik yang sama di dunia nyata, bukan detik yang sama di klip: klip lain bisa punya awal berbeda.
        load({ id: other.id }, { wall: wallTimeOf(info.waktu[index] || 0) });
      });
      box.append(b);
    }
  }

  // ---- tindak lanjut -----------------------------------------------------------------------
  function renderMarks() {
    document.querySelectorAll("[data-tanda]").forEach((b) => {
      b.classList.toggle("active", !!item && item.tanda === b.dataset.tanda);
    });
    $("insp-catatan").value = (item && item.catatan) || "";
  }

  async function sendMark(body) {
    if (!item) return;
    try {
      const r = await fetch(`/api/riwayat/${encodeURIComponent(item.id)}/tindak`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const result = await r.json();
      if (!r.ok || !result.ok) throw new Error(result.error || "Belum tersimpan.");
      item = result.kejadian;
      renderMarks();
      say(item.tanda ? `Tersimpan: ${TANDA[item.tanda]}.` : "Tersimpan.");
      document.dispatchEvent(new CustomEvent("sahabat-riwayat-ubah"));
    } catch (error) {
      say(error.message);
    }
  }

  // ---- tangkapan gambar --------------------------------------------------------------------
  function snapshot() {
    if (!info || !images[index]) return;
    const { w, h } = size();
    const band = 30;
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h + band;
    const c = off.getContext("2d");
    c.fillStyle = "#000";
    c.fillRect(0, 0, w, h + band);
    paint(c, w, h);
    const when = new Date(item.waktu * 1000).toLocaleString("id-ID");
    c.fillStyle = "#e8edf3";
    c.font = "15px Segoe UI, sans-serif";
    c.fillText(`${item.kamera_nama || "Kamera"} · ${when} · frame ${index + 1}`, 8, h + 21);
    off.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `SAHABAT-${item.id}-frame${index + 1}.png`;
      document.body.append(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    }, "image/png");
  }

  // ---- zoom dan geser ----------------------------------------------------------------------
  const pointers = new Map();
  let pinch = null;

  function zoomAt(factor, cx, cy) {
    const { w } = size();
    const scale = canvas.width / w;
    const s2 = Math.max(1, Math.min(6, view.s * factor));
    const px = view.x + cx / (scale * view.s);
    const py = view.y + cy / (scale * view.s);
    view.s = s2;
    view.x = px - cx / (scale * s2);
    view.y = py - cy / (scale * s2);
    clamp();
    draw();
  }

  function clamp() {
    const { w, h } = size();
    view.x = Math.max(0, Math.min(w - w / view.s, view.x));
    view.y = Math.max(0, Math.min(h - h / view.s, view.y));
  }

  canvas.addEventListener("wheel", (ev) => {
    if (!info) return;
    ev.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomAt(ev.deltaY < 0 ? 1.2 : 1 / 1.2, ev.clientX - r.left, ev.clientY - r.top);
  }, { passive: false });

  canvas.addEventListener("pointerdown", (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!pointers.has(ev.pointerId) || !info) return;
    const prev = pointers.get(ev.pointerId);
    pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    const { w } = size();
    const scale = canvas.width / w;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch) {
        const r = canvas.getBoundingClientRect();
        zoomAt(dist / pinch, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
      }
      pinch = dist;
      return;
    }
    view.x -= (ev.clientX - prev.x) / (scale * view.s);
    view.y -= (ev.clientY - prev.y) / (scale * view.s);
    clamp();
    draw();
  });
  const release = (ev) => { pointers.delete(ev.pointerId); if (pointers.size < 2) pinch = null; };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  // ---- tombol ------------------------------------------------------------------------------
  $("insp-play").addEventListener("click", () => {
    if (!info) return;
    if (!playing && index >= images.length - 1) seek(0);
    setPlaying(!playing);
  });
  $("insp-prev").addEventListener("click", () => { setPlaying(false); seek(index - 1); });
  $("insp-next").addEventListener("click", () => { setPlaying(false); seek(index + 1); });
  $("insp-start").addEventListener("click", () => { setPlaying(false); seek(0); });
  $("insp-speed").addEventListener("change", () => { speed = Number($("insp-speed").value) || 1; });
  $("insp-pos").addEventListener("input", () => { setPlaying(false); seek(Number($("insp-pos").value)); });
  $("insp-reset").addEventListener("click", () => { view = { s: 1, x: 0, y: 0 }; draw(); });
  $("insp-skeleton").addEventListener("change", draw);
  $("insp-highlight").addEventListener("change", draw);
  $("insp-snap").addEventListener("click", snapshot);

  // Unduh MP4 lewat fetch: galat server ditampilkan sebagai pesan, tidak tersimpan sebagai berkas; tombol dikunci
  // selama MP4 dibuat supaya tidak dibuat dua kali.
  $("insp-mp4").addEventListener("click", async () => {
    if (!item || !info) return;
    const mine = item.id;
    const button = $("insp-mp4");
    button.disabled = true;
    say("Membuat MP4… (bisa beberapa detik)");
    try {
      const r = await fetch(`/api/riwayat/${encodeURIComponent(mine)}/unduh`, { cache: "no-store" });
      if (!r.ok) {
        const result = await r.json().catch(() => ({}));
        throw new Error(result.error || "MP4 belum bisa dibuat.");
      }
      const blob = await r.blob();
      const match = /filename="([^"]+)"/.exec(r.headers.get("content-disposition") || "");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = match ? match[1] : `SAHABAT-${mine}.mp4`;
      document.body.append(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
      say("MP4 siap diunduh.");
    } catch (error) {
      say(error.message);
    } finally {
      button.disabled = false;
    }
  });
  document.querySelectorAll("[data-tanda]").forEach((b) => b.addEventListener("click", () => {
    sendMark({ tanda: item && item.tanda === b.dataset.tanda ? "" : b.dataset.tanda });
  }));
  $("insp-catatan-save").addEventListener("click", () => sendMark({ catatan: $("insp-catatan").value }));
  window.addEventListener("resize", () => { if (info) draw(); });

  document.addEventListener("keydown", (ev) => {
    if (!$("history-dialog").open || $("insp").hidden || !info) return;
    // Tombol, pilihan, penggeser, dan kotak teks memakai tombol keyboardnya sendiri (spasi menekan tombol yang
    // sedang terfokus, panah menggeser penggeser); pemutar tidak boleh ikut menanggapinya.
    const tag = ev.target && ev.target.tagName;
    if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
    // Tombol terfokus (mis. sesudah menekan Jeda): spasi/Enter sudah menekan tombol itu sendiri, panah tetap
    // melangkah per frame.
    if (tag === "BUTTON" && (ev.code === "Space" || ev.code === "Enter")) return;
    if (ev.code === "ArrowLeft") { ev.preventDefault(); setPlaying(false); seek(index - 1); }
    else if (ev.code === "ArrowRight") { ev.preventDefault(); setPlaying(false); seek(index + 1); }
    else if (ev.code === "Space") { ev.preventDefault(); if (!ev.repeat) $("insp-play").click(); }
    else if (ev.code === "Home") { ev.preventDefault(); if (!ev.repeat) { setPlaying(false); seek(0); } }
  });

  window.SahabatInspeksi = {
    open(it) { load(it); },
    close() {
      token++;
      setPlaying(false);
      clearTimeout(playTimer);
      info = null;
      images = [];
      item = null;
      $("insp").hidden = true;
      say("");
    },
  };
})();
