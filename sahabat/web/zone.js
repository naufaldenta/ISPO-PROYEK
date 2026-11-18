// Editor zona pantau: klik sudut-sudut area di atas gambar kamera. Titik disimpan ternormalisasi (0..1)
// terhadap gambar, jadi tetap pas walau ukuran jendela atau resolusi kamera berubah.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const MAX_POINTS = 24;
  const editor = $("zone-editor");
  const svg = $("zone-svg");

  let saved = null;        // zona yang berlaku sekarang (dari server), null = seluruh gambar
  let points = [];         // titik yang sedang digambar
  let box = null;          // letak gambar kamera di layar

  document.addEventListener("sahabat-zona", (ev) => {
    saved = ev.detail;
    $("zone").hidden = false;
    $("zone").textContent = saved ? "Zona: aktif" : "Zona";
  });

  // Gambar kamera memakai object-fit: contain, jadi letaknya dihitung dari ukuran aslinya.
  function imageBox() {
    const img = $("video");
    const r = img.getBoundingClientRect();
    const nw = img.naturalWidth || 640;
    const nh = img.naturalHeight || 480;
    const scale = Math.min(r.width / nw, r.height / nh);
    const w = nw * scale;
    const h = nh * scale;
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, w, h };
  }

  function place() {
    box = imageBox();
    Object.assign(svg.style, { left: box.left + "px", top: box.top + "px", width: box.w + "px", height: box.h + "px" });
    svg.setAttribute("viewBox", `0 0 ${box.w} ${box.h}`);
    draw();
  }

  function el(name, attrs) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  function draw() {
    svg.textContent = "";
    const px = points.map(([x, y]) => [x * box.w, y * box.h]);
    if (px.length >= 3) {
      const d = "M" + px.map((p) => p.join(",")).join("L") + "Z";
      svg.append(el("path", { d: `M0,0H${box.w}V${box.h}H0Z ${d}`, "fill-rule": "evenodd", class: "zone-shade" }));
    }
    if (px.length >= 2) {
      svg.append(el(px.length >= 3 ? "polygon" : "polyline",
                    { points: px.map((p) => p.join(",")).join(" "), class: "zone-line" }));
    }
    px.forEach(([x, y], i) => svg.append(el("circle", { cx: x, cy: y, r: i === 0 ? 9 : 7, class: i === 0 ? "zone-dot first" : "zone-dot" })));
    $("zone-count").textContent = points.length < 3
      ? `${points.length} titik (butuh minimal 3)`
      : `${points.length} titik`;
    $("zone-save").disabled = points.length < 3;
    $("zone-undo").disabled = points.length === 0;
  }

  function open() {
    points = saved ? saved.map((p) => p.slice()) : [];
    $("zone-message").textContent = "";
    editor.hidden = false;
    place();
  }

  function close() {
    editor.hidden = true;
  }

  function send(titik) {
    $("zone-message").textContent = "Menyimpan…";
    fetch("/api/zona", { method: "POST", headers: { "Content-Type": "application/json" },
                         body: JSON.stringify({ titik }) })
      .then((r) => r.json())
      .then((r) => {
        if (!r.ok) { $("zone-message").textContent = r.error || "Zona belum tersimpan."; return; }
        saved = r.titik;
        $("zone").textContent = saved ? "Zona: aktif" : "Zona";
        close();
      })
      .catch(() => { $("zone-message").textContent = "Zona belum tersimpan. Coba lagi."; });
  }

  svg.addEventListener("click", (ev) => {
    if (points.length >= MAX_POINTS) {
      $("zone-message").textContent = `Paling banyak ${MAX_POINTS} titik.`;
      return;
    }
    const r = svg.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
    points.push([Math.round(x * 10000) / 10000, Math.round(y * 10000) / 10000]);
    $("zone-message").textContent = "";
    draw();
  });

  $("zone").addEventListener("click", open);
  $("zone-undo").addEventListener("click", () => { points.pop(); draw(); });
  $("zone-whole").addEventListener("click", () => send(null));
  $("zone-cancel").addEventListener("click", close);
  $("zone-save").addEventListener("click", () => send(points));
  window.addEventListener("resize", () => { if (!editor.hidden) place(); });
  $("video").addEventListener("load", () => { if (!editor.hidden && (!box || !box.w)) place(); });
  document.addEventListener("keydown", (ev) => {
    if (editor.hidden) return;
    if (ev.code === "Escape") { ev.preventDefault(); close(); }
    else if ((ev.ctrlKey || ev.metaKey) && ev.code === "KeyZ") { ev.preventDefault(); points.pop(); draw(); }
  });
})();
