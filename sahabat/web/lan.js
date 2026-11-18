// Panel "HP / laptop lain": menyalakan akses dari jaringan lokal, kode QR, alamat, dan kode akses.
// Hanya tampil di laptop SAHABAT sendiri (keadaan "jaringan" hanya dikirim server ke laptop itu).
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const dialog = $("lan-dialog");
  let on = false;
  let busy = false;

  function label(state) {
    on = !!state.hidup;
    const viewers = state.penonton || 0;
    $("lan").hidden = false;
    $("lan").dataset.on = on ? "1" : "0";
    $("lan").textContent = on ? `HP / laptop lain: ${viewers} tersambung` : "HP / laptop lain";
  }

  function show(state) {
    label(state);
    $("lan-toggle").textContent = "Jaringan lokal: " + (on ? "hidup" : "mati");
    $("lan-toggle").dataset.on = on ? "1" : "0";
    $("lan-on").hidden = !on;
    if (on) {
      $("lan-qr").src = "/api/jaringan/qr.png?t=" + Date.now();
      const list = $("lan-urls");
      list.textContent = "";
      (state.alamat || []).forEach((url) => {
        const li = document.createElement("li");
        li.textContent = url;
        list.append(li);
      });
      if (!(state.alamat || []).length) {
        const li = document.createElement("li");
        li.textContent = "Laptop belum tersambung ke WiFi / jaringan mana pun.";
        list.append(li);
      }
      $("lan-code").textContent = state.kode;
      $("lan-viewers").textContent = state.penonton
        ? `${state.penonton} layar lain sedang tersambung.` : "Belum ada layar lain yang tersambung.";
    }
    $("lan-message").textContent = state.galat || "";
  }

  async function call(url, options) {
    if (busy) return;
    busy = true;
    try {
      const response = await fetch(url, options);
      const result = await response.json();
      if (!response.ok && !result.galat) throw new Error(result.error || "Gagal.");
      show(result);
    } catch (error) {
      $("lan-message").textContent = error.message;
    } finally {
      busy = false;
    }
  }

  document.addEventListener("sahabat-jaringan", (ev) => {
    label(ev.detail);
    if (dialog.open && ev.detail.penonton !== undefined && on) {
      $("lan-viewers").textContent = ev.detail.penonton
        ? `${ev.detail.penonton} layar lain sedang tersambung.` : "Belum ada layar lain yang tersambung.";
    }
  });

  $("lan").addEventListener("click", () => {
    dialog.showModal();
    call("/api/jaringan", { cache: "no-store" });
  });
  $("lan-close").addEventListener("click", () => dialog.close());
  $("lan-toggle").addEventListener("click", () => {
    $("lan-message").textContent = on ? "Mematikan…" : "Menyalakan…";
    call("/api/jaringan", { method: "POST", headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ hidup: !on }) });
  });
  $("lan-new-code").addEventListener("click", () => call("/api/jaringan/kode", { method: "POST" }));
})();
