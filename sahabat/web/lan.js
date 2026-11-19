// Panel "HP / laptop lain": menyalakan akses dari jaringan lokal, kode QR, alamat, dan kode akses.
// Hanya tampil di laptop SAHABAT sendiri (keadaan "jaringan" hanya dikirim server ke laptop itu).
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const dialog = $("lan-dialog");
  let on = false;
  let busy = false;
  let firewallCheck = 0;                     // nomor pemeriksaan terbaru; hasil pemeriksaan lama diabaikan

  const FIREWALL_TEXT = {
    belum: "<b>Windows Firewall masih menutup akses dari HP</b>, jadi HP hanya memuat terus. Tekan tombol di " +
      "bawah, lalu pilih <b>Ya</b> di jendela izin Windows. Cukup sekali di laptop ini.",
    diblokir: "<b>Windows Firewall memblokir SAHABAT</b> (mungkin dulu pernah dipilih Batal), jadi HP hanya " +
      "memuat terus. Tekan tombol di bawah, lalu pilih <b>Ya</b> di jendela izin Windows.",
  };

  function showFirewall(status) {
    const text = FIREWALL_TEXT[status];
    $("lan-firewall").hidden = !text || !on;
    if (text) $("lan-firewall-text").innerHTML = text;
  }

  async function checkFirewall() {
    const id = ++firewallCheck;
    try {
      const response = await fetch("/api/jaringan/firewall", { cache: "no-store" });
      const result = await response.json();
      if (id === firewallCheck) showFirewall(result.firewall);
      return result.firewall;
    } catch (error) {
      return "tidak-diketahui";
    }
  }

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
    if (on) checkFirewall(); else showFirewall("");
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
  $("lan-firewall-allow").addEventListener("click", async () => {
    const button = $("lan-firewall-allow");
    button.disabled = true;
    $("lan-message").textContent = "Menunggu izin Windows… pilih Ya di jendela yang muncul.";
    try {
      const result = await (await fetch("/api/jaringan/firewall", { method: "POST" })).json();
      if (!result.ok) throw new Error(result.error || "Gagal.");
      $("lan-message").textContent = "Memeriksa ulang firewall…";
      let status = "";
      for (let i = 0; i < 5 && status !== "izin"; i++) status = await checkFirewall();
      $("lan-message").textContent = status === "izin"
        ? "Firewall sudah mengizinkan. Muat ulang halaman di HP." : "Firewall belum berubah. Coba tekan lagi.";
    } catch (error) {
      $("lan-message").textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
  $("lan-new-code").addEventListener("click", () => call("/api/jaringan/kode", { method: "POST" }));
})();
