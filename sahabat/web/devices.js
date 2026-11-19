// Pemilih input terpisah dari tampilan alarm dan logika deteksi.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const dialog = $("device-dialog");
  let catalog = null;
  let busy = false;

  function setBusy(value) {
    busy = value;
    $("device-refresh").disabled = value;
    $("device-apply").disabled = value || !catalog;
    $("camera-select").disabled = value || !catalog;
    $("microphone-select").disabled = value || !catalog || !catalog.microphone_available;
  }

  function fill(select, entries, selected, automatic, videos = [], ipcams = []) {
    select.replaceChildren();
    const options = [{ id: null, name: automatic }, ...entries];
    if (![...options, ...videos, ...ipcams].some((item) => item.id === selected)) {
      options.push({ id: selected, name: "Pilihan tersimpan (tidak terdeteksi)" });
    }
    options.forEach((item) => select.add(new Option(item.name, JSON.stringify(item.id))));
    if (ipcams.length) {
      const group = document.createElement("optgroup");
      group.label = "Kamera CCTV (atur di panel Kamera)";
      ipcams.forEach((item) => group.append(new Option("CCTV: " + item.name, JSON.stringify(item.id))));
      select.append(group);
    }
    if (videos.length) {
      const group = document.createElement("optgroup");
      group.label = "Video dari folder video (diputar berulang)";
      videos.forEach((item) => group.append(new Option("Video: " + item.name, JSON.stringify(item.id))));
      select.append(group);
    }
    select.value = JSON.stringify(selected);
  }

  function status(state) {
    if (!state || !dialog.open) return;
    const camera = catalog && catalog.cameras.find((d) => d.id === state.camera);
    if (state.camera_is_video) {
      $("camera-status").textContent = "Video " + state.camera.slice("video:".length) + ": " + (state.camera_connected
        ? "diputar berulang" : (state.camera_error || "membuka video…"));
    } else if (state.camera_is_cctv) {
      $("camera-status").textContent = "CCTV " + (state.camera_name || "(tidak ada di daftar)") + ": " +
        (state.camera_connected ? "tersambung" : (state.camera_error || "menghubungkan…"));
    } else {
      const name = camera ? camera.name : (state.camera === null ? "Otomatis" : "Pilihan tersimpan");
      $("camera-status").textContent = "Kamera " + name + ": " + (state.camera_connected
        ? "tersambung" : (state.camera_error || "menghubungkan / tidak terhubung"));
    }
    $("microphone-status").textContent = state.microphone_status + (state.microphone_name ? " — " + state.microphone_name : "");
  }

  async function scan(preserve = false) {
    if (busy) return;
    const previous = preserve && catalog ? {
      camera: JSON.parse($("camera-select").value), microphone: JSON.parse($("microphone-select").value)
    } : null;
    setBusy(true);
    $("device-message").textContent = "Mencari kamera dan mikrofon…";
    try {
      const response = await fetch("/api/perangkat", { cache: "no-store" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Daftar perangkat belum tersedia.");
      catalog = result;
      fill($("camera-select"), result.cameras, previous ? previous.camera : result.camera, "Otomatis — kamera pertama tersedia",
           result.videos || [], result.ipcams || []);
      fill($("microphone-select"), result.microphones, previous ? previous.microphone : result.microphone, "Otomatis — mikrofon bawaan Windows");
      status(result);
      const videoCount = (result.videos || []).length;
      $("device-message").textContent = result.errors.length ? result.errors.join(" ") :
        `${result.cameras.length} kamera, ${result.microphones.length} mikrofon, dan ${videoCount} video ditemukan.`;
    } catch (error) {
      $("device-message").textContent = error.message;
    } finally {
      setBusy(false);
    }
  }

  function open() {
    if (!dialog.open) dialog.showModal();
    showVolume();
    scan();
  }

  // ---- video cadangan ----
  $("video-add").addEventListener("click", () => $("video-file").click());
  $("video-file").addEventListener("change", async () => {
    const file = $("video-file").files[0];
    $("video-file").value = "";
    if (!file || busy) return;
    setBusy(true);
    $("device-message").textContent = `Menyalin ${file.name} ke folder video…`;
    try {
      const response = await fetch("/api/video?nama=" + encodeURIComponent(file.name), { method: "POST", body: file });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || "Video belum tersalin.");
      setBusy(false);
      await scan(true);
      $("camera-select").value = JSON.stringify(result.id);
      $("device-message").textContent = `${result.nama} siap. Tekan Terapkan untuk memutarnya sebagai pengganti kamera.`;
    } catch (error) {
      $("device-message").textContent = error.message;
    } finally {
      setBusy(false);
    }
  });
  $("video-folder").addEventListener("click", async () => {
    try {
      const response = await fetch("/api/video/folder", { method: "POST" });
      const result = await response.json();
      $("device-message").textContent = result.ok
        ? "Folder video dibuka di Explorer. Setelah menyalin video ke sana, tekan Cari ulang."
        : (result.error || "Folder video tidak bisa dibuka.");
    } catch (error) {
      $("device-message").textContent = "Folder video tidak bisa dibuka.";
    }
  });

  // ---- volume alarm (disimpan app.js di browser laptop ini) ----
  function showVolume() {
    let v = 1;
    try {
      const saved = parseFloat(localStorage.getItem("sahabat-volume"));
      if (Number.isFinite(saved)) v = Math.min(1, Math.max(0, saved));
    } catch (e) { /* pakai bawaan */ }
    $("volume").value = Math.round(v * 100);
    volumeLabel();
  }
  function volumeLabel() {
    const v = Number($("volume").value);
    $("volume-value").textContent = v === 0 ? "0% (senyap)" : v + "%";
  }
  $("volume").addEventListener("input", () => {
    volumeLabel();
    document.dispatchEvent(new CustomEvent("sahabat-volume", { detail: Number($("volume").value) / 100 }));
  });
  $("volume-test").addEventListener("click", () => document.dispatchEvent(new CustomEvent("sahabat-volume-test")));
  $("devices").addEventListener("click", open);
  $("start-devices").addEventListener("click", open);
  $("device-close").addEventListener("click", () => dialog.close());
  $("device-refresh").addEventListener("click", () => scan(true));
  document.addEventListener("sahabat-devices", (event) => status(event.detail));

  $("device-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy || !catalog) return;
    const choice = { camera: JSON.parse($("camera-select").value) };
    if (catalog.microphone_available) choice.microphone = JSON.parse($("microphone-select").value);
    setBusy(true);
    $("device-message").textContent = "Menerapkan pilihan…";
    try {
      const response = await fetch("/api/perangkat", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(choice)
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || "Pilihan belum diterapkan.");
      Object.assign(catalog, result);
      status(result);
      $("device-message").textContent = result.camera_is_video
        ? "Video diputar sebagai pengganti kamera untuk sesi ini. Tutup panel untuk melihatnya."
        : "Pilihan disimpan. Status sambungan diperbarui otomatis; tutup panel untuk melihat kamera.";
    } catch (error) {
      $("device-message").textContent = error.message;
    } finally {
      setBusy(false);
    }
  });
})();
