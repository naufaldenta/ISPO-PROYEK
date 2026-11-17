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

  function fill(select, entries, selected, automatic) {
    select.replaceChildren();
    const options = [{ id: null, name: automatic }, ...entries];
    if (!options.some((item) => item.id === selected)) {
      options.push({ id: selected, name: "Pilihan tersimpan (tidak terdeteksi)" });
    }
    options.forEach((item) => select.add(new Option(item.name, JSON.stringify(item.id))));
    select.value = JSON.stringify(selected);
  }

  function status(state) {
    if (!state || !dialog.open) return;
    const camera = catalog && catalog.cameras.find((d) => d.id === state.camera);
    const name = camera ? camera.name : (state.camera === null ? "Otomatis" : "Pilihan tersimpan");
    $("camera-status").textContent = "Kamera " + name + ": " + (state.camera_connected
      ? "tersambung" : (state.camera_error || "menghubungkan / tidak terhubung"));
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
      fill($("camera-select"), result.cameras, previous ? previous.camera : result.camera, "Otomatis — kamera pertama tersedia");
      fill($("microphone-select"), result.microphones, previous ? previous.microphone : result.microphone, "Otomatis — mikrofon bawaan Windows");
      status(result);
      $("device-message").textContent = result.errors.length ? result.errors.join(" ") :
        `${result.cameras.length} kamera dan ${result.microphones.length} mikrofon ditemukan.`;
    } catch (error) {
      $("device-message").textContent = error.message;
    } finally {
      setBusy(false);
    }
  }

  function open() {
    if (!dialog.open) dialog.showModal();
    scan();
  }
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
      $("device-message").textContent = "Pilihan disimpan. Status sambungan diperbarui otomatis; tutup panel untuk melihat kamera.";
    } catch (error) {
      $("device-message").textContent = error.message;
    } finally {
      setBusy(false);
    }
  });
})();
