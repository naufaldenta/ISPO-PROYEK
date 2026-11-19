// Perhitungan murni pemutar inspeksi (tanpa DOM), dipakai inspeksi.js dan diuji lewat Node.
(function (root) {
  "use strict";

  // Frame terakhir yang waktunya <= t (detik sejak frame pertama).
  function frameAt(waktu, t) {
    let i = 0;
    while (i + 1 < waktu.length && waktu[i + 1] <= t + 1e-9) i++;
    return i;
  }

  // Posisi penanda pada garis waktu (0..1). Garis waktu bergerak per frame, jadi penanda juga dihitung per frame:
  // frame bisa tidak merata (kamera siaga ±2 fps lalu penuh 10 fps), dan posisi detik akan menunjuk frame yang salah.
  function markerPositions(waktu, kejadianT, overlay, ambang) {
    const n = Math.max(1, waktu.length - 1);
    const skor = [];
    (overlay || []).forEach((f, i) => {
      if (f && f.skor && (f.skor.jatuh || f.skor.kekerasan >= ambang)) skor.push(i / n);
    });
    let kejadian = null;
    if (kejadianT !== null && kejadianT !== undefined && waktu.length) {
      let i = waktu.findIndex((t) => t >= kejadianT - 1e-6);
      if (i < 0) i = waktu.length - 1;
      kejadian = waktu.length > 1 ? i / n : 0;
    }
    return { kejadian, skor };
  }

  const api = { frameAt, markerPositions };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SahabatInspeksiHitung = api;
})(typeof window !== "undefined" ? window : this);
