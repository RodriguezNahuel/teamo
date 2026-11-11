// Corre teamo.py (Python + numpy, via Pyodide) en un hilo aparte,
// asi la pagina no se traba mientras se calcula cada cuadro.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.5/full/pyodide.js");

let webFrame = null;

async function init() {
  postMessage({ type: "status", text: "cargando python…" });
  const py = await loadPyodide();
  postMessage({ type: "status", text: "cargando numpy…" });
  await py.loadPackage("numpy");
  const src = await (await fetch("teamo.py", { cache: "no-cache" })).text();
  py.FS.writeFile("/home/pyodide/teamo.py", src);
  const m = py.pyimport("teamo");
  webFrame = m.web_frame;
  postMessage({
    type: "ready",
    para: String(m.PARA), carta: String(m.CARTA), de: String(m.DE),
    firma: String(m.FIRMA_HUD), tLetter: Number(m.T_LETTER),
  });
}

const booting = init().catch((e) => postMessage({ type: "error", text: String(e) }));

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type !== "frame") return;
  await booting;
  if (!webFrame) return;
  const s = performance.now();
  const res = webFrame(m.t, m.C, m.R, m.aspect, m.top, m.bottom, m.quality);
  const out = res.toJs();
  res.destroy();
  const rgb = out[1] instanceof Uint8Array ? out[1] : new Uint8Array(out[1]);
  postMessage({
    type: "frame", id: m.id, C: m.C, R: m.R, chars: out[0], rgb,
    bloom: out[2], sway: out[3], letters: out[4], ms: performance.now() - s,
  }, [rgb.buffer]);
};
