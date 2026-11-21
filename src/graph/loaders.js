// Parses a SNAP-style edge list ("u v" per line, '#' comments).
// Node ids are remapped to a dense 0..n-1 range.
export function parseEdgeList(text, name = 'edge-list') {
  const ids = new Map();
  const list = [];
  const seen = new Set();
  const idOf = (raw) => {
    let id = ids.get(raw);
    if (id === undefined) {
      id = ids.size;
      ids.set(raw, id);
    }
    return id;
  };
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t || t[0] === '#' || t[0] === '%') continue;
    const parts = t.split(/[\s,]+/);
    if (parts.length < 2) continue;
    const u = idOf(parts[0]);
    const v = idOf(parts[1]);
    if (u === v) continue;
    const key = u < v ? u * 4294967296 + v : v * 4294967296 + u;
    if (seen.has(key)) continue;
    seen.add(key);
    list.push(u, v);
  }
  return { name, n: ids.size, edges: Uint32Array.from(list) };
}

export async function loadEdgeListURL(url, name) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return parseEdgeList(await res.text(), name);
}
