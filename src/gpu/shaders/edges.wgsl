// Edge rendering as a line-list: draw(2 * m) vertices, no vertex buffers.
// Vertex v belongs to edge v/2, endpoint v%2. The shader looks up the node
// index in the edge list, then the node's position in the storage buffer.

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let node = edgeIdx[vi];   // edge list is flat [u0, v0, u1, v1, ...]
  return vec4<f32>(toClip(positions[node]), 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(0.59, 0.67, 0.82, cam.edgeAlpha);
}
