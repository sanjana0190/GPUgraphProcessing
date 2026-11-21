// 2D camera shared by both renderers. world -> clip: (p - center) * scale
export class Camera {
  constructor() {
    this.centerX = 0;
    this.centerY = 0;
    this.halfWidth = 100; // world units visible from center to the right edge
    this.aspect = 1; // height / width
    this.autoFit = true;
  }
  get scaleX() {
    return 1 / this.halfWidth;
  }
  get scaleY() {
    return 1 / (this.halfWidth * this.aspect);
  }
  setViewport(w, h) {
    this.aspect = h / Math.max(1, w);
  }
  /** Fit a bounding box. `smooth` in (0, 1]: 1 = jump, smaller = ease. */
  fit(b, smooth = 1) {
    if (!b || !Number.isFinite(b.minX) || !Number.isFinite(b.maxX)) return;
    const cx = (b.minX + b.maxX) / 2;
    const cy = (b.minY + b.maxY) / 2;
    const hw = Math.max((b.maxX - b.minX) / 2, (b.maxY - b.minY) / 2 / this.aspect, 1) * 1.08;
    this.centerX += (cx - this.centerX) * smooth;
    this.centerY += (cy - this.centerY) * smooth;
    this.halfWidth += (hw - this.halfWidth) * smooth;
  }
  /** Convert canvas pixel coords to world coords. */
  toWorld(px, py, w, h) {
    const x = (px / w) * 2 - 1;
    const y = 1 - (py / h) * 2;
    return [x / this.scaleX + this.centerX, y / this.scaleY + this.centerY];
  }
  zoomAt(px, py, w, h, factor) {
    const [wx, wy] = this.toWorld(px, py, w, h);
    this.halfWidth *= factor;
    const [nx, ny] = this.toWorld(px, py, w, h);
    this.centerX += wx - nx;
    this.centerY += wy - ny;
  }
  panPixels(dx, dy, w, h) {
    this.centerX -= (dx / w) * 2 * this.halfWidth;
    this.centerY += (dy / h) * 2 * this.halfWidth * this.aspect;
  }
}
