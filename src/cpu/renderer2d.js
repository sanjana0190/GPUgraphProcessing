// Canvas 2D renderer for the CPU engine. It draws from the CPU position array
// every frame; this is the "classic" approach the GPU renderer replaces.
import { paletteRGB } from '../graph/colors.js';

const BUCKETS = 6;

export class Canvas2DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  setGraph(graph, colorValues) {
    this.edges = graph.edges;
    this.n = graph.n;
    // Group nodes by color bucket so we change fillStyle only a few times per frame.
    this.buckets = Array.from({ length: BUCKETS }, () => []);
    for (let i = 0; i < graph.n; i++) {
      const b = Math.min(BUCKETS - 1, Math.floor(colorValues[i] * BUCKETS));
      this.buckets[b].push(i);
    }
    this.bucketColors = this.buckets.map((_, b) => {
      const [r, g, bl] = paletteRGB((b + 0.5) / BUCKETS);
      return `rgb(${r},${g},${bl})`;
    });
  }

  draw(pos, camera, style) {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#0b0e14';
    ctx.fillRect(0, 0, w, h);

    // world -> pixels, matching the GPU camera math
    const sx = (camera.scaleX * w) / 2;
    const sy = (camera.scaleY * h) / 2;
    const cx = camera.centerX;
    const cy = camera.centerY;
    const X = (x) => (x - cx) * sx + w / 2;
    const Y = (y) => h / 2 - (y - cy) * sy;

    const e = this.edges;
    ctx.strokeStyle = `rgba(150,170,210,${style.edgeAlpha})`;
    ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1) * 0.6;
    ctx.beginPath();
    for (let i = 0; i < e.length; i += 2) {
      const a = e[i], b = e[i + 1];
      ctx.moveTo(X(pos[2 * a]), Y(pos[2 * a + 1]));
      ctx.lineTo(X(pos[2 * b]), Y(pos[2 * b + 1]));
    }
    ctx.stroke();

    const r = style.nodeRadiusPx;
    for (let b = 0; b < BUCKETS; b++) {
      ctx.fillStyle = this.bucketColors[b];
      ctx.beginPath();
      for (const i of this.buckets[b]) {
        const x = X(pos[2 * i]);
        const y = Y(pos[2 * i + 1]);
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }
}
