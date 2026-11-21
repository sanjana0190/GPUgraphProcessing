// WebGPU renderer: draws edges and nodes straight from the layout's storage
// buffers. No vertex buffers and no CPU copies of positions.
//
//   edges: draw(2m)      line-list,      vertex shader indexes edgeIdx -> positions
//   nodes: draw(6, n)    triangle-list,  instanced quads, SDF circles in the fragment shader
import { createShader } from './device.js';
import renderCommonWGSL from './shaders/render_common.wgsl?raw';
import nodesWGSL from './shaders/nodes.wgsl?raw';
import edgesWGSL from './shaders/edges.wgsl?raw';
import { createBufferWithData } from './buffers.js';

const SAMPLES = 4; // MSAA for smooth lines
const CAMERA_BYTES = 32;

export class GPURenderer {
  constructor(gpu, canvas) {
    this.gpu = gpu;
    this.device = gpu.device;
    this.canvas = canvas;
    this.context = canvas.getContext('webgpu');
    this.context.configure({ device: this.device, format: gpu.format, alphaMode: 'opaque' });
    this.cameraData = new Float32Array(CAMERA_BYTES / 4);
    this.cameraBuffer = this.device.createBuffer({
      label: 'camera',
      size: CAMERA_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  async init() {
    const { device } = this;
    const V = GPUShaderStage.VERTEX;
    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'render-bgl',
      entries: [
        { binding: 0, visibility: V | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        // read-only here: vertex shaders may not write storage buffers
        { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: V, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: V, buffer: { type: 'read-only-storage' } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    const blend = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    const target = { format: this.gpu.format, blend };

    const edgesModule = await createShader(device, renderCommonWGSL + '\n' + edgesWGSL, 'edges');
    const nodesModule = await createShader(device, renderCommonWGSL + '\n' + nodesWGSL, 'nodes');
    [this.edgePipeline, this.nodePipeline] = await Promise.all([
      device.createRenderPipelineAsync({
        label: 'edges',
        layout,
        vertex: { module: edgesModule, entryPoint: 'vs' },
        fragment: { module: edgesModule, entryPoint: 'fs', targets: [target] },
        primitive: { topology: 'line-list' },
        multisample: { count: SAMPLES },
      }),
      device.createRenderPipelineAsync({
        label: 'nodes',
        layout,
        vertex: { module: nodesModule, entryPoint: 'vs' },
        fragment: { module: nodesModule, entryPoint: 'fs', targets: [target] },
        primitive: { topology: 'triangle-list' },
        multisample: { count: SAMPLES },
      }),
    ]);
    return this;
  }

  /** Point the renderer at a layout's buffers. */
  setGraph(layout, colorValues) {
    this.n = layout.n;
    this.m = layout.m;
    this.colorBuffer?.destroy();
    this.colorBuffer = createBufferWithData(this.device, colorValues, GPUBufferUsage.STORAGE, 'node-colors');
    this.bindGroup = this.device.createBindGroup({
      label: 'render-bg',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: layout.posBuffer } },
        { binding: 2, resource: { buffer: layout.edgesBuffer } },
        { binding: 3, resource: { buffer: this.colorBuffer } },
      ],
    });
  }

  ensureMSAATarget() {
    const { width, height } = this.canvas;
    if (!this.msaaTexture || this.msaaTexture.width !== width || this.msaaTexture.height !== height) {
      this.msaaTexture?.destroy();
      this.msaaTexture = this.device.createTexture({
        label: 'msaa-color',
        size: [width, height],
        sampleCount: SAMPLES,
        format: this.gpu.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  }

  writeCamera(camera, style) {
    const c = this.cameraData;
    c[0] = camera.centerX;
    c[1] = camera.centerY;
    c[2] = camera.scaleX;
    c[3] = camera.scaleY;
    c[4] = (style.nodeRadiusPx * 2) / this.canvas.width;
    c[5] = (style.nodeRadiusPx * 2) / this.canvas.height;
    c[6] = style.edgeAlpha;
    this.device.queue.writeBuffer(this.cameraBuffer, 0, c);
  }

  /** Record the render pass. Edges first so nodes draw on top. */
  encode(encoder, timestampWrites) {
    this.ensureMSAATarget();
    const pass = encoder.beginRenderPass({
      label: 'render',
      colorAttachments: [
        {
          view: this.msaaTexture.createView(),
          resolveTarget: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.043, g: 0.055, b: 0.078, a: 1 },
          loadOp: 'clear',
          storeOp: 'discard', // only the resolved image is needed
        },
      ],
      timestampWrites,
    });
    pass.setBindGroup(0, this.bindGroup);
    if (this.m > 0) {
      pass.setPipeline(this.edgePipeline);
      pass.draw(this.m * 2);
    }
    pass.setPipeline(this.nodePipeline);
    pass.draw(6, this.n); // 6 vertices per quad, n instances
    pass.end();
  }

  destroy() {
    this.colorBuffer?.destroy();
    this.msaaTexture?.destroy();
    this.cameraBuffer?.destroy();
  }
}
