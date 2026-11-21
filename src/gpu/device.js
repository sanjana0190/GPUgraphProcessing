// WebGPU bootstrap: adapter -> device, with the optional features we use.
//
// On a MacBook this goes Chrome/Safari -> WebGPU -> Metal -> Apple GPU.
// The same code runs on Vulkan (Linux/Android) and D3D12 (Windows).
export async function initWebGPU({ onLost } = {}) {
  if (!('gpu' in navigator)) {
    throw new Error(
      'WebGPU is not available in this browser. Use Chrome/Edge 113+, Safari 26+, or Firefox 145+ (Apple Silicon).',
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found (GPU blocklisted or disabled?).');

  const requiredFeatures = [];
  const hasTimestamps = adapter.features.has('timestamp-query');
  if (hasTimestamps) requiredFeatures.push('timestamp-query');

  // Ask for the adapter's real limits instead of the conservative defaults,
  // so large graphs (big storage buffers) fit.
  const requiredLimits = {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
  };

  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });

  device.lost.then((info) => {
    console.error(`WebGPU device lost (${info.reason}): ${info.message}`);
    if (onLost) onLost(info);
  });
  // Validation errors that are not caught by an error scope end up here.
  device.addEventListener('uncapturederror', (e) => {
    console.error('WebGPU uncaptured error:', e.error.message);
  });

  const info = adapter.info || {};
  return {
    adapter,
    device,
    hasTimestamps,
    adapterLabel: [info.vendor, info.architecture, info.description].filter(Boolean).join(' / ') || 'unknown GPU',
    format: navigator.gpu.getPreferredCanvasFormat(),
  };
}

/**
 * Create a shader module and surface compiler errors with line numbers.
 * There is no printf in WGSL, so getting good compile errors matters.
 */
export async function createShader(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  for (const msg of info.messages) {
    const line = code.split('\n')[msg.lineNum - 1] ?? '';
    const log = msg.type === 'error' ? console.error : console.warn;
    log(`[${label}] ${msg.type} at ${msg.lineNum}:${msg.linePos}: ${msg.message}\n  ${line}`);
  }
  if (errors.length) throw new Error(`Shader "${label}" failed to compile: ${errors[0].message}`);
  return module;
}
