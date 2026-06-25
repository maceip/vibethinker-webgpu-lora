/*
 *   ,;
 *  \@@#\:          :/.        .:;;:
 * _@@@@@@#+\|/!;;!-@@@--;    ,@@@@@;
 * .!_*@@@@@@@@@@@@@@@@@@@;   |@@@@@\
 *     .:!|+@@@@@##@@@@@@@#!  -@@@@@#,
 *         .\@@@*;,\@@@@@@@@+,*@@@@@@+.
 *     :*#@@@@@@@@@@@@@@-+@@@@@@@\@@@@-.
 *     .#@@@@@#@@@@#*@@@+ /@@@@@@;\@@@@+.
 *      ;\/:,  -@@@@;|@@@\ ,+@@@@!.+@@@@*:
 *             ,@@@@#*@@@@@#+__!.  ,*@@@@@/
 *              \##+_@@@@@@@@,      ,+@@@_:
 *                   ;;,,..,:         !;.
 */

export async function initWebGPUDevice({ log = () => {} } = {}) {
  log('requesting WebGPU device…');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter (use a WebGPU-capable browser)');

  /*
   * TECHNIQUE: Hard requirement on immediate_address_space + subgroups
   *   We do not support fallback paths. This keeps the kernel set small and
   *   guarantees the fast immediate + subgroup reduction paths are available.
   */
  if (!navigator.gpu.wgslLanguageFeatures?.has('immediate_address_space'))
    throw new Error('WGSL immediate_address_space is not available (upgrade to Chrome 149+)');
  if (!adapter.features.has('subgroups'))
    throw new Error(
      'GPU lacks the required "subgroups" feature. The current fast WGSL kernels require subgroups and no fallback kernel set is bundled.',
    );

  const hasSubgroupId = !!navigator.gpu.wgslLanguageFeatures?.has('subgroup_id');
  const hasLinearIndexing = !!navigator.gpu.wgslLanguageFeatures?.has('linear_indexing');
  const hasF16 = adapter.features.has('shader-f16');
  const hasTimestamp = adapter.features.has('timestamp-query');

  const reqFeatures = ['subgroups'];
  if (adapter.features.has('shader-f16')) reqFeatures.push('shader-f16');
  if (hasTimestamp) reqFeatures.push('timestamp-query');

  const dev = await adapter.requestDevice({
    requiredFeatures: reqFeatures,
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });
  dev.addEventListener?.('uncapturederror', (e) => console.error('GPUERR', e.error.message));
  log(`WebGPU ready. maxBuffer=${(Number(adapter.limits.maxBufferSize) / 1e9).toFixed(2)}GB` +
      ` subgroupId=${hasSubgroupId} linearIdx=${hasLinearIndexing} f16=${hasF16} tsQuery=${hasTimestamp}`);
  return dev;
}
