/*
 * Emberglass — end-to-end in-browser LoRA training validation ("ladder").
 * Loads the REAL VibeThinker-3B (int4) once, then runs four rungs on the public
 * trainer path and logs "TRAIN ..." / "RUNGn ..." lines; "TRAIN DONE" at the end.
 *   Rung 1: forward+loss on real held-out examples (emits exact tokens for an
 *           external torch CE cross-check).
 *   Rung 2: finite-difference gradient check on a real LoRA weight.
 *   Rung 3: overfit a tiny example (loss -> ~0).
 *   Rung 4: baseline MLX adapter loss vs a freshly-trained adapter, same inputs.
 */
import { QWEN25_3B } from '../src/config.js';
import { urlReader } from '../src/readers.js';
import { ModelSession } from '../src/services/model_session.js';
import { AdapterRegistry } from '../src/services/adapter_registry.js';
import { TrainingController } from '../src/services/training_controller.js';
import { QwenLoraTrainer, createTrainableAdapter } from '../src/qwgpu/trainer.js';
import { loadLoraAdapterGPU } from '../src/lora_gpu.js';

const L = (...a) => console.log('TRAIN', ...a);
const ALL7 = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];

async function readF32(dev, buf, floats) {
  const rb = dev.createBuffer({ size: floats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, floats * 4);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}

window.runLadder = async () => {
  try {
    const session = new ModelSession({ cfg: QWEN25_3B, log: (m) => console.log('TRAIN load:', m) });
    await session.loadWith(urlReader('/model'), '/model');
    const dev = session.dev,
      rt = session.rt;
    const ctrl = new TrainingController({ session, adapters: new AdapterRegistry(), log: () => {} });

    const samples = await (await fetch('/test/train_samples.json')).json();
    // Build short, completion-bearing examples (trim the ~700-token system prompt so
    // the trained span fits a small window; torch validates the EXACT emitted tokens).
    const shorten = (ex) => {
      const user = (ex.messages.find((m) => m.role === 'user')?.content || '').slice(0, 600);
      const asst = (ex.messages.find((m) => m.role === 'assistant')?.content || '').slice(0, 280);
      return {
        messages: [
          { role: 'system', content: 'You are a bug bounty triage analyst. Assign one disposition.' },
          { role: 'user', content: user },
        ],
        completion: asst,
      };
    };
    const evalEx = samples.test.map(shorten);

    // ---------- Rung 1: forward + loss on real held-out examples ----------
    L('=== RUNG 1: forward+loss parity (real model) ===');
    const adapter1 = createTrainableAdapter(rt, { name: 'r1', rank: 16, alpha: 32, targetModules: ALL7 });
    const tr1 = new QwenLoraTrainer(rt, { maxTrainSeq: 768, lmHeadBlock: 128 });
    tr1.attach(adapter1); // B=0 => zero delta => base-model loss
    const cases = [];
    for (let i = 0; i < evalEx.length; i++) {
      const mb = ctrl.prepareExample(evalEx[i]);
      if (mb.tokens.length > 768) {
        mb.tokens = mb.tokens.slice(0, 768);
        mb.lossMask = mb.lossMask.slice(0, 768);
      }
      tr1.zeroGrads();
      const r = await tr1.microStep(mb.tokens, mb.lossMask);
      L(`RUNG1 case ${i}: loss=${r.loss.toFixed(4)} active=${r.numActive} T=${mb.tokens.length}`);
      cases.push({ id: i, tokens: mb.tokens, mask: mb.lossMask });
    }
    console.log('RUNG1_CASES ' + JSON.stringify({ cases }));

    // ---------- Rung 2: finite-difference gradient check ----------
    L('=== RUNG 2: finite-difference gradient check ===');
    const synth = ctrl.prepareExample({
      messages: [
        { role: 'system', content: 'Triage.' },
        { role: 'user', content: 'Is reflected XSS with a working PoC in scope?' },
      ],
      completion: 'valid_impactful',
    });
    L(`RUNG2 synth T=${synth.tokens.length}`);
    const adapter2 = createTrainableAdapter(rt, { name: 'r2', rank: 16, alpha: 32, targetModules: ALL7 });
    const tr2 = new QwenLoraTrainer(rt, { maxTrainSeq: 256, lmHeadBlock: 128, maxGradNorm: 0 });
    tr2.attach(adapter2);

    const fdCheck = async (which) => {
      // populate grads at current weights
      tr2.zeroGrads();
      await tr2.microStep(synth.tokens, synth.lossMask);
      const key = tr2.trainedKeys[0];
      const st = tr2.state[key];
      const isB = which === 'B';
      const gbuf = isB ? st.dB : st.dA;
      const wbuf = isB ? st.mod.B : st.mod.A;
      const n = isB ? st.rank * st.N : st.rank * st.K;
      const grad = await readF32(dev, gbuf, n);
      let idx = 0,
        best = -1;
      for (let i = 0; i < n; i++) if (Math.abs(grad[i]) > best) { best = Math.abs(grad[i]); idx = i; }
      const analytic = grad[idx];
      const W = await readF32(dev, wbuf, n);
      const eps = 1e-2;
      const lossAt = async (v) => {
        const a = W.slice();
        a[idx] = v;
        dev.queue.writeBuffer(wbuf, 0, a);
        rt.invalidateLora();
        tr2.zeroGrads();
        const r = await tr2.microStep(synth.tokens, synth.lossMask);
        return r.loss;
      };
      // Central diff at eps and 2*eps, then Richardson-extrapolate to kill the
      // O(eps^2) truncation term: g ≈ (4*D(eps) - D(2eps)) / 3.
      const cdiff = async (h) => {
        const lp = await lossAt(W[idx] + h);
        const lm = await lossAt(W[idx] - h);
        return (lp - lm) / (2 * h);
      };
      const d1 = await cdiff(eps);
      const d2 = await cdiff(2 * eps);
      const numeric = (4 * d1 - d2) / 3;
      dev.queue.writeBuffer(wbuf, 0, W); // restore
      rt.invalidateLora();
      const relerr = Math.abs(analytic - numeric) / (Math.abs(numeric) + 1e-6);
      const ok = relerr < 0.03 || Math.abs(analytic - numeric) < 1e-3;
      L(`RUNG2 d${which}[${idx}] (${key}): analytic=${analytic.toExponential(3)} numeric=${numeric.toExponential(3)} relerr=${relerr.toExponential(2)} ${ok ? 'PASS' : 'FAIL'}`);
      return ok;
    };
    await fdCheck('B'); // B=0 at init -> dB is the active gradient
    // take one step so A acquires gradient (dA depends on B!=0), then check dA
    await tr2.optimizerStep();
    await fdCheck('A');

    // ---------- Rung 3: overfit a tiny example ----------
    L('=== RUNG 3: overfit smoke test ===');
    const adapter3 = createTrainableAdapter(rt, { name: 'r3', rank: 16, alpha: 32, targetModules: ALL7 });
    const tr3 = new QwenLoraTrainer(rt, { lr: 1e-3, maxTrainSeq: 256, lmHeadBlock: 128, maxGradNorm: 1.0, warmupSteps: 5, totalSteps: 80 });
    tr3.attach(adapter3);
    let first3 = null,
      best3 = Infinity;
    for (let step = 1; step <= 80; step++) {
      const r = await tr3.trainStep([{ tokens: synth.tokens, lossMask: synth.lossMask }]);
      if (step === 1) first3 = r.loss;
      best3 = Math.min(best3, r.loss);
      if (step % 10 === 0 || step === 1) L(`RUNG3 step ${step}: loss=${r.loss.toFixed(4)} lr=${r.lr.toExponential(2)} |g|=${r.gradNorm.toFixed(3)}`);
    }
    L(`RUNG3 overfit: first=${first3.toFixed(4)} best=${best3.toFixed(4)} drop=${(first3 - best3).toFixed(4)} ${best3 < first3 * 0.25 ? 'PASS' : 'WEAK'}`);

    // ---------- Rung 4: baseline MLX adapter vs freshly trained adapter ----------
    L('=== RUNG 4: baseline (MLX) vs ours, same inputs ===');
    const stf = await (await fetch('/test/baseline_adapter/adapters.safetensors')).arrayBuffer();
    const cfgf = await (await fetch('/test/baseline_adapter/adapter_config.json')).text();
    const fileLike = (name, buf, txt) => ({ name, async arrayBuffer() { return buf; }, async text() { return txt ?? new TextDecoder().decode(buf); } });
    const baseline = await loadLoraAdapterGPU(dev, [fileLike('adapters.safetensors', stf), fileLike('adapter_config.json', null, cfgf)], QWEN25_3B);
    L(`RUNG4 baseline adapter modules=${Object.keys(baseline.modules).length} rank=${Object.values(baseline.modules)[0].rank} scale=${Object.values(baseline.modules)[0].scale}`);
    const trB = new QwenLoraTrainer(rt, { maxTrainSeq: 768, lmHeadBlock: 128 });
    trB.attach(baseline);
    const evalLoss = async (tr) => {
      rt.setLora(tr.adapter); // all trainers share one rt — bind the right adapter first
      rt.invalidateLora();
      let s = 0,
        n = 0;
      for (const ex of evalEx) {
        const mb = ctrl.prepareExample(ex);
        if (mb.tokens.length > 768) { mb.tokens = mb.tokens.slice(0, 768); mb.lossMask = mb.lossMask.slice(0, 768); }
        tr.zeroGrads();
        const r = await tr.microStep(mb.tokens, mb.lossMask);
        s += r.loss;
        n++;
      }
      return s / n;
    };
    const baseLoss = await evalLoss(tr1); // fresh adapter (B=0) = base model
    const blLoss = await evalLoss(trB); // MLX baseline adapter
    L(`RUNG4 eval loss — base=${baseLoss.toFixed(4)}  MLX-baseline=${blLoss.toFixed(4)}  (baseline improves base by ${(baseLoss - blLoss).toFixed(4)})`);

    // train a fresh adapter a few steps on the train split, re-eval
    const adapterT = createTrainableAdapter(rt, { name: 'ours', rank: 16, alpha: 32, targetModules: ALL7 });
    const trT = new QwenLoraTrainer(rt, { lr: 1e-4, maxTrainSeq: 768, lmHeadBlock: 128, maxGradNorm: 1.0, warmupSteps: 2 });
    trT.attach(adapterT);
    const trainEx = samples.train.map(shorten);
    const oursBefore = await evalLoss(trT);
    for (let step = 1; step <= 12; step++) {
      const ex = trainEx[(step - 1) % trainEx.length];
      const mb = ctrl.prepareExample(ex);
      if (mb.tokens.length > 768) { mb.tokens = mb.tokens.slice(0, 768); mb.lossMask = mb.lossMask.slice(0, 768); }
      const r = await trT.trainStep([{ tokens: mb.tokens, lossMask: mb.lossMask }]);
      if (step % 4 === 0 || step === 1) L(`RUNG4 train step ${step}: loss=${r.loss.toFixed(4)} |g|=${r.gradNorm.toFixed(3)}`);
    }
    const oursAfter = await evalLoss(trT);
    L(`RUNG4 ours eval — before=${oursBefore.toFixed(4)} after-12-steps=${oursAfter.toFixed(4)} drop=${(oursBefore - oursAfter).toFixed(4)}`);

    L('DONE');
    console.log('TRAIN DONE');
  } catch (e) {
    console.log('TRAIN FATAL ' + (e && e.stack ? e.stack : e));
  }
};
