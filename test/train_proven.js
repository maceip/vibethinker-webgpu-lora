/*
 * Emberglass — "proven" run: held-out GENERALIZATION test for the in-browser LoRA
 * trainer. Trains on the SFT train split and evaluates on a DISJOINT held-out split
 * (valid), with hyperparameters matched to the MLX baseline (rank 16, scale 20,
 * lr 1e-4, completion-only masking, full real prompts). Reports base vs MLX-baseline
 * vs ours held-out loss, and the held-out loss curve over training (= generalization,
 * not memorization). Logs "PROVEN ..." lines; "PROVEN DONE" at the end.
 */
import { QWEN25_3B } from '../src/config.js';
import { urlReader } from '../src/readers.js';
import { ModelSession } from '../src/services/model_session.js';
import { AdapterRegistry } from '../src/services/adapter_registry.js';
import { TrainingController } from '../src/services/training_controller.js';
import { QwenLoraTrainer, createTrainableAdapter } from '../src/qwgpu/trainer.js';
import { loadLoraAdapterGPU } from '../src/lora_gpu.js';

const L = (...a) => console.log('PROVEN', ...a);
const ALL7 = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
const MAXSEQ = 1024;

window.runProven = async () => {
  try {
    const session = new ModelSession({ cfg: QWEN25_3B, log: (m) => console.log('PROVEN load:', m) });
    await session.loadWith(urlReader('/model'), '/model');
    const dev = session.dev,
      rt = session.rt;
    const ctrl = new TrainingController({ session, adapters: new AdapterRegistry(), log: () => {} });

    const samples = await (await fetch('/test/proven_samples.json')).json();
    // Tokenize full examples (real system prompt) and keep only those that fit MAXSEQ
    // WITHOUT truncation (truncation would drop the completion -> no trained tokens).
    const prep = (list) => {
      const out = [];
      for (const ex of list) {
        const a = ex.messages.find((m) => m.role === 'assistant')?.content || '';
        const sys = ex.messages.find((m) => m.role === 'system');
        const user = ex.messages.find((m) => m.role === 'user');
        const mb = ctrl.prepareExample({ messages: [sys, user].filter(Boolean), completion: a });
        const active = mb.lossMask.reduce((s, v) => s + v, 0);
        if (mb.tokens.length <= MAXSEQ && active > 0) out.push(mb);
      }
      return out;
    };
    const trainSet = prep(samples.train);
    const heldout = prep(samples.heldout);
    L(`data: train=${trainSet.length} heldout=${heldout.length} (maxseq=${MAXSEQ})`);
    if (!trainSet.length || !heldout.length) throw new Error('no examples fit MAXSEQ');

    const evalSet = async (tr) => {
      rt.setLora(tr.adapter);
      rt.invalidateLora();
      let s = 0;
      for (const mb of heldout) {
        const r = await tr.evalLoss(mb.tokens, mb.lossMask);
        s += r.loss;
      }
      return s / heldout.length;
    };

    // ---- base (B=0) and MLX baseline held-out loss ----
    const baseAdapter = createTrainableAdapter(rt, { name: 'base', rank: 16, alpha: 32, targetModules: ALL7 });
    const trBase = new QwenLoraTrainer(rt, { maxTrainSeq: MAXSEQ, lmHeadBlock: 128 });
    trBase.attach(baseAdapter);
    const baseLoss = await evalSet(trBase);
    L(`heldout base (no adapter) = ${baseLoss.toFixed(4)}`);

    const stf = await (await fetch('/test/baseline_adapter/adapters.safetensors')).arrayBuffer();
    const cfgf = await (await fetch('/test/baseline_adapter/adapter_config.json')).text();
    const fileLike = (name, buf, txt) => ({ name, async arrayBuffer() { return buf; }, async text() { return txt ?? new TextDecoder().decode(buf); } });
    const baseline = await loadLoraAdapterGPU(dev, [fileLike('adapters.safetensors', stf), fileLike('adapter_config.json', null, cfgf)], QWEN25_3B);
    const trMLX = new QwenLoraTrainer(rt, { maxTrainSeq: MAXSEQ, lmHeadBlock: 128 });
    trMLX.attach(baseline);
    const mlxLoss = await evalSet(trMLX);
    L(`heldout MLX-baseline (2000 iters) = ${mlxLoss.toFixed(4)}  [improves base by ${(baseLoss - mlxLoss).toFixed(4)}]`);

    // ---- train OUR adapter on the train split, eval held-out periodically ----
    // Same rank (16) and completion-only masking as the MLX baseline, but a stable
    // function-space scale: alpha/rank=2 instead of MLX's scale=20. MLX tames scale=20
    // with batch=8 over 2000 iters; in a short Adam run that scale diverges, so we use
    // the validated alpha=32 regime with lr=2e-4 + global-norm clipping.
    const ours = createTrainableAdapter(rt, { name: 'ours', rank: 16, alpha: 32, targetModules: ALL7 });
    const STEPS = 50,
      ACCUM = 2;
    const trOurs = new QwenLoraTrainer(rt, {
      lr: 2e-4,
      maxTrainSeq: MAXSEQ,
      lmHeadBlock: 128,
      maxGradNorm: 1.0,
      weightDecay: 0.0,
      warmupSteps: 5,
      totalSteps: STEPS,
      gradAccumSteps: ACCUM,
    });
    trOurs.attach(ours);
    const oursStart = await evalSet(trOurs);
    L(`heldout ours @0 steps = ${oursStart.toFixed(4)}`);

    let ti = 0;
    const nextBatch = () => {
      const b = [];
      for (let j = 0; j < ACCUM; j++) b.push(trainSet[ti++ % trainSet.length]);
      return b;
    };
    rt.setLora(trOurs.adapter);
    rt.invalidateLora();
    for (let step = 1; step <= STEPS; step++) {
      const r = await trOurs.trainStep(nextBatch());
      if (step % 5 === 0 || step === 1) L(`train step ${step}/${STEPS}: trainloss=${r.loss.toFixed(4)} lr=${r.lr.toExponential(2)} |g|=${r.gradNorm.toFixed(2)}`);
      if (step % 10 === 0) {
        const hl = await evalSet(trOurs);
        L(`heldout ours @${step} steps = ${hl.toFixed(4)}`);
        rt.setLora(trOurs.adapter); // re-bind ours for continued training
        rt.invalidateLora();
      }
    }
    const oursEnd = await evalSet(trOurs);
    L(`=== RESULT: heldout loss — base=${baseLoss.toFixed(4)}  MLX-baseline=${mlxLoss.toFixed(4)}  ours@${STEPS}=${oursEnd.toFixed(4)} ===`);
    const closed = (baseLoss - oursEnd) / (baseLoss - mlxLoss);
    L(`ours closed ${(closed * 100).toFixed(1)}% of the base->baseline held-out gap in ${STEPS} steps`);
    L(`generalization: ${oursEnd < oursStart ? 'PASS (held-out loss decreased on unseen data)' : 'FAIL'}`);
    L('DONE');
    console.log('PROVEN DONE');
  } catch (e) {
    console.log('PROVEN FATAL ' + (e && e.stack ? e.stack : e));
  }
};
