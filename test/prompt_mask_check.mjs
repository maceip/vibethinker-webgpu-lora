/*
 * Emberglass — Qwen2.5 WebGPU runtime (custom kernels, int4, runtime LoRA)
 * CPU unit checks for ChatML formatting + completion-only shifted-label masking.
 * No GPU / no model needed. Prints "PROMPT ..." lines; "PROMPT DONE" when finished.
 */

// training_controller -> trainer.js references WebGPU globals (GPUBufferUsage) at
// module top-level. Provide a numeric shim so this CPU-only test can import it in
// Node; values match the WebGPU spec flag bits but are never used for real GPU work.
if (typeof globalThis.GPUBufferUsage === 'undefined') {
  globalThis.GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  };
}
if (typeof globalThis.GPUShaderStage === 'undefined') {
  globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
}

const { chatML, formatMessages } = await import('../src/services/prompt_formatter.js');
const { TrainingController } = await import('../src/services/training_controller.js');

const IM_END = 151645;
let PASS = 0,
  FAIL = 0;
function check(name, cond, extra = '') {
  console.log(`PROMPT ${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : '  ' + extra}`);
  cond ? PASS++ : FAIL++;
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ---- chatML(): default system injection + assistant generation prompt ----
{
  const out = chatML([{ role: 'user', content: 'hi' }]);
  const want =
    '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' +
    '<|im_start|>user\nhi<|im_end|>\n' +
    '<|im_start|>assistant\n';
  check('chatML injects default system', out === want, JSON.stringify(out));
  check('chatML ends with generation prompt', out.endsWith('<|im_start|>assistant\n'));
}
{
  const out = chatML([
    { role: 'system', content: 'S' },
    { role: 'user', content: 'hi' },
  ]);
  const want =
    '<|im_start|>system\nS<|im_end|>\n' +
    '<|im_start|>user\nhi<|im_end|>\n' +
    '<|im_start|>assistant\n';
  check('chatML respects explicit system', out === want, JSON.stringify(out));
}

// ---- formatMessages(): uses tokenizer template, falls back on throw ----
{
  const good = { apply_chat_template: () => 'TEMPLATED<|im_start|>assistant\n' };
  check('formatMessages uses apply_chat_template', formatMessages(good, [{ role: 'user', content: 'x' }]) === 'TEMPLATED<|im_start|>assistant\n');
  const broken = {
    apply_chat_template: () => {
      throw new Error('no template');
    },
  };
  const fb = formatMessages(broken, [{ role: 'user', content: 'hi' }]);
  check('formatMessages falls back to chatML', fb === chatML([{ role: 'user', content: 'hi' }]));
  check('fallback ends with add_generation_prompt', fb.endsWith('<|im_start|>assistant\n'));
}

// ---- prepareExample(): completion-only shifted-label masking ----
// Fake tokenizer: id-per-whitespace-token (ids irrelevant; lengths are what matter).
const fakeTk = {
  apply_chat_template: (m) => chatML(m),
  encode: (str) =>
    String(str)
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .map((_, i) => i + 1),
};
const ctrl = new TrainingController({ session: { rt: null, tokenizer: fakeTk } });

{
  // prompt -> 3 ids, completion -> 2 ids, + IM_END  => T = 6
  const { tokens, lossMask } = ctrl.prepareExample({ prompt: 'a b c', completion: 'x y' });
  check('tokens length = prompt+comp+eos', tokens.length === 6, `len=${tokens.length}`);
  check('last token is <|im_end|>', tokens[tokens.length - 1] === IM_END);
  // firstTrainPos = promptLen-1 = 2; train t in [2, T-2]=[2,4]; T-1 stays 0.
  check('completion-only mask', eq(lossMask, [0, 0, 1, 1, 1, 0]), JSON.stringify(lossMask));
  // mask[t]=1 means position t predicts tokens[t+1]; first trained label must be
  // the first completion token (index promptLen = 3), reached from position 2.
  const firstTrained = lossMask.indexOf(1);
  check('first trained label is first completion token', firstTrained === 2 && firstTrained + 1 === 3);
}
{
  const { lossMask } = ctrl.prepareExample({ prompt: 'a b c', completion: 'x y', trainPromptToo: true });
  check('trainPromptToo masks all but final', eq(lossMask, [1, 1, 1, 1, 1, 0]), JSON.stringify(lossMask));
}
{
  // messages path should route through formatMessages/apply_chat_template
  const { tokens, lossMask } = ctrl.prepareExample({ messages: [{ role: 'user', content: 'hello there' }], completion: 'ok' });
  check('messages path produces a trained span', lossMask.some((m) => m === 1) && tokens[tokens.length - 1] === IM_END);
  check('no position trains past sequence end', lossMask[lossMask.length - 1] === 0);
}

console.log(`PROMPT ${FAIL === 0 ? 'ALL PASS' : 'FAILED'} (${PASS}/${PASS + FAIL})`);
console.log('PROMPT DONE');
if (FAIL) process.exitCode = 1;
