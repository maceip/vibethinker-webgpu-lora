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

export class GenerationController {
  constructor({ session, adapters, systemPrompt, log = () => {} }) {
    this.session = session;
    this.adapters = adapters;
    this.systemPrompt = systemPrompt;
    this.log = log;
  }

  /*
   * TECHNIQUE: Streaming text output with TextNode append (O(n) not O(n^2))
   *   Uses a single Text node and appends characters instead of setting
   *   .textContent repeatedly. Avoids quadratic cost during long generations.
   */
  async runTriage({ adapterName, report, outputNode, maxTemperature = 0.0 }) {
    const rt = this.session.rt;
    if (!rt) return;
    outputNode.textContent = '';
    const node = document.createTextNode('');
    outputNode.appendChild(node); // O(n) streaming, not O(n^2)
    this.adapters.applyToRuntime(adapterName, rt);
    this.log(`generating (adapter=${adapterName})…`);
    const messages = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: report },
    ];
    const t0 = performance.now();
    let n = 0;
    for await (const delta of this.session.generate(messages, { maxTokens: rt.maxCtx, temperature: maxTemperature })) {
      node.appendData(delta);
      n++;
    }
    const dt = (performance.now() - t0) / 1000;
    this.log(`done: ${n} tokens in ${dt.toFixed(1)}s (${(n / dt).toFixed(1)} tok/s) adapter=${adapterName}`);
  }
}
