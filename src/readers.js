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

// tf-free range readers used by the WebGPU runtime loader.
// A reader contract: { range(path, start, end) -> ArrayBuffer, text(path) -> string }

/*
 * TECHNIQUE: Minimal reader abstraction
 *   Three implementations (url, hf, file) all obey the same tiny interface.
 *   Lets the pure-WebGPU loader (runtime + safetensors_loader) work
 *   identically whether loading from same-origin /model, HF, or a dropped folder.
 */
export function urlReader(baseUrl, headers = {}) {
  const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
  return {
    async range(path, start, end) {
      const r = await fetch(base + path, {
        headers: { ...headers, Range: `bytes=${start}-${end - 1}` },
      });
      if (!r.ok && r.status !== 206) {
        throw new Error(`range ${path} ${start}-${end}: ${r.status}`);
      }
      return await r.arrayBuffer();
    },
    async text(path) {
      const r = await fetch(base + path, { headers });
      if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
      return await r.text();
    },
  };
}

/** Reader over a Hugging Face repo (resolve endpoint, CORS + Range). */
export function hfReader(repo, token = '', rev = 'main') {
  return urlReader(
    `https://huggingface.co/${repo}/resolve/${rev}`,
    token ? { Authorization: `Bearer ${token}` } : {},
  );
}

/** Range reader over user-provided File objects (webkitdirectory / drag-drop). */
export function fileReader(fileMap) {
  const pick = (path) => fileMap[path] || fileMap[path.split('/').pop()];
  return {
    async range(path, start, end) {
      const f = pick(path);
      if (!f) throw new Error(`file not provided: ${path}`);
      return await f.slice(start, end).arrayBuffer();
    },
    async text(path) {
      const f = pick(path);
      if (!f) throw new Error(`file not provided: ${path}`);
      return await f.text();
    },
  };
}
