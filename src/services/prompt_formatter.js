// Faithful Qwen2.5 ChatML fallback (matches VibeThinker-3B's chat_template for
// non-tool messages, including the default system prompt injection).
export function chatML(messages) {
  let s = (messages[0]?.role === 'system') ? '' : '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n';
  for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  return s + '<|im_start|>assistant\n';
}

export function formatMessages(tokenizer, messages) {
  try { return tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true }); }
  catch { return chatML(messages); }
}
