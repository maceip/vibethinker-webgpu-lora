export class AdapterRegistry {
  constructor() {
    this.adapters = { none: null };
  }

  add(name, modules) {
    this.adapters[name] = { modules };
    return this.adapters[name];
  }

  get(name) {
    return this.adapters[name] || null;
  }

  applyToRuntime(name, rt) {
    const adapter = this.get(name);
    if (adapter) rt.setLora(adapter);
    else rt.clearLora();
    return adapter;
  }
}
