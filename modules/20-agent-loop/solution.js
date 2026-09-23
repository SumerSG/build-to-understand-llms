// The agent loop harness (reference solution).
// An "agent" is a loop written in ordinary code: call the model, parse its intent, execute the tools it
// asked for, append the results to the message array, repeat until a stop condition fires.

// ---------- worked examples and constants (given to the learner) ----------

/** The wire format the scripted models in this lab emit. */
export const OPEN_TAG = '<tool_call>';
export const CLOSE_TAG = '</tool_call>';

/**
 * Per-message overhead in tokens: every message carries its role and framing tokens on the wire,
 * not just its text. OpenAI's token-counting cookbook uses 3 per message for current chat models
 * (4 for the earliest gpt-3.5-turbo); the exact number depends on the chat template, so treat 4 as
 * an order-of-magnitude constant.
 */
export const TOKENS_PER_MESSAGE = 4;

/** Format one call the way a model is asked to emit it. Used by the scripted models and the demo. */
export function toolCall(name, args = {}) {
  return `${OPEN_TAG}${JSON.stringify({ name, args })}${CLOSE_TAG}`;
}

/** Turn a handler's return value into the string that goes back to the model. */
export function resultToString(value) {
  if (typeof value === 'string') return value;
  if (value === undefined) return '';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * A model you can test a loop with: either a function of (messages, schemas), or a list of replies
 * returned in order. When a scripted list runs out the model replies with an empty string, which
 * contains no tool calls and so ends the loop with 'final' instead of spinning until maxTurns.
 */
export function mockModel(script) {
  if (typeof script === 'function') return async (messages, schemas) => script(messages, schemas);
  const replies = script.slice();
  let at = 0;
  return async () => (at < replies.length ? replies[at++] : '');
}

// ---------- step 1: the tool registry ----------

/** One property against its spec: type first, then an optional enum of allowed values. */
function checkType(name, spec, value) {
  const actual = Array.isArray(value) ? 'array' : typeof value;
  if (spec.type === 'integer') {
    if (!Number.isInteger(value)) return `argument "${name}" must be an integer, got ${JSON.stringify(value)}`;
  } else if (spec.type && spec.type !== actual) {
    return `argument "${name}" must be a ${spec.type}, got ${actual}`;
  }
  if (spec.enum && !spec.enum.includes(value)) {
    return `argument "${name}" must be one of ${spec.enum.map((v) => JSON.stringify(v)).join(', ')}`;
  }
  return null;
}

/** Check `args` against a minimal JSON-schema subset; returns an error string, or null when valid. */
export function validateArgs(schema, args) {
  if (!schema || schema.type !== 'object') return null; // no schema declared: nothing to check
  if (args === null || typeof args !== 'object' || Array.isArray(args)) return 'arguments must be an object';
  for (const name of schema.required ?? []) {
    if (args[name] === undefined) return `missing required argument "${name}"`;
  }
  const properties = schema.properties ?? {};
  for (const [name, value] of Object.entries(args)) {
    const spec = properties[name];
    if (!spec) continue; // unknown arguments are ignored rather than rejected
    const problem = checkType(name, spec, value);
    if (problem) return problem;
  }
  return null;
}

/** The tools an agent may call: their schemas for the model, their handlers for the harness. */
export class ToolRegistry {
  constructor() {
    this.tools = new Map();
    /** (name, args) => 'allow' | 'ask' | 'deny' (or a promise of one). null allows every call. */
    this.policy = null;
    /** (name, args) => true to approve an 'ask' decision (or a promise of true). null approves nothing. */
    this.confirm = null;
  }

  /** Add a tool. `parameters` is a minimal JSON schema: { type:'object', properties, required }. */
  register(name, { description = '', parameters = null, handler } = {}) {
    if (typeof handler !== 'function') throw new Error(`ToolRegistry.register: ${name} needs a handler function`);
    this.tools.set(name, { name, description, parameters, handler });
    return this;
  }

  /** The tool list as a model would be shown it (no handlers). */
  schemas() {
    return [...this.tools.values()].map(({ name, description, parameters }) => ({ name, description, parameters }));
  }

  has(name) {
    return this.tools.has(name);
  }

  /**
   * Run a tool and return its result as a string. Unknown tools, invalid arguments, denied or
   * unconfirmed calls and handler exceptions all come back as 'Error: ...' strings instead of
   * throwing: the model gets to read the mistake and try again, and one bad call cannot take the
   * whole run down.
   */
  async call(name, args = {}) {
    const tool = this.tools.get(name);
    if (!tool) {
      const known = [...this.tools.keys()].join(', ') || 'none';
      return `Error: unknown tool "${name}". Available tools: ${known}`;
    }
    const problem = validateArgs(tool.parameters, args);
    if (problem) return `Error: ${problem}`;
    const decision = await this.decide(name, args);
    if (decision !== 'allow') return decision;
    try {
      return resultToString(await tool.handler(args));
    } catch (err) {
      return `Error: ${err && err.message ? err.message : String(err)}`;
    }
  }

  /**
   * The permission check, run after validation and before the handler. Returns 'allow', or the
   * 'Error: ...' string to send back instead. It fails closed: a policy that throws or returns
   * anything but 'allow' or 'ask' denies, and only a confirm callback resolving to true approves.
   */
  async decide(name, args) {
    let decision = 'allow';
    if (this.policy) {
      try {
        decision = await this.policy(name, args);
      } catch {
        decision = 'deny';
      }
    }
    if (decision === 'allow') return 'allow';
    if (decision === 'ask') {
      let approved = false;
      try {
        approved = this.confirm ? (await this.confirm(name, args)) === true : false;
      } catch {
        approved = false;
      }
      if (approved) return 'allow';
      return `Error: "${name}" needs confirmation from the user and was not approved, so it did not run`;
    }
    return `Error: permission denied: the policy does not allow "${name}"`;
  }
}

// ---------- step 2: parsing tool calls out of text ----------

/**
 * Pull every <tool_call>{...}</tool_call> block out of model text. Malformed blocks (bad JSON, no
 * name, no closing tag) are skipped: a model that writes nonsense must not crash the harness.
 */
export function parseToolCalls(text) {
  const calls = [];
  if (typeof text !== 'string') return calls;
  let at = 0;
  for (;;) {
    const open = text.indexOf(OPEN_TAG, at);
    if (open < 0) break;
    const close = text.indexOf(CLOSE_TAG, open + OPEN_TAG.length);
    if (close < 0) break; // unterminated block: nothing reliable left to parse
    const body = text.slice(open + OPEN_TAG.length, close);
    at = close + CLOSE_TAG.length;
    let parsed = null;
    try {
      parsed = JSON.parse(body);
    } catch {
      continue; // skip this block, keep scanning for the next one
    }
    if (!parsed || typeof parsed.name !== 'string') continue;
    const args = parsed.args && typeof parsed.args === 'object' && !Array.isArray(parsed.args) ? parsed.args : {};
    calls.push({ name: parsed.name, args });
  }
  return calls;
}

// ---------- step 3: limits (truncation and a token budget) ----------

/** Cut a long tool result down to `limit` characters and say so, so the context cannot be flooded. */
export function truncate(text, limit) {
  const s = String(text ?? '');
  if (!(limit > 0) || s.length <= limit) return s;
  return `${s.slice(0, limit)}... [truncated ${s.length - limit} characters]`;
}

/**
 * Rough token count for a string. English text under a GPT-2/Llama-style BPE averages roughly
 * 4 characters per token, so `ceil(length / 4)` is the standard back-of-the-envelope estimate.
 */
export function estimateTokens(text) {
  if (typeof text !== 'string' || text.length === 0) return 0;
  return Math.ceil(text.length / 4);
}

/** What a transcript costs: per-message framing, the text, and the tool name when there is one. */
export function contextTokens(messages) {
  let total = 0;
  for (const msg of messages ?? []) {
    total += TOKENS_PER_MESSAGE + estimateTokens(msg.content) + estimateTokens(msg.name);
  }
  return total;
}

// ---------- step 4: the loop ----------

/**
 * Run the loop. Each turn: check the budget, ask the model for text, append it as an assistant
 * message, and either execute the tool calls it contains (one 'tool' message per call) or stop.
 *
 * Stop reasons: 'final' (the model answered without calling a tool), 'max_turns', 'budget'
 * (the transcript grew past maxTokens), 'error' (the model call itself threw). A failing tool is
 * not an error: it is a result the model can read.
 */
export async function runAgentLoop({
  model,
  tools = null,
  messages = [],
  maxTurns = 8,
  maxToolChars = 500,
  maxTokens = Infinity,
  onEvent = null,
} = {}) {
  const emit = (event) => { if (onEvent) onEvent(event); };
  const transcript = messages.slice(); // never mutate the caller's array
  let turns = 0;
  const stop = (stopReason, extra = {}) => {
    const tokens = contextTokens(transcript);
    emit({ type: 'stop', turn: turns, stopReason, tokens });
    return { messages: transcript, turns, stopReason, tokens, ...extra };
  };

  for (;;) {
    if (contextTokens(transcript) > maxTokens) return stop('budget');
    if (turns >= maxTurns) return stop('max_turns');
    turns++;
    emit({ type: 'turn', turn: turns });

    let text;
    try {
      text = await model(transcript, tools ? tools.schemas() : []);
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      emit({ type: 'error', turn: turns, error: message });
      return { messages: transcript, turns, stopReason: 'error', tokens: contextTokens(transcript), error: message };
    }
    transcript.push({ role: 'assistant', content: String(text ?? '') });
    emit({ type: 'assistant', turn: turns, content: String(text ?? '') });

    const calls = parseToolCalls(text);
    if (calls.length === 0) return stop('final');

    for (const call of calls) {
      emit({ type: 'tool_call', turn: turns, name: call.name, args: call.args });
      const result = tools
        ? await tools.call(call.name, call.args)
        : 'Error: no tools are available in this run';
      const content = truncate(result, maxToolChars);
      transcript.push({ role: 'tool', name: call.name, content });
      emit({ type: 'tool_result', turn: turns, name: call.name, content });
    }
  }
}

// ---------- step 5: observability ----------

/** One transcript row per message: [label, characters, single-line preview]. */
export function renderTranscript(messages, { width = 60 } = {}) {
  return (messages ?? []).map((msg) => {
    const label = msg.role === 'tool' ? `tool:${msg.name ?? '?'}` : String(msg.role);
    const text = String(msg.content ?? '');
    const flat = text.replace(/\s+/g, ' ').trim();
    const preview = flat.length <= width ? flat : `${flat.slice(0, width - 1)}…`;
    return [label, text.length, preview];
  });
}

/** Fold an onEvent trace into the numbers you would put on a dashboard. */
export function traceSummary(events) {
  const summary = {
    turns: 0,
    assistantChars: 0,
    toolCalls: 0,
    toolErrors: 0,
    toolChars: 0,
    byTool: {},
    stopReason: null,
  };
  for (const e of events ?? []) {
    if (e.type === 'turn') summary.turns = Math.max(summary.turns, e.turn);
    else if (e.type === 'assistant') summary.assistantChars += String(e.content ?? '').length;
    else if (e.type === 'tool_call') {
      summary.toolCalls++;
      summary.byTool[e.name] = (summary.byTool[e.name] ?? 0) + 1;
    } else if (e.type === 'tool_result') {
      const content = String(e.content ?? '');
      summary.toolChars += content.length;
      if (content.startsWith('Error:')) summary.toolErrors++;
    } else if (e.type === 'stop') summary.stopReason = e.stopReason;
    else if (e.type === 'error') summary.stopReason = 'error';
  }
  return summary;
}
