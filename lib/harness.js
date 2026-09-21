// lib/harness.js — the reference agent loop.
// This file is the REFERENCE SOLUTION for module 20 (The agent loop harness).
//
// An "agent" is a loop written in ordinary code: call the model, parse its intent, execute the tools it
// asked for, append the results, repeat until a stop condition. The model contributes text; the harness
// owns everything that matters operationally — which tools exist, whether the arguments are valid, how
// long results may be, how many turns are allowed, and what the transcript looks like afterwards.
//
// The message array is the single source of truth. Roles: 'system', 'user', 'assistant', 'tool'.
// Tool output is DATA, never instructions: it is appended as a 'tool' message and the model may be wrong
// about it (prompt injection through tool results is a real attack, see the module's concept section).

/** The wire format the scripted models in this lab emit: <tool_call>{"name":..,"args":{..}}</tool_call> */
const OPEN_TAG = '<tool_call>';
const CLOSE_TAG = '</tool_call>';

/** Check `args` against a minimal JSON-schema subset; returns an error string, or null when valid. */
function validateArgs(schema, args) {
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

/** Turn a handler's return value into the string that goes back to the model. */
function resultToString(value) {
  if (typeof value === 'string') return value;
  if (value === undefined) return '';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** The tools an agent may call: their schemas for the model, their handlers for the harness. */
export class ToolRegistry {
  constructor() {
    this.tools = new Map();
  }

  /** Add a tool. `parameters` is a minimal JSON schema: { type:'object', properties, required }. */
  register(name, { description = '', parameters = null, handler }) {
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
   * Run a tool and return its result as a string. Unknown tools, invalid arguments and handler
   * exceptions all come back as 'Error: ...' strings instead of throwing: the model gets to read the
   * mistake and try again, and one bad call cannot take the whole loop down.
   */
  async call(name, args = {}) {
    const tool = this.tools.get(name);
    if (!tool) {
      const known = [...this.tools.keys()].join(', ') || 'none';
      return `Error: unknown tool "${name}". Available tools: ${known}`;
    }
    const problem = validateArgs(tool.parameters, args);
    if (problem) return `Error: ${problem}`;
    try {
      return resultToString(await tool.handler(args));
    } catch (err) {
      return `Error: ${err && err.message ? err.message : String(err)}`;
    }
  }
}

/**
 * Pull every <tool_call>{...}</tool_call> block out of model text. Malformed blocks (bad JSON, no name,
 * missing closing tag) are skipped: a model that writes nonsense should not crash the harness.
 */
export function parseToolCalls(text) {
  const calls = [];
  if (typeof text !== 'string') return calls;
  let at = 0;
  while (true) {
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

/** Cut a long tool result down to `limit` characters and say so, so the context cannot be flooded. */
function truncate(text, limit) {
  if (limit <= 0 || text.length <= limit) return text;
  return `${text.slice(0, limit)}... [truncated ${text.length - limit} characters]`;
}

/**
 * Run the loop. Each turn: ask the model for text, append it as an assistant message, and either
 * execute the tool calls it contains (appending one 'tool' message per call) or stop.
 * Stop reasons: 'final' (the model answered without calling a tool), 'max_turns', 'error' (the model
 * itself threw — tool failures are not errors, they are results the model can read).
 */
export async function runAgentLoop({ model, tools = null, messages = [], maxTurns = 8, maxToolChars = 2000, onEvent }) {
  const emit = (event) => { if (onEvent) onEvent(event); };
  const transcript = messages.slice();
  let turns = 0;
  while (turns < maxTurns) {
    turns++;
    emit({ type: 'turn', turn: turns });
    let text;
    try {
      text = await model(transcript, tools ? tools.schemas() : []);
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      emit({ type: 'error', turn: turns, error: message });
      return { messages: transcript, turns, stopReason: 'error', error: message };
    }
    transcript.push({ role: 'assistant', content: text });
    emit({ type: 'assistant', turn: turns, content: text });

    const calls = parseToolCalls(text);
    if (calls.length === 0) {
      emit({ type: 'stop', turn: turns, stopReason: 'final' });
      return { messages: transcript, turns, stopReason: 'final' };
    }
    for (const call of calls) {
      emit({ type: 'tool_call', turn: turns, name: call.name, args: call.args });
      const result = tools
        ? await tools.call(call.name, call.args)
        : `Error: no tools are available in this run`;
      const content = truncate(result, maxToolChars);
      transcript.push({ role: 'tool', name: call.name, content });
      emit({ type: 'tool_result', turn: turns, name: call.name, content });
    }
  }
  emit({ type: 'stop', turn: turns, stopReason: 'max_turns' });
  return { messages: transcript, turns, stopReason: 'max_turns' };
}

/**
 * A model you can test a loop with: either a function of the messages, or a list of replies returned in
 * order. When a scripted list runs out the model replies with an empty string, which has no tool calls
 * and therefore ends the loop with 'final' rather than spinning until maxTurns.
 */
export function mockModel(script) {
  if (typeof script === 'function') {
    return async (messages, tools) => script(messages, tools);
  }
  const replies = script.slice();
  let at = 0;
  return async () => (at < replies.length ? replies[at++] : '');
}
