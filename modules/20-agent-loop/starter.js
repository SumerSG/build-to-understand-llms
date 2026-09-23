// Module 20 — The agent loop harness.
// An "agent" is a loop written in ordinary code: call the model, parse its intent, execute the tools
// it asked for, append the results to the message array, repeat until a stop condition fires.
//
// The message array is the single source of truth. Roles: 'system', 'user', 'assistant', 'tool'.
// Tool output is DATA, never instructions — the harness only ever parses tool calls out of ASSISTANT
// messages, which is what stops a poisoned tool result from driving the loop.
//
// Everything above the "step 1" line is done for you; read it, it sets the conventions.

// ---------- worked examples and constants (done for you) ----------

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

/**
 * One property against its spec: type first, then an optional enum of allowed values.
 * Returns an error string, or null when the value is acceptable. `validateArgs` (step 1) calls this
 * once per argument — note the shape of the messages: they name the argument and say what was expected.
 */
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

// ---------- step 1: the tool registry ----------

/**
 * Check `args` against a minimal JSON-schema subset: { type:'object', properties, required }.
 * Return an error string describing the FIRST problem found, or null when the arguments are valid.
 *
 * Rules: a tool with no schema (null, or type other than 'object') accepts anything; `args` must be
 * a plain object; every name in `required` must be present; every argument that appears in
 * `properties` must satisfy `checkType`; arguments the schema does not mention are ignored.
 */
export function validateArgs(schema, args) {
  // TODO: step 1
  return null;
}

/** The tools an agent may call: their schemas for the model, their handlers for the harness. */
export class ToolRegistry {
  constructor() {
    this.tools = new Map();
  }

  /** Add a tool. `parameters` is a minimal JSON schema, or null for "no declared arguments". */
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
   * Run a tool and return its result as a string. An unknown tool, invalid arguments and a handler
   * that throws must all come back as an `Error: ...` string instead of throwing: the model gets to
   * read the mistake and try again, and one bad call cannot take the whole run down.
   */
  async call(name, args = {}) {
    // TODO: step 1 — unknown tool -> `Error: unknown tool "x". Available tools: ...`,
    // invalid arguments -> `Error: <what validateArgs said>`, otherwise run the handler
    // (it may be async) and stringify its return value with resultToString.
    return '';
  }
}

// ---------- step 2: parsing tool calls out of text ----------

/**
 * Pull every <tool_call>{"name":..,"args":{..}}</tool_call> block out of model text and return
 * them in order as `{ name, args }`. A block whose body is not valid JSON, or that has no string
 * `name`, or that is never closed, is skipped — scanning continues after it. A non-object `args`
 * becomes `{}`. A non-string input returns `[]`. This function must never throw.
 */
export function parseToolCalls(text) {
  // TODO: step 2
  return [];
}

// ---------- step 3: limits (truncation and a token budget) ----------

/**
 * Cut `text` down to `limit` characters and append `... [truncated N characters]`, where N is how
 * many characters were DROPPED. A limit of 0, a negative limit or Infinity means "no limit".
 * Text exactly `limit` characters long is returned unchanged.
 */
export function truncate(text, limit) {
  // TODO: step 3
  return String(text ?? '');
}

/**
 * Rough token count for a string: English text under a GPT-2/Llama-style BPE averages roughly
 * 4 characters per token, so use `ceil(length / 4)`. Anything that is not a string costs 0.
 */
export function estimateTokens(text) {
  // TODO: step 3
  return 0;
}

/**
 * What a transcript costs: for each message, TOKENS_PER_MESSAGE of role framing, plus its
 * `content`, plus its `name` when it has one (a tool message carries the tool name too).
 */
export function contextTokens(messages) {
  // TODO: step 3
  return 0;
}

// ---------- step 4: the loop ----------

/**
 * Run the agent loop and return `{ messages, turns, stopReason, tokens, error? }`.
 *
 * Each iteration: stop with 'budget' if `contextTokens(transcript) > maxTokens`; stop with
 * 'max_turns' if `turns` has reached `maxTurns`; otherwise count a turn, call
 * `await model(transcript, tools ? tools.schemas() : [])`, append the reply as an assistant
 * message, and parse it. No tool calls means the model answered: stop with 'final'. Otherwise run
 * each call through `tools.call`, `truncate` the result to `maxToolChars`, and append it as
 * `{ role: 'tool', name, content }` before looping.
 *
 * A model that throws stops the run with 'error' and the message in `error`. A tool that fails is
 * NOT an error: `tools.call` already turned it into a string the model can read.
 * Do not mutate the caller's `messages` array. Call `onEvent` (when given) with
 * `{type:'turn'|'assistant'|'tool_call'|'tool_result'|'stop'|'error', turn, ...}` — step 5 reads it.
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
  // TODO: step 4
  return { messages: messages.slice(), turns: 0, stopReason: 'max_turns', tokens: 0 };
}

// ---------- step 5: observability ----------

/**
 * One row per message, ready for a table: `[label, characters, preview]`.
 * `label` is the role, or `tool:<name>` for a tool message. `characters` is the raw length of the
 * content. `preview` collapses all whitespace to single spaces and is cut to `width` characters,
 * ending in '…' when it was cut.
 */
export function renderTranscript(messages, { width = 60 } = {}) {
  // TODO: step 5
  return [];
}

/**
 * Fold an onEvent trace into `{ turns, assistantChars, toolCalls, toolErrors, toolChars, byTool,
 * stopReason }`. `turns` is the highest turn number seen; `byTool` counts calls per tool name;
 * `toolErrors` counts results that start with 'Error:'; `stopReason` comes from the stop event
 * (an 'error' event means the stop reason is 'error').
 */
export function traceSummary(events) {
  // TODO: step 5
  return { turns: 0, assistantChars: 0, toolCalls: 0, toolErrors: 0, toolChars: 0, byTool: {}, stopReason: null };
}
