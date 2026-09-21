// Tests for lib/harness.js — run with: node --test lib/tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ToolRegistry, mockModel, parseToolCalls, runAgentLoop } from '../harness.js';

/** The three-tool catalogue the demo task uses: look up a price, compute a total, save a receipt. */
function makeTools() {
  const saved = [];
  const tools = new ToolRegistry();
  tools.register('lookup_price', {
    description: 'Look up the unit price of an item in the catalogue.',
    parameters: { type: 'object', properties: { item: { type: 'string' } }, required: ['item'] },
    handler: ({ item }) => {
      const catalogue = { lamp: 12, boat: 30, hat: 5 };
      if (!(item in catalogue)) throw new Error(`no such item: ${item}`);
      return String(catalogue[item]);
    },
  });
  tools.register('total', {
    description: 'Multiply a unit price by a count and add tax.',
    parameters: {
      type: 'object',
      properties: { price: { type: 'number' }, count: { type: 'integer' }, tax: { type: 'number' } },
      required: ['price', 'count'],
    },
    handler: ({ price, count, tax = 0.1 }) => String(Math.round(price * count * (1 + tax) * 100) / 100),
  });
  tools.register('save_receipt', {
    description: 'Save a receipt line.',
    parameters: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
    handler: ({ text }) => {
      saved.push(text);
      return 'saved';
    },
  });
  return { tools, saved };
}

/** Shorthand for the wire format the model emits. */
const callText = (name, args) => `<tool_call>${JSON.stringify({ name, args })}</tool_call>`;

test('ToolRegistry: schemas, has, and a successful call', async () => {
  const { tools } = makeTools();
  assert.equal(tools.has('lookup_price'), true);
  assert.equal(tools.has('nope'), false);
  const schemas = tools.schemas();
  assert.deepEqual(schemas.map((s) => s.name), ['lookup_price', 'total', 'save_receipt']);
  assert.ok(schemas.every((s) => typeof s.description === 'string' && s.parameters.type === 'object'));
  assert.ok(!('handler' in schemas[0]), 'handlers are not shown to the model');
  assert.equal(await tools.call('lookup_price', { item: 'lamp' }), '12');
  assert.equal(await tools.call('total', { price: 12, count: 3, tax: 0 }), '36');
});

test('ToolRegistry: bad calls come back as error strings, never thrown', async () => {
  const { tools } = makeTools();
  assert.match(await tools.call('missing_tool', {}), /^Error: unknown tool "missing_tool"/);
  assert.match(await tools.call('lookup_price', {}), /^Error: missing required argument "item"/);
  assert.match(await tools.call('lookup_price', { item: 7 }), /^Error: argument "item" must be a string/);
  assert.match(await tools.call('total', { price: 1, count: 1.5 }), /^Error: argument "count" must be an integer/);
  assert.match(await tools.call('lookup_price', { item: 'car' }), /^Error: no such item: car/);
  assert.match(await tools.call('lookup_price', 'not an object'), /^Error: arguments must be an object/);
  // Unknown extra arguments are tolerated.
  assert.equal(await tools.call('lookup_price', { item: 'hat', colour: 'red' }), '5');
});

test('ToolRegistry: enum validation and non-string results', async () => {
  const tools = new ToolRegistry();
  tools.register('pick', {
    parameters: { type: 'object', properties: { colour: { type: 'string', enum: ['red', 'blue'] } }, required: ['colour'] },
    handler: ({ colour }) => ({ colour, ok: true }),
  });
  assert.equal(await tools.call('pick', { colour: 'red' }), '{"colour":"red","ok":true}');
  assert.match(await tools.call('pick', { colour: 'green' }), /must be one of "red", "blue"/);
});

test('parseToolCalls: finds every well-formed block and skips the rest', () => {
  const text = `thinking ${callText('lookup_price', { item: 'lamp' })} and ${callText('total', { price: 12, count: 2 })}`;
  assert.deepEqual(parseToolCalls(text), [
    { name: 'lookup_price', args: { item: 'lamp' } },
    { name: 'total', args: { price: 12, count: 2 } },
  ]);
  assert.deepEqual(parseToolCalls('no calls here'), []);
  assert.deepEqual(parseToolCalls(''), []);
  assert.deepEqual(parseToolCalls(undefined), []);
});

test('parseToolCalls: malformed blocks never throw', () => {
  const cases = [
    '<tool_call>{not json}</tool_call>',
    '<tool_call></tool_call>',
    '<tool_call>{"args":{"a":1}}</tool_call>',        // no name
    '<tool_call>{"name":42}</tool_call>',             // name is not a string
    '<tool_call>{"name":"x","args":',                 // unterminated block
    '<tool_call>[1,2,3]</tool_call>',                 // not an object
  ];
  for (const text of cases) assert.deepEqual(parseToolCalls(text), [], text);
  // A bad block does not hide a good one that follows it.
  assert.deepEqual(
    parseToolCalls(`<tool_call>{oops}</tool_call>${callText('total', { price: 1, count: 1 })}`),
    [{ name: 'total', args: { price: 1, count: 1 } }],
  );
  // Non-object args are replaced by {} rather than passed through.
  assert.deepEqual(parseToolCalls('<tool_call>{"name":"x","args":"nope"}</tool_call>'), [{ name: 'x', args: {} }]);
});

test('runAgentLoop: a scripted 3-tool task finishes with stopReason "final"', async () => {
  const { tools, saved } = makeTools();
  const model = mockModel([
    callText('lookup_price', { item: 'lamp' }),
    callText('total', { price: 12, count: 3, tax: 0.1 }),
    callText('save_receipt', { text: '3 lamps = 39.6' }),
    'Three lamps cost 39.6 with tax. Receipt saved.',
  ]);
  const events = [];
  const result = await runAgentLoop({
    model,
    tools,
    messages: [{ role: 'system', content: 'be brief' }, { role: 'user', content: 'price of 3 lamps with tax?' }],
    onEvent: (event) => events.push(event),
  });
  assert.equal(result.stopReason, 'final');
  assert.equal(result.turns, 4);
  assert.deepEqual(saved, ['3 lamps = 39.6']);
  assert.deepEqual(result.messages.map((m) => m.role), [
    'system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant', 'tool', 'assistant',
  ]);
  const toolMessages = result.messages.filter((m) => m.role === 'tool');
  assert.deepEqual(toolMessages.map((m) => m.name), ['lookup_price', 'total', 'save_receipt']);
  assert.deepEqual(toolMessages.map((m) => m.content), ['12', '39.6', 'saved']);
  assert.equal(result.messages.at(-1).content, 'Three lamps cost 39.6 with tax. Receipt saved.');
  // The event trace mirrors the transcript.
  assert.equal(events.filter((e) => e.type === 'tool_call').length, 3);
  assert.equal(events.filter((e) => e.type === 'turn').length, 4);
  assert.deepEqual(events.at(-1), { type: 'stop', turn: 4, stopReason: 'final' });
});

test('runAgentLoop: the input message array is not mutated', async () => {
  const { tools } = makeTools();
  const messages = [{ role: 'user', content: 'hi' }];
  const result = await runAgentLoop({ model: mockModel(['hello']), tools, messages });
  assert.equal(messages.length, 1);
  assert.equal(result.messages.length, 2);
});

test('runAgentLoop: a failing tool becomes a tool message, not a crash', async () => {
  const { tools } = makeTools();
  const model = mockModel([
    callText('lookup_price', { item: 'car' }),
    callText('lookup_price', { item: 'hat' }),
    'A hat costs 5.',
  ]);
  const result = await runAgentLoop({ model, tools, messages: [{ role: 'user', content: 'price?' }] });
  assert.equal(result.stopReason, 'final');
  const toolMessages = result.messages.filter((m) => m.role === 'tool');
  assert.match(toolMessages[0].content, /^Error: no such item: car/);
  assert.equal(toolMessages[1].content, '5');
});

test('runAgentLoop: maxTurns is respected', async () => {
  const { tools } = makeTools();
  const model = mockModel(() => callText('lookup_price', { item: 'hat' })); // never answers
  const result = await runAgentLoop({ model, tools, messages: [], maxTurns: 3 });
  assert.equal(result.stopReason, 'max_turns');
  assert.equal(result.turns, 3);
  assert.equal(result.messages.filter((m) => m.role === 'assistant').length, 3);
});

test('runAgentLoop: a model that throws stops with "error"', async () => {
  const { tools } = makeTools();
  const model = async () => { throw new Error('model offline'); };
  const events = [];
  const result = await runAgentLoop({ model, tools, messages: [], onEvent: (e) => events.push(e) });
  assert.equal(result.stopReason, 'error');
  assert.equal(result.turns, 1);
  assert.match(result.error, /model offline/);
  assert.equal(events.at(-1).type, 'error');
});

test('runAgentLoop: long tool results are truncated with a marker', async () => {
  const tools = new ToolRegistry();
  tools.register('dump', { parameters: null, handler: () => 'x'.repeat(5000) });
  const model = mockModel([callText('dump', {}), 'done']);
  const result = await runAgentLoop({ model, tools, messages: [], maxToolChars: 100 });
  const toolMessage = result.messages.find((m) => m.role === 'tool');
  assert.equal(toolMessage.content.length, 100 + '... [truncated 4900 characters]'.length);
  assert.match(toolMessage.content, /\.\.\. \[truncated 4900 characters\]$/);
});

test('runAgentLoop: without tools a tool call is reported back to the model', async () => {
  const model = mockModel([callText('anything', {}), 'giving up']);
  const result = await runAgentLoop({ model, messages: [] });
  assert.equal(result.stopReason, 'final');
  assert.match(result.messages.find((m) => m.role === 'tool').content, /^Error: no tools/);
});

test('mockModel: replies in order, then empty, and sees the transcript', async () => {
  const scripted = mockModel(['one', 'two']);
  assert.equal(await scripted([]), 'one');
  assert.equal(await scripted([]), 'two');
  assert.equal(await scripted([]), '');
  const fromFunction = mockModel((messages, tools) => `${messages.length} messages, ${tools.length} tools`);
  const { tools } = makeTools();
  const result = await runAgentLoop({ model: fromFunction, tools, messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(result.messages.at(-1).content, '1 messages, 3 tools');
});
