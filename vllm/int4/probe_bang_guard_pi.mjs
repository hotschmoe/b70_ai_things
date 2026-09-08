// Actual Pi extension recovery into the live candidate after one injected failure.
// Run as a job of the leased campaign server. This is not a natural-loop rate test.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer, request } from 'node:http';
import { once } from 'node:events';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const [outArg, model, guardDir, sdkDir] = process.argv.slice(2);
assert.ok(outArg && model && guardDir && sdkDir, 'out model guardDir sdkDir required');
const out = resolve(outArg);
await mkdir(out, { recursive: false });
const pkg = JSON.parse(await readFile(join(sdkDir, 'package.json'), 'utf8'));
const identity = await (await fetch('http://127.0.0.1:18125/v1/models')).json();
assert.ok(identity.data.some(m => m.id === model));
await writeFile(join(out, 'models-live.json'), JSON.stringify(identity));
const attempts = [];
const errors = [];
let emitted = 0;
let child;
const expected = 'RECOVERY_739251';
const server = createServer(async (req, res) => {
  try {
    assert.equal(req.url, '/v1/chat/completions');
    let raw = '';
    for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw);
    attempts.push({ time: new Date().toISOString(), body, injected: attempts.length === 0 });
    await writeFile(join(out, 'attempts.json'), JSON.stringify(attempts, null, 2));
    assert.ok(attempts.length <= 4, 'bounded recovery');
    if (attempts.length === 1) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      for (let i = 0; i < 128 && !res.destroyed; i++) {
        res.write(`data: ${JSON.stringify({ id: 'injected', object: 'chat.completion.chunk',
          created: 1, model: 'hotschmoe-dd', choices: [{ index: 0, delta: { content: '!' }, finish_reason: null }] })}\n\n`);
        emitted++;
        await delay(4);
      }
      if (!res.destroyed) res.end('data: [DONE]\n\n');
    } else {
      // Translation is explicit; client alias exercises default salt injection.
      body.model = model;
      body.chat_template_kwargs = { enable_thinking: false };
      body.max_tokens = 512;
      const upstream = request('http://127.0.0.1:18125/v1/chat/completions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
      }, response => {
        res.writeHead(response.statusCode, response.headers);
        response.pipe(res);
      });
      upstream.on('error', error => { errors.push(String(error)); res.destroy(); });
      res.on('close', () => upstream.destroy());
      upstream.end(JSON.stringify(body));
    }
  } catch (error) { errors.push(String(error)); res.destroy(); }
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
try {
  await writeFile(join(out, 'models.json'), JSON.stringify({ providers: { 'bang-live': {
    baseUrl: `http://127.0.0.1:${server.address().port}/v1`, api: 'openai-completions', apiKey: 'local-test',
    models: [{ id: 'hotschmoe-dd', name: 'hotschmoe-dd', reasoning: false, input: ['text'],
      contextWindow: 200000, maxTokens: 512, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
  } } }));
  await writeFile(join(out, 'settings.json'), JSON.stringify({ compaction: { enabled: false }, retry: { enabled: false } }));
  const env = { ...process.env, PI_CODING_AGENT_DIR: out, PI_OFFLINE: '1' };
  delete env.BANG_GUARD_CACHE_SALT_PROVIDERS;
  delete env.BANG_GUARD_MAX_RETRIES;
  const prompt = 'Background inventory records:\n' + 'The warehouse stores parts and maintains an inventory ledger.\n'.repeat(8000)
    + `\nThe task is to reply with exactly ${expected} and nothing else. Preserve this task if recovery is needed.`;
  child = spawn(process.execPath, [join(sdkDir, pkg.bin.pi), '--offline', '-p', '--no-session', '--no-tools',
    '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes', '--no-context-files',
    '-e', resolve(guardDir, 'src/bang-guard.ts'), '--provider', 'bang-live', '--model', 'hotschmoe-dd'],
    { cwd: out, env, stdio: ['pipe', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.stderr.on('data', chunk => { stderr += chunk; });
  child.stdin.end(prompt);
  const timer = setTimeout(() => child.kill('SIGKILL'), 900000);
  let code, signal;
  try { [code, signal] = await once(child, 'exit'); }
  finally { clearTimeout(timer); }
  await writeFile(join(out, 'stdout.txt'), stdout);
  await writeFile(join(out, 'stderr.txt'), stderr);
  assert.equal(signal, null);
  assert.equal(code, 0);
  assert.equal(stdout.trim(), expected);
  assert.equal(attempts.length, 2);
  assert.ok(emitted >= 32 && emitted < 128);
  assert.ok(attempts.every(a => typeof a.body.cache_salt === 'string'));
  assert.notEqual(attempts[0].body.cache_salt, attempts[1].body.cache_salt);
  assert.ok(!JSON.stringify(attempts[1].body.messages).includes('!'.repeat(32)));
  assert.deepEqual(errors, []);
  await writeFile(join(out, 'result.json'), JSON.stringify({ passed: true, pi_version: pkg.version,
    injected_failures: 1, live_recoveries: 1, emitted, model,
    limitation: 'Injected initial corruption; proves client recovery into live model, not natural fault recovery or rarity.' }, null, 2));
  console.log('PASS actual Pi cancellation, salt rotation, clean context and live long-prompt recovery');
} finally {
  if (child && child.exitCode === null) child.kill('SIGTERM');
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
