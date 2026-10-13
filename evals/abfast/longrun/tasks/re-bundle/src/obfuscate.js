// usage: node obfuscate.js IN.js OUT.js   (javascript-obfuscator 5.9.0 resolved from OBF_NODE_MODULES)
const path = require('path');
const JO = require(path.join(process.env.OBF_NODE_MODULES, 'javascript-obfuscator'));
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const res = JO.obfuscate(src, {
  compact: true, target: 'node', seed: 20261014,
  controlFlowFlattening: true, controlFlowFlatteningThreshold: 0.6,
  deadCodeInjection: true, deadCodeInjectionThreshold: 0.25,
  stringArray: true, stringArrayEncoding: ['base64'], stringArrayThreshold: 0.9, rotateStringArray: true, shuffleStringArray: true,
  stringArrayWrappersCount: 2, stringArrayWrappersType: 'function',
  identifierNamesGenerator: 'hexadecimal', numbersToExpressions: true, simplify: true, splitStrings: true, splitStringsChunkLength: 6,
  transformObjectKeys: true, unicodeEscapeSequence: false, selfDefending: false, debugProtection: false, renameGlobals: false,
});
fs.writeFileSync(process.argv[3], res.getObfuscatedCode());
