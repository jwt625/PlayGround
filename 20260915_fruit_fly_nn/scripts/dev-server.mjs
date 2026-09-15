/** Check the requested port before starting Vite; never displace another service. */
import { createConnection } from 'node:net';
import { spawn } from 'node:child_process';
const args=process.argv.slice(2);
const option=(name,fallback)=>{const i=args.indexOf(name);return i>=0?args[i+1]:fallback;};
const port=Number(option('--port','5173'));
const host=option('--host','127.0.0.1');
const connectHost=host==='0.0.0.0'?'127.0.0.1':host;
if(!Number.isInteger(port)||port<1||port>65535)throw new Error('Invalid port');
const occupied=await new Promise(resolve=>{const s=createConnection({host:connectHost,port});s.setTimeout(1500);s.once('connect',()=>{s.destroy();resolve(true)});s.once('error',()=>resolve(false));s.once('timeout',()=>{s.destroy();resolve(true)});});
if(occupied){console.log(`Port ${port} is already occupied. No server started. Inspect its owner with: lsof -nP -iTCP:${port} -sTCP:LISTEN. Reuse it if it is this project, or choose an available --port.`);process.exit(0);}
console.log(`Port ${port} is available; starting this project's Vite server.`);
const extra=[];if(!args.includes('--host'))extra.push('--host',host);if(!args.includes('--port'))extra.push('--port',String(port));if(!args.includes('--strictPort'))extra.push('--strictPort');
const child=spawn(process.execPath,['node_modules/vite/bin/vite.js',...args,...extra],{stdio:'inherit'});
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>child.kill(signal));
child.on('exit',code=>process.exit(code??1));
