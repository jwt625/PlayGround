// Lists every name the chat skins pass to the avatar lookup and whether media/cast.js resolves it.
// Usage: node check_cast.js COURSE_DIR
// emulate the names chat_skins.js passes to av(); report which resolve via LMS_CAST
const fs=require('fs'),vm=require('vm');const root=process.argv[2];
const ctx={window:{},document:{},console};ctx.globalThis=ctx.window;vm.createContext(ctx);
for(const f of ['courses/ferroelectric.js','courses/feedback.js','media/cast.js']){try{vm.runInContext(fs.readFileSync(root+'/'+f,'utf8'),ctx);}catch(e){console.log('ERR',f,e.message)}}
const C=ctx.window.LMS_CAST||{};
function castFor(n){if(n==null)return null;n=String(n).trim();if(C[n])return C[n];const l=n.toLowerCase();for(const k in C){if(k.toLowerCase()===l)return C[k];for(const a of (C[k].aliases||[]))if(String(a).toLowerCase()===l)return C[k];}return null;}
const generic=(s)=>!s||/^(messages|inbox|chat|text messages?)$/i.test(String(s).trim());
const names={};
const add=(n,w)=>{(names[n]=names[n]||new Set()).add(w)};
function walk(o,cid){if(!o||typeof o!=='object')return;if(Array.isArray(o)){o.forEach(x=>walk(x,cid));return;}
 if(o.type==='chat'){const msgs=o.messages||[],meta=o.meta||{};const all=[...new Set([].concat(meta.participants||[],msgs.map(m=>m.from)))].filter(Boolean);
  const me=[...new Set(msgs.filter(m=>m.side==='right').map(m=>m.from))];const others=all.filter(p=>me.indexOf(p)<0);const sk=o.skin||o.app;
  all.forEach(p=>add(p,cid+':'+sk));
  if(['imessage','signal','whatsapp','phone'].includes(sk)){const t=others.length===1?others[0]:!generic(meta.header)?meta.header:others.join(', ')||'Conversation';add(t,cid+':'+sk+':title');}
  if(sk==='teams'){const n=!generic(meta.header)?String(meta.header).replace(/^#\s*/,''):all.join(', ');add(n,cid+':teams:grp');}
  if(meta.header) add(meta.header, cid+':'+sk+':header(collectImages)');
 }
 for(const k in o)if(typeof o[k]==='object')walk(o[k],cid);}
for(const id in ctx.window.LMS_COURSES)walk(ctx.window.LMS_COURSES[id],id);
let miss=0;for(const n of Object.keys(names).sort()){const c=castFor(n);if(!c)miss++;console.log((c?'OK  ':'MISS')+' '+JSON.stringify(n)+' -> '+(c?c.avatar:'-')+'  ['+[...names[n]].slice(0,3).join(' ')+']');}
console.log('missing',miss);
// check files exist
for(const k in C){if(!fs.existsSync(root+'/'+C[k].avatar))console.log('NOFILE',k,C[k].avatar);}
