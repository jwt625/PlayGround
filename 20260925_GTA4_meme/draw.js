/* Shared deterministic canvas renderer: browser preview and MP4 use the same frames. */
(function () {
  const W=1440,H=900;
  const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
  const duration=c=>c.intro+c.cards.length*c.hold+c.outro;
  const assetPaths=c=>[...new Set([c.background,...(c.backgrounds||[]).map(b=>b.image),
    ...c.cards.flatMap(card=>[card.background,card.image])].filter(Boolean))];
  function motion(c,card,p){
    p=clamp(p);
    const profile=typeof card.motion==='object'?card.motion:c.motionProfiles?.[card.motion];
    const side=card.side||1;
    // Old saved templates retain their original motion and single background.
    const fallback={
      background:{x:[-side*17/W,side*17/W],y:[0,0],zoom:[1.08,1.105]},
      portrait:{x:[side*28/W,-side*28/W],y:[12/H,-6/H],height:[1.13,1.165]}
    };
    const sample=(layer,key)=>{
      const l=profile?.[layer],range=l?.[key]||fallback[layer][key];
      // Nearly linear travel with restrained easing; no stops, bobbing, or oscillation.
      const e=clamp(l?.ease||0),q=p*(1-e)+(p*p*(3-2*p))*e;
      return range[0]+(range[1]-range[0])*q;
    };
    return {background:{x:sample('background','x'),y:sample('background','y'),zoom:sample('background','zoom')},
      portrait:{x:sample('portrait','x'),y:sample('portrait','y'),height:sample('portrait','height')}};
  }
  function backgroundRect(bg,pose){
    const scale=Math.max(W/bg.width,H/bg.height)*Math.max(1,pose.zoom);
    const width=bg.width*scale,height=bg.height*scale;
    // Clamp oversized user-edited pans so no empty strip can enter the frame.
    return {x:clamp((W-width)/2+pose.x*W,W-width,0),
      y:clamp((H-height)/2+pose.y*H,H-height,0),width,height};
  }
  function lines(ctx,text,maxWidth){
    const out=[]; let line='';
    for(const word of text.split(/\s+/)){const test=line?line+' '+word:word;if(ctx.measureText(test).width>maxWidth&&line){out.push(line);line=word;}else line=test;}
    if(line)out.push(line);return out;
  }
  function label(ctx,text,x,y,size,color='#fff',weight=700){ctx.fillStyle=color;ctx.font=`${weight} ${size}px Arial, sans-serif`;ctx.fillText(text,x,y);}
  function title(ctx,c,t,end=false){
    const length=end?c.outro:c.intro;
    ctx.globalAlpha=clamp(Math.min(t/.5,(length-t)/.5));
    ctx.textAlign='center';ctx.lineJoin='round';
    ctx.font='900 112px Arial Black, Arial, sans-serif';ctx.lineWidth=3;ctx.strokeStyle='#f3f0e8';ctx.fillStyle='#f3f0e8';
    c.title.forEach((s,i)=>{ctx.strokeText(s,W/2,365+i*105);ctx.fillText(s,W/2,365+i*105);});
    label(ctx,end?'LOADING THE NEXT FUNDING ROUND…':c.subtitle,W/2,570,20,'#9c9c96',500);
    label(ctx,'IV  /  AI INDUSTRY PARODY',W/2,640,18,'#686863',600);
    ctx.textAlign='left';ctx.globalAlpha=1;
  }
  function draw(ctx,c,images,t){
    ctx.save();ctx.scale(ctx.canvas.width/W,ctx.canvas.height/H);ctx.fillStyle='#050505';ctx.fillRect(0,0,W,H);
    if(t<c.intro){title(ctx,c,t);ctx.restore();return;}
    // With no outro, hold the final composition when the preview reaches its end.
    const body=t-c.intro, i=c.outro===0?Math.min(c.cards.length-1,Math.floor(body/c.hold)):Math.floor(body/c.hold);
    if(i>=c.cards.length){title(ctx,c,body-c.cards.length*c.hold,true);ctx.restore();return;}
    const card=c.cards[i], local=body-i*c.hold,p=clamp(local/c.hold),side=card.side||1;
    const fadeIn=i===0&&c.intro===0?1:local/c.fade;
    const fadeOut=i===c.cards.length-1&&c.outro===0?1:(c.hold-local)/c.fade;
    ctx.globalAlpha=clamp(Math.min(fadeIn,fadeOut));
    // Fade the whole composite as a group by covering it with black afterward.
    const opacity=ctx.globalAlpha;ctx.globalAlpha=1;
    const pose=motion(c,card,p),bg=images[card.background||c.background];
    const rect=backgroundRect(bg,pose.background);
    ctx.drawImage(bg,rect.x,rect.y,rect.width,rect.height);
    ctx.fillStyle='rgba(247,247,244,0.17)';ctx.fillRect(0,0,W,H);
    const person=images[card.image], ph=H*pose.portrait.height,pw=ph*person.width/person.height;
    const cx=side===1?W*.69:W*.31;
    ctx.drawImage(person,cx-pw/2+pose.portrait.x*W,pose.portrait.y*H,pw,ph);
    if(c.labels){
      const left=side===1,x=left?68:858,width=515;
      const grad=ctx.createLinearGradient(left?0:W,0,left?W*.58:W*.42,0);
      grad.addColorStop(0,'rgba(9,12,15,0.90)');grad.addColorStop(.7,'rgba(9,12,15,0.64)');grad.addColorStop(1,'rgba(9,12,15,0)');
      ctx.fillStyle=grad;ctx.fillRect(left?0:W*.42,0,W*.58,H);
      label(ctx,'GRAND THEFT ALIGNMENT  /  IV',x,76,17,'#d6d6cf');
      label(ctx,card.tag,x,365,18,card.tint);
      ctx.font='900 49px Arial, sans-serif';ctx.fillStyle='#f4f2eb';
      const names=lines(ctx,card.name,width);names.forEach((s,n)=>ctx.fillText(s,x,425+n*55));
      const y=435+(names.length-1)*55;
      label(ctx,'CAST AS '+card.role,x,y+30,15,'#c7c7c0',500);
      ctx.fillStyle=card.tint;ctx.fillRect(x,y+63,54,4);
      ctx.font='500 27px Arial, sans-serif';ctx.fillStyle='#e8e6df';
      lines(ctx,card.caption,width-10).forEach((s,n)=>ctx.fillText(s,x,y+112+n*38));
      label(ctx,String(i+1).padStart(2,'0')+' / '+String(c.cards.length).padStart(2,'0')+'   •   LOADING COMPUTE CITY',x,802,14,'#a9aaa5',600);
      label(ctx,'FICTIONAL MEME DIALOGUE',x,832,12,'#888a86',500);
    }
    ctx.fillStyle=`rgba(0,0,0,${1-opacity})`;ctx.fillRect(0,0,W,H);ctx.restore();
  }
  globalThis.MemeRenderer={draw,duration,assetPaths,motion,backgroundRect,W,H};
})();
