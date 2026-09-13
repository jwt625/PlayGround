import type {EquipmentState} from './equipment-state';
/** Dynamic status light and thermal effects, independent of baked chassis condition. */
export function equipmentEffects(c:CanvasRenderingContext2D,state:EquipmentState,x:number,y:number,width:number,height:number,time:number){
 const lampX=x+width*.72,lampY=y+height*.56;
 c.save();const lit=state.passive?state.field:state.powered;
 // A dark inset remains at the same position when power is removed.
 c.fillStyle='#101719';c.fillRect(lampX-3,lampY-2,6,4);
 if(lit){const color=state.condition>0?'#edba70':state.passive?'#8ed9db':'#a4edc9';c.shadowColor=color;c.shadowBlur=5;c.fillStyle=color;c.fillRect(lampX-2,lampY-1,4,2);c.shadowBlur=0;}
 if(state.work==='tripped'||state.work==='uncooled'){c.fillStyle=`rgba(244,157,91,${.55+.3*Math.sin(time*7)})`;c.beginPath();c.moveTo(x+width*.5,y-10);c.lineTo(x+width*.5-5,y-2);c.lineTo(x+width*.5+5,y-2);c.closePath();c.fill();}
 if(state.hot){c.strokeStyle='#e1b77a60';c.lineWidth=1;for(let i=0;i<3;i++){const sx=x+width*(.3+i*.2);c.beginPath();c.moveTo(sx,y+height*.15);c.quadraticCurveTo(sx+Math.sin(time*3+i)*4,y-5,sx+Math.cos(time*2+i)*3,y-12);c.stroke();}}
 if(state.smoke){for(let i=0;i<4;i++){const t=(time*.5+i*.25)%1;c.fillStyle=`rgba(120,123,112,${.22*(1-t)})`;c.beginPath();c.arc(x+width*.5+Math.sin(time+i)*5,y+height*.15-t*28,3+t*6,0,Math.PI*2);c.fill();}}
 c.restore();
}
