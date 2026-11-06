document.querySelectorAll('.zoom').forEach(button=>button.addEventListener('click',()=>{const dialog=document.getElementById('figureDialog');const img=dialog.querySelector('img');img.src=button.dataset.src;img.alt=button.querySelector('img').alt;dialog.showModal();}));
document.getElementById('closeFigure').addEventListener('click',()=>document.getElementById('figureDialog').close());
document.getElementById('figureDialog').addEventListener('click',event=>{if(event.target===event.currentTarget)event.currentTarget.close();});
const slider=document.getElementById('cutoff');
const ns='http://www.w3.org/2000/svg';
let domain=.3;
function svgElement(tag,attributes,text){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attributes))el.setAttribute(k,String(v));if(text!==undefined)el.textContent=text;return el;}
function drawDistribution(threshold){const svg=document.getElementById('scoreChart');svg.replaceChildren();const W=900,H=290,L=58,R=24,T=26,B=45;const bins=window.picHistogram.filter(r=>r.left<domain);const ymax=Math.max(...bins.map(r=>r.within_outcome_fraction))*1.18;const x=v=>L+v/domain*(W-L-R),y=v=>H-B-v/ymax*(H-T-B);
for(let j=0;j<=4;j++){const v=ymax*j/4;svg.append(svgElement('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:'#e1e8eb'}));svg.append(svgElement('text',{x:L-9,y:y(v)+4,'text-anchor':'end',fill:'#71838a','font-size':11},(v*100).toFixed(0)+'%'));}
for(let j=0;j<=6;j++){const v=domain*j/6;svg.append(svgElement('text',{x:x(v),y:H-B+22,'text-anchor':'middle',fill:'#71838a','font-size':11},v.toFixed(2)));}
for(const outcome of [0,1]){for(const r of bins.filter(r=>r.outcome===outcome)){const width=x(Math.min(r.right,domain))-x(r.left);svg.append(svgElement('rect',{x:x(r.left)+1,y:y(r.within_outcome_fraction),width:Math.max(0,width-2),height:H-B-y(r.within_outcome_fraction),fill:outcome?'#cc704b':'#638ca8',opacity:outcome ? 0.66 : 0.4}));}}
svg.append(svgElement('text',{x:L,y:14,fill:'#71838a','font-size':11},'各结局组内记录比例'));
svg.append(svgElement('text',{x:(L+W-R)/2,y:H-3,'text-anchor':'middle',fill:'#71838a','font-size':11},'冻结 RF 的预测分数'));
if(threshold<=domain){svg.append(svgElement('line',{x1:x(threshold),x2:x(threshold),y1:T,y2:H-B,stroke:'#18323d','stroke-width':2,'stroke-dasharray':'5 4'}));svg.append(svgElement('text',{x:Math.min(x(threshold)+8,W-R-90),y:T+12,fill:'#18323d','font-size':12},'阈值 '+threshold.toFixed(4)));}
document.getElementById('rangeNote').textContent=domain===.3?'图中放大 0–0.3 区间；完整分数可切换至 0–1。':'展示完整 0–1 分数范围。';}
function render(){const row=window.picThresholds[Number(slider.value)];slider.setAttribute('aria-valuetext',row.threshold.toFixed(4));document.getElementById('cutoffValue').textContent=row.threshold.toFixed(4);document.getElementById('sens').textContent=(row.sensitivity*100).toFixed(1)+'%';document.getElementById('spec').textContent=(row.specificity*100).toFixed(1)+'%';document.getElementById('fp').textContent=row.fp.toLocaleString();document.getElementById('fn').textContent=row.fn.toLocaleString();drawDistribution(row.threshold);}
function reset(){slider.value=window.picThresholds.reduce((best,r,i)=>Math.abs(r.threshold-window.picFrozenThreshold)<Math.abs(window.picThresholds[best].threshold-window.picFrozenThreshold)?i:best,0);render();}
slider.max=window.picThresholds.length-1;slider.addEventListener('input',render);document.getElementById('reset').addEventListener('click',reset);
document.querySelectorAll('[data-domain]').forEach(button=>button.addEventListener('click',()=>{domain=Number(button.dataset.domain);document.querySelectorAll('[data-domain]').forEach(b=>b.classList.toggle('active',b===button));render();}));
reset();
