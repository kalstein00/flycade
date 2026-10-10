'use strict';
// Read-only view state. Indices address the actual bounded graph; IDs remain strings.
class NeuronInspector {
  constructor() { this.reset(); }
  reset() { this.selected=null;this.filter='all';this.zoom=1;this.frame='all';this.action=0;this.history=[];this.stream=null; }
  mean(values) { return Array.isArray(values) && values.length && values.every(Number.isFinite) ? values.reduce((a,b)=>a+b,0)/values.length : null; }
  accept(envelope) {
    const sample=envelope.sample;if(!sample)return;
    const stream=[sample.run_id,sample.session_id,sample.stream,sample.worker,envelope.generation].join(':');
    if(stream!==this.stream){this.history=[];this.stream=stream;}
    const last=this.history.at(-1);if(last?.id===sample.sample_id)return;
    const means=Object.fromEntries(envelope.graph.nodes.map((node,index)=>[node.index,this.mean(sample.activity[index])]));
    this.history.push({id:sample.sample_id,time:sample.observed_unix,sequence:envelope.sequence,means,probabilities:sample.probabilities,
      gap:!!last&&(envelope.sequence!==last.sequence+1||sample.observed_unix-last.time>2/Math.max(1,envelope.observer?.requested_hz||3)||sample.observed_unix<=last.time)});
    this.history=this.history.filter(row=>sample.observed_unix-row.time<=30).slice(-90);
  }
  chart(svg, value, min, max) {
    const ns='http://www.w3.org/2000/svg',element=(tag,attrs,text)=>{const node=document.createElementNS(ns,tag);for(const [key,v] of Object.entries(attrs))node.setAttribute(key,v);if(text!==undefined)node.textContent=text;svg.append(node);return node;};
    svg.dataset.count=this.history.length;
    const latest=this.history.at(-1)?.time||0,x=time=>32+(time-latest+30)/30*240,y=v=>12+(max-v)/(max-min)*64;
    for(const v of [min,(min+max)/2,max]){element('line',{x1:32,x2:272,y1:y(v),y2:y(v),class:'axis'});element('text',{x:3,y:y(v)+4},String(v));}
    element('text',{x:32,y:94},'−30 s');element('text',{x:246,y:94},'0 s');
    let segment=[];const flush=()=>{if(!segment.length)return;const d=segment.map(([a,b],i)=>`${i?'L':'M'}${a} ${b}`).join(' ');element('path',{d,'data-series':'true',fill:'none',class:'series'});segment=[];};
    for(const row of this.history){const v=value(row);if(row.gap||v==null)flush();if(v!=null&&Number.isFinite(v)){segment.push([x(row.time),y(v)]);element('circle',{cx:x(row.time),cy:y(v),r:1.5,'data-time':row.time,'data-value':v,class:'point'});}}
    flush();
  }
  render(root, graph, sample, redraw) {
    const find=selector=>root.querySelector(selector), ns='http://www.w3.org/2000/svg';
    const svg=find('#circuit');
    const nodes=graph.nodes.map((node,index)=>({...node,mean:this.mean(sample.activity[index])}));
    const selected=nodes.find(node=>node.index===this.selected);
    const visible=nodes.filter(node=>this.filter==='all'||node.group===this.filter||node.index===this.selected && this.filter==='neighbors'||this.filter==='neighbors'&&graph.edges.some(([a,b])=>(a===this.selected&&b===node.index)||(b===this.selected&&a===node.index)));
    const positions=new Map(visible.map(node=>[node.index,node]));
    const links=graph.edges.filter(([a,b])=>positions.has(a)&&positions.has(b));
    const width=340/this.zoom,height=350/this.zoom,center=this.zoom===1?{x:170,y:175}:selected||{x:170,y:175};
    svg.setAttribute('viewBox',`${center.x-width/2} ${center.y-height/2} ${width} ${height}`);
    const element=(tag,attrs)=>{const node=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);svg.append(node);return node;};
    for(const [a,b] of links){const start=positions.get(a),end=positions.get(b);element('line',{x1:start.x,y1:start.y,x2:end.x,y2:end.y,'data-edge':`${a}:${b}`,class:a===this.selected||b===this.selected?'selected-edge':''});}
    const choose=index=>{this.selected=index;redraw();};
    for(const node of visible){
      const mean=node.mean,neutral=[101,113,122],end=mean<0?[133,189,232]:[239,181,110];
      const color=mean===null?'none':`rgb(${neutral.map((c,i)=>Math.round(c+(end[i]-c)*Math.abs(mean))).join(',')})`;
      const attrs={'data-node':node.index,'data-group':node.group,fill:color,stroke:'#bac4c7','stroke-width':node.index===this.selected?2.5:.5,tabindex:0,role:'button','aria-label':`${node.root_id} · ${node.cell_type||'종류 미상'} · ${mean===null?'활성 누락':mean.toFixed(5)}`,'aria-pressed':String(node.index===this.selected)};
      if(mean!==null)attrs['data-mean']=mean;
      let mark;
      if(node.group==='input')mark=element('circle',{...attrs,cx:node.x,cy:node.y,r:4.5});
      else if(node.group==='output')mark=element('rect',{...attrs,x:node.x-4,y:node.y-4,width:8,height:8});
      else mark=element('path',{...attrs,d:`M${node.x} ${node.y-5} l5 5 -5 5 -5 -5 Z`});
      mark.addEventListener('click',()=>choose(node.index));
      mark.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose(node.index);}});
      const title=document.createElementNS(ns,'title');title.textContent=attrs['aria-label'];mark.append(title);
    }
    for(const [x,label] of [[35,'입력'],[145,'내부'],[255,'출력']])element('text',{x,y:20}).textContent=label;
    find('#node-filter').value=this.filter;
    find('#node-filter').addEventListener('change',event=>{this.filter=event.target.value;redraw();});
    find('#zoom-in').disabled=this.zoom>=3;find('#zoom-out').disabled=this.zoom<=1;
    find('#zoom-in').addEventListener('click',()=>{this.zoom=Math.min(3,this.zoom+.5);redraw();});
    find('#zoom-out').addEventListener('click',()=>{this.zoom=Math.max(1,this.zoom-.5);redraw();});
    find('#zoom-level').textContent=`${this.zoom.toFixed(1)}배`;
    find('#filter-counts').textContent=`표시 ${visible.length}/${graph.used_nodes}개 노드 · ${links.length}/${graph.used_edges}개 연결. 표시 제외 ${graph.used_nodes-visible.length}개 노드 · ${graph.used_edges-links.length}개 연결. 정책 변경 없음.`;
    find('#activity-summary').textContent=`표시 노드 |평균| ≥ 0.1: ${visible.filter(n=>n.mean!==null&&Math.abs(n.mean)>=.1).length} / ${visible.length} · 활성 누락 ${visible.filter(n=>n.mean===null).length}`;
    find('#graph-counts').textContent=`관측 부분집합 ${nodes.length} / 정책 사용 ${graph.used_nodes} · 연결 ${graph.edges.length} / 사용 ${graph.used_edges}. 미표시 노드는 미사용 또는 0을 뜻하지 않습니다.`;
    find('#graph-version').textContent=`${graph.version} · SHA-256 ${graph.sha256}`;
    const groupNames={input:'입력',internal:'내부',output:'출력'};
    const ranked=[...visible].sort((a,b)=>((b.mean===null?-Infinity:Math.abs(b.mean))-(a.mean===null?-Infinity:Math.abs(a.mean)))||a.index-b.index).slice(0,10);
    if(selected&&!ranked.some(n=>n.index===selected.index))ranked.push(selected);
    for(const node of ranked){
      const row=document.createElement('tr');row.dataset.neuronRow=node.index;row.setAttribute('aria-selected',String(node.index===this.selected));
      const identity=document.createElement('td'),button=document.createElement('button');button.type='button';button.dataset.nodeSelect=node.index;button.textContent=node.root_id;button.addEventListener('click',()=>choose(node.index));identity.append(button);
      const annotation=document.createElement('small');annotation.textContent=`${node.cell_type||'종류 미상'} · ${node.neuropil||node.region||'영역 미상'}`;identity.append(annotation);
      const group=document.createElement('td');group.textContent=groupNames[node.group]+(node.index===this.selected?' · 선택':'');
      const value=document.createElement('td');value.textContent=node.mean===null?'누락':node.mean.toFixed(5);row.append(identity,group,value);find('#neurons').append(row);
    }
    find('#selected-neuron').textContent=selected?`${selected.root_id} · ${selected.cell_type||'종류: 알 수 없음'} · 영역: ${selected.neuropil||selected.region||'알 수 없음'} · ${groupNames[selected.group]} · 정책 사용 노드`:'회로나 표에서 뉴런을 선택하세요.';
    const value=find('#selected-value');value.textContent=selected?(selected.mean===null?'활성 누락':`모델 활성 평균 ${selected.mean.toFixed(5)} · 무차원`):'선택 없음';if(selected?.mean!=null)value.dataset.value=selected.mean;
    const action=find('#history-action');
    for(const [index,buttons] of sample.actions.entries()){const option=document.createElement('option');option.value=index;option.textContent=`${index} · ${buttons.join(' + ')||'NOOP'}`;action.append(option);}
    action.value=this.action;action.addEventListener('change',event=>{this.action=Number(event.target.value);redraw();});
    find('#history-activity-label').textContent=selected?`선택 뉴런 ${selected.root_id} · 평균 (−1~+1, 무차원)`:'표시 부분집합의 노드 평균 · 무차원 (−1~+1)';
    this.chart(find('#history-activity'),row=>selected?row.means[selected.index]:this.mean(Object.values(row.means)), -1,1);
    this.chart(find('#history-probability'),row=>row.probabilities[this.action],0,1);
    find('#history-note').textContent=`최근 30초 · 최대 90개 · 현재 ${this.history.length}개. 가로축: 마지막 표본 대비 초. 누락/전송 생략 구간은 끊음 · 새 세션/스트림에서 초기화.`;
    const frame=find('#frame-choice');frame.value=this.frame;frame.addEventListener('change',event=>{this.frame=event.target.value;redraw();});
    const image=find('#selected-input');image.hidden=this.frame==='all';if(this.frame!=='all'){image.src=sample.pixels[Number(this.frame)];image.alt=`선택 정책 입력 프레임 ${Number(this.frame)+1}`;}
  }
}
window.NeuronInspector=NeuronInspector;
