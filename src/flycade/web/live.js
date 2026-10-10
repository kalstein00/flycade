'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const number = (value, digits = 0) => value == null ? '—' : Number(value).toLocaleString('ko-KR', {maximumFractionDigits: digits});
const time = value => new Date(value * 1000).toLocaleTimeString('ko-KR');
const actionName = buttons => buttons.length ? buttons.join(' + ') : 'NOOP';
const set = (root, selector, value) => { $(selector, root).textContent = value; };
let current = null, runID = null, lastReceived = 0, connected = false, busy = false, timer;
const terminal = {disabled:'관측 꺼짐 · 데이터 없음', completed:'학습 종료 · 저장 완료', saved:'학습 중단 · 저장 완료', failed:'학습 실패 · 마지막 표본', interrupted:'학습 중단 · 마지막 표본', trainer_stopped:'학습 프로세스 종료 · 마지막 표본'};
function status() {
  const now = Date.now() / 1000;
  let message = !connected ? '연결 끊김 · 재연결 중' : current?.status === 'no_data' ? '관측 데이터 없음' : terminal[current?.status];
  if (connected && current?.control?.state === 'complete' && current.control.stop && current.control.active) message = '저장 완료 · 종료 정리 중';
  if (connected && current?.control?.recovery?.state === 'validating') message = '복구 정상본 검증 중';
  if (connected && current?.control?.recovery?.state === 'failed') message = '복구 실패 · 정상본 확인 필요';
  if (!message) message = current?.sample ? (now - current.sample.observed_unix > 2 ? '표본 지연 · 마지막 표본' : '연결됨 · 학습 중') : '연결됨 · 첫 표본 대기';
  $('#status').textContent = message;
  $('#received').textContent = lastReceived ? `마지막 수신 ${time(lastReceived)}` : '수신된 표본 없음';
  $('#lag').textContent = current?.sample ? `표본 경과 ${number(now - current.sample.observed_unix, 1)}초` : '지연 —';
}
function operation(control) {
  if (!control) {$('#operation').textContent='저장 상태 확인 불가';return;}
  const labels={idle:'학습 중 · 저장 전',waiting_boundary:'저장 요청 접수 · 안전 경계 대기',saving:'체크포인트 저장 중',complete:control.active?(control.stop?'저장 완료 · 종료 정리 중':'저장 완료 · 학습 계속'):'저장 완료 · 종료 완료',failed:'저장 또는 학습 실패',trainer_stopped:'프로세스 종료 · 마지막 정상본 확인 필요'};
  $('#operation').textContent=control.pending_request ? '저장 요청 전송 · 접수 대기' : labels[control.state] || control.state;
  if(control.delay_seconds) $('#operation').textContent+=` · 대기 ${number(control.delay_seconds,1)}초`;
  if(control.error) $('#operation').textContent+=` · ${control.error}`;
  const recovery=control.recovery;
  let recoveryMessage='';
  if(recovery?.state==='validating') recoveryMessage='복구 중 · 정상본의 전체 학습 상태를 검증합니다.';
  if(recovery?.state==='restored') recoveryMessage=`복구 완료 · update ${number(recovery.restored_updates)}에서 재개 · 되돌린 진도 ${number(recovery.lost_updates)} update · ${number(recovery.lost_transitions)}전이. 기록되지 않은 작업도 유실됐을 수 있습니다.`;
  if(recovery?.state==='failed') recoveryMessage='복구 실패 · 검증 가능한 정상본이 없습니다. 일치하는 로컬 백업을 복원하거나 새 Run을 시작하세요.';
  $('#rollback').textContent=recoveryMessage;
  $('#recovery-details').hidden=!recovery?.rejected_candidates?.length;
  $('#recovery-errors').textContent=(recovery?.rejected_candidates||[]).map(row=>`${row.checkpoint_id}: ${row.error}`).join('\n');
  const saved=control.last_save;
  $('#recovery').textContent=recovery?.state==='failed'?'복구 가능한 정상본 없음':saved?`마지막 복구 시점 ${time(saved.completed_unix)} · update ${number(saved.updates)} · ${number(saved.transitions)}전이`:'정상 저장본 없음';
  $('#reset-notice').textContent=control.reset ? '세션 시작: 새 에피소드 · 누적 학습 진도 유지' : '';
}
function circuit(root, graph, sample) {
  const svg = $('#circuit', root), ns = 'http://www.w3.org/2000/svg';
  const element = (tag, attrs) => {const node = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) node.setAttribute(k, v); svg.append(node); return node;};
  const positions = new Map(graph.nodes.map(n => [n.index, n]));
  for (const [a,b] of graph.edges) {const start = positions.get(a), end = positions.get(b); element('line', {x1:start.x,y1:start.y,x2:end.x,y2:end.y});}
  for (const [index, node] of graph.nodes.entries()) {
    const values = sample.activity[index], mean = values.reduce((a,b) => a+b, 0)/values.length;
    const neutral = [101,113,122], end = mean < 0 ? [133,189,232] : [239,181,110];
    const color = `rgb(${neutral.map((c,i) => Math.round(c+(end[i]-c)*Math.abs(mean))).join(',')})`;
    const attrs = {'data-node':node.index,'data-mean':mean,fill:color,stroke:'#bac4c7','stroke-width':.5};
    let mark;
    if (node.group === 'input') mark = element('circle', {...attrs,cx:node.x,cy:node.y,r:4.5});
    else if (node.group === 'output') mark = element('rect', {...attrs,x:node.x-4,y:node.y-4,width:8,height:8});
    else mark = element('path', {...attrs,d:`M${node.x} ${node.y-5} l5 5 -5 5 -5 -5 Z`});
    const title = document.createElementNS(ns,'title');title.textContent = `${node.root_id} · ${node.cell_type || '종류 미상'} · 평균 ${mean.toFixed(5)}`;mark.append(title);
  }
  for (const [x,label] of [[35,'입력'],[145,'내부'],[255,'출력']]) element('text',{x,y:20}).textContent = label;
  const means = sample.activity.map(v => v.reduce((a,b) => a+b,0)/v.length);
  set(root, '#activity-summary', `표시 노드 |평균| ≥ 0.1: ${means.filter(v => Math.abs(v)>=.1).length} / ${means.length}`);
  set(root, '#graph-counts', `노드 ${graph.nodes.length} / 사용 ${graph.used_nodes} · 연결 ${graph.edges.length} / 사용 ${graph.used_edges}. 제외: 노드 ${graph.excluded_nodes}, 연결 ${graph.excluded_edges}.`);
  set(root, '#graph-version', `${graph.version} · SHA-256 ${graph.sha256}`);
}
async function render(envelope) {
  const s = envelope.sample;
  if (!s) { $('#view').replaceChildren(Object.assign(document.createElement('p'), {className:'empty',textContent:'관측 데이터 없음 · 학습의 첫 표본을 기다립니다.'})); return; }
  const fragment = document.importNode($('#sample-template').content, true);
  fragment.firstElementChild.dataset.sampleId = s.sample_id;
  $('#raw', fragment).src = s.raw;
  for (const [index, uri] of s.pixels.entries()) { const img = new Image(); img.src = uri; img.alt = `정책 입력 프레임 ${index+1} (오래된 순)`; $('#inputs', fragment).append(img); }
  const [stack,h,w,c] = s.pixel_shape;
  set(fragment,'#input-contract', `${w}×${h} · ${c===3?'RGB':'회색조'} · ${stack}프레임, 오래된 순. uint8 원본 전처리 픽셀 (정책 내부 /255).`);
  set(fragment,'#chosen',actionName(s.buttons)); set(fragment,'#episode',s.episode); set(fragment,'#step',s.episode_step);
  set(fragment,'#reward',number(s.transition.reward,3)); set(fragment,'#progress',`${number(s.transition.max_progress)} px`);
  const reasons = {death:'사망',completion:'완료',no_progress:'진행 없음',external_limit:'프레임 제한',game_timeout:'게임 시간 초과'};
  set(fragment,'#reason',s.transition.reason ? reasons[s.transition.reason] || s.transition.reason : '계속 진행');
  set(fragment,'#argmax',s.argmax);
  for (const [index, probability] of s.probabilities.entries()) {
    const row = document.createElement('div');row.className = `action-row ${index===s.action?'selected':''}`;row.dataset.action=index;
    const label = document.createElement('span');label.className='action-label';label.textContent=`${index} · ${actionName(s.actions[index])}${index===s.action?' · 선택':''}`;
    const value = document.createElement('span');value.className='action-value';value.textContent=`${(probability*100).toFixed(1)}%`;
    const meter = document.createElement('meter');meter.min=0;meter.max=1;meter.value=probability;meter.setAttribute('aria-label',label.textContent);
    row.append(label,value,meter);$('#actions',fragment).append(row);
  }
  circuit(fragment, envelope.graph, s);
  const m = envelope.final_metrics || s.metrics;
  const metrics = [['누적 전이',number(m.transitions)],['최대 진행',`${number(m.max_progress)} px`],['완료 / 에피소드',`${number(m.episode_outcomes?.completion || 0)} / ${number(m.episodes)}`],['학습 시간',`${number(m.training_seconds,1)} s`],['전이 / 초',number(m.transitions_per_second,1)],['PPO loss',number(m.loss,4)],['탐색 entropy',number(m.entropy,4)],['optimizer step',number(m.optimizer_steps)],['프로세스 최대 RAM',`${number((m.peak_rss_bytes ?? s.resources.peak_rss_bytes)/2**20)} MiB`],['정책 현재 VRAM',`${number(s.resources.cuda_allocated_bytes/2**20)} MiB`],['표본 / 전송 전 폐기',`${envelope.observer.published} / ${envelope.observer.dropped}`],['모델 업데이트',number(m.updates)]];
  for (const [label,value] of metrics) {const div=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;div.append(dt,dd);$('#metrics',fragment).append(div);}
  set(fragment,'#metrics-time',`지표 갱신 ${time(envelope.final_metrics ? envelope.updated_unix : s.metrics_unix)}`);
  set(fragment,'#save-state',m.checkpoint_id ? `저장 완료 · 체크포인트 ${m.checkpoint_id}` : '학습 진행 중 · 이 세션의 체크포인트 저장 전');
  set(fragment,'#sample-id',s.sample_id);set(fragment,'#stream',`${s.stream} / ${s.worker}`);set(fragment,'#policy-version',`update ${s.policy_version}`);set(fragment,'#sample-time',new Date(s.observed_unix*1000).toISOString());set(fragment,'#observation-hash',s.observation_sha256);
  // Decode all panel images before a single DOM replacement: no mixed observations.
  await Promise.all([...fragment.querySelectorAll('img')].map(img => img.decode()));
  const focused = [...$('#view').querySelectorAll('summary')].indexOf(document.activeElement);
  for (const [index, detail] of [...$('#view').querySelectorAll('details')].entries()) {const next=fragment.querySelectorAll('details')[index];if(next)next.open=detail.open;}
  $('#view').replaceChildren(fragment);
  if (focused >= 0) $('#view').querySelectorAll('summary')[focused]?.focus({preventScroll:true});
}
function accepts(next) {
  if (runID && next.run_id !== runID) return false;
  if (next.status==='no_data') return !current || current.status==='no_data';
  if (next.sample && (next.sample.run_id!==next.run_id || next.sample.session_id!==next.session_id || next.sample.stream!=='training' || next.sample.worker!==0)) return false;
  if (!current || current.status==='no_data') return true;
  if (next.generation !== current.generation) return next.generation > current.generation;
  if (next.session_id !== current.session_id || next.sequence < current.sequence || next.updated_unix < current.updated_unix) return false;
  return !next.sample || !current.sample || (next.sample.step>=current.sample.step && next.sample.policy_version>=current.sample.policy_version);
}
async function poll() {
  if (busy) return;
  busy=true;$('#reconnect').disabled=true;
  try {
    const response = await fetch('/api/latest',{cache:'no-store',signal:AbortSignal.timeout(2000)});
    if(!response.ok)throw new Error('Unavailable');
    const next=await response.json();
    if (accepts(next)) {
      if (!current || next.generation!==current.generation || next.sequence!==current.sequence || next.status!==current.status) await render(next);
      operation(next.control);
      current=next;runID=next.run_id;lastReceived=Date.now()/1000;
      $('#run').textContent=next.run_id.slice(0,8);$('#run').title=next.run_id;
      $('#session').textContent=next.session_id?.slice(0,8)||'—';$('#session').title=next.session_id||'';
      $('#stage').textContent=next.sample?.stage||'—';$('#update').textContent=number(next.final_metrics?.updates ?? next.sample?.policy_version);
      $('#frequency').textContent=next.observer ? `최대 ${next.observer.requested_hz} Hz · 실측 ${number(next.observer.actual_hz,1)} Hz` : '표본 빈도 —';
    }
    connected=true;
  } catch {connected=false;}
  finally {busy=false;$('#reconnect').disabled=false;status();clearTimeout(timer);timer=setTimeout(poll,200);}
}
$('#reconnect').addEventListener('click',()=>{clearTimeout(timer);poll();});
setInterval(status,1000);
poll();
