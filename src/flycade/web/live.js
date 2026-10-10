'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const number = (value, digits = 0) => value == null ? '—' : Number(value).toLocaleString('ko-KR', {maximumFractionDigits: digits});
const time = value => new Date(value * 1000).toLocaleTimeString('ko-KR');
const actionName = buttons => buttons.length ? buttons.join(' + ') : 'NOOP';
const set = (root, selector, value) => { $(selector, root).textContent = value; };
let catalog=null, selectedRun=null, selectedWorker=0, selectionVersion=0, requestController=null;
let current = null, runID = null, lastReceived = 0, connected = false, busy = false, timer;
const terminal = {disabled:'관측 꺼짐 · 데이터 없음', completed:'학습 종료 · 저장 완료', saved:'학습 중단 · 저장 완료', failed:'학습 실패 · 마지막 표본', interrupted:'학습 중단 · 마지막 표본', trainer_stopped:'학습 프로세스 종료 · 마지막 표본'};
function status() {
  const now = Date.now() / 1000;
  let message = !connected ? '연결 끊김 · 재연결 중' : current?.status === 'no_data' ? '관측 데이터 없음' : terminal[current?.status];
  if (connected && current?.control?.state === 'complete' && current.control.stop && current.control.active) message = '저장 완료 · 종료 정리 중';
  if (connected && current?.control?.recovery?.state === 'validating') message = '복구 정상본 검증 중';
  if (connected && current?.control?.recovery?.state === 'failed') message = '복구 실패 · 정상본 확인 필요';
  if (!message) message = current?.sample ? (now - current.sample.observed_unix > 2 ? '표본 지연 · 마지막 표본' : '연결됨 · 학습 중') : '연결됨 · 첫 표본 대기';
  if (connected && current?.control?.training_paused) message = '고정 정책 평가 중 · 학습 대기 · 마지막 학습 표본';
  $('#status').textContent = message;
  $('#received').textContent = lastReceived ? `마지막 수신 ${time(lastReceived)}` : '수신된 표본 없음';
  $('#lag').textContent = current?.sample ? `표본 경과 ${number(now - current.sample.observed_unix, 1)}초` : '지연 —';
}
function operation(control) {
  if (!control) {$('#operation').textContent='저장 상태 확인 불가';return;}
  const labels={idle:'학습 중 · 저장 전',waiting_boundary:'저장 요청 접수 · 안전 경계 대기',saving:'체크포인트 저장 중',complete:control.active?(control.stop?'저장 완료 · 종료 정리 중':'저장 완료 · 학습 계속'):'저장 완료 · 종료 완료',failed:'저장 또는 학습 실패',trainer_stopped:'프로세스 종료 · 마지막 정상본 확인 필요'};
  $('#operation').textContent=control.pending_request ? '저장 요청 전송 · 접수 대기' : labels[control.state] || control.state;
  if(control.training_paused) $('#operation').textContent='순차 CPU 평가 중 · 학습 대기 · snapshot '+control.evaluating_snapshot;
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
function storageInformation(info) {
  if (!info || info.error) {$('#storage-summary').textContent='확인 실패';$('#storage-result').textContent=info?.error || '저장 공간 정보를 읽을 수 없습니다.';return;}
  $('#storage-summary').textContent=`남은 공간 ${number(info.free_bytes/1024**3,2)} GiB${info.low_space?' · 공간 부족':''}`;
  const p=info.policy,c=info.cleanup;
  $('#storage-policy').textContent=`최근 평가 ${p.keep_evaluations}개 + 보호본 · 학습 영상 ${p.keep_videos}개 · 로그별 ${number(p.log_bytes/1024**2,2)} MiB`;
  $('#storage-result').textContent=c.status==='failed'?`정리 실패 · ${(c.errors||[]).join(' / ')} · 마지막 정상 저장본을 확인하세요.`:c.status==='not_run'?'아직 정리한 내역 없음':`최근 정리 ${time(c.at_unix)} · 삭제 ${(c.removed||[]).length}건 · 로그 축소 ${(c.trimmed_logs||[]).length}건`;
}
function runInformation(info) {
  if (!info) return;
  const lineage=info.lineage, budget=info.budget;
  $('#run-kind').textContent=({full_state_branch:'전체 상태 분기',warm_start:'가중치 재사용 · 새 실험'})[lineage?.kind] || '새 학습 실험';
  $('#run-identity').textContent=`Run ${info.run_id}`;
  $('#lineage').textContent=lineage ? `부모 Run ${lineage.parent_run_id} · 체크포인트 ${lineage.parent_checkpoint_id} · 부모 update ${number(lineage.parent_updates)}. ${lineage.kind==='warm_start'?'전체 가중치 로드 · optimizer·난수·진도·저장/평가 스케줄은 새로 시작합니다.':'모델·optimizer·난수·누적 진도·스케줄을 이어받습니다. 초기 정책은 조상 Run의 초기본입니다.'}` : '부모 없음 · 새 초기 정책에서 시작';
  $('#budget-info').textContent=budget ? `총 예산 ${number(budget.total_updates)} updates · 최초 예산 ${number(budget.original_updates)} · 고정 학습률, 누적 스케줄 유지` : `예산 기록 확인 실패: ${info.error}`;
  $('#budget-events').textContent=budget?.events.length ? budget.events.map(event=>`${new Date(event.created_unix*1000).toLocaleString('ko-KR')} · update ${number(event.at_updates)}에서 예산 ${number(event.previous_updates)} → ${number(event.total_updates)}`).join(' / ') : '예산 연장 이력 없음';
}
const inspector=new NeuronInspector();
let renderVersion=0;
async function render(envelope) {
  const version=++renderVersion;
  const s = envelope.sample;
  inspector.accept(envelope);
  if (!s) { $('#view').replaceChildren(Object.assign(document.createElement('p'), {className:'empty',textContent:'관측 데이터 없음 · 학습의 첫 표본을 기다립니다.'})); return; }
  const fragment = document.importNode($('#sample-template').content, true);
  fragment.firstElementChild.dataset.sampleId = s.sample_id;
  $('#raw', fragment).src = s.raw;
  for (const [index, uri] of s.pixels.entries()) { const img = new Image(); img.src = uri; img.alt = `정책 입력 프레임 ${index+1} (오래된 순)`; $('#inputs', fragment).append(img); }
  const [stack,h,w,c] = s.pixel_shape;
  set(fragment,'#input-contract', `${w}×${h} · ${c===3?'RGB':'회색조'} · ${stack}프레임, 오래된 순. uint8 원본 전처리 픽셀 (정책 내부 /255). 표시용 역정규화 없음.`);
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
  for(const [index] of s.pixels.entries()){const option=document.createElement('option');option.value=index;option.textContent=`프레임 ${index+1}`;$('#frame-choice',fragment).append(option);}
  inspector.render(fragment,envelope.graph,s,()=>render(current));
  const m = envelope.final_metrics || s.metrics;
  const metrics = [['누적 전이',number(m.transitions)],['최대 진행',`${number(m.max_progress)} px`],['완료 / 에피소드',`${number(m.episode_outcomes?.completion || 0)} / ${number(m.episodes)}`],['학습 시간',`${number(m.training_seconds,1)} s`],['전이 / 초',number(m.transitions_per_second,1)],['PPO loss',number(m.loss,4)],['탐색 entropy',number(m.entropy,4)],['optimizer step',number(m.optimizer_steps)],['프로세스 최대 RAM',`${number((m.peak_rss_bytes ?? s.resources.peak_rss_bytes)/2**20)} MiB`],['정책 현재 VRAM',`${number(s.resources.cuda_allocated_bytes/2**20)} MiB`],['표본 / 전송 전 폐기',`${envelope.observer.published} / ${envelope.observer.dropped}`],['모델 업데이트',number(m.updates)]];
  for (const [label,value] of metrics) {const div=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;div.append(dt,dd);$('#metrics',fragment).append(div);}
  set(fragment,'#metrics-time',`지표 갱신 ${time(envelope.final_metrics ? envelope.updated_unix : s.metrics_unix)}`);
  set(fragment,'#save-state',m.checkpoint_id ? `저장 완료 · 체크포인트 ${m.checkpoint_id}` : '학습 진행 중 · 이 세션의 체크포인트 저장 전');
  set(fragment,'#sample-id',s.sample_id);set(fragment,'#stream',`${s.stream} / ${s.worker}`);set(fragment,'#policy-version',`update ${s.policy_version}`);set(fragment,'#sample-time',new Date(s.observed_unix*1000).toISOString());set(fragment,'#observation-hash',s.observation_sha256);
  // Decode all panel images before a single DOM replacement: no mixed observations.
  await Promise.all([...fragment.querySelectorAll('img[src]')].map(img => img.decode()));
  if(version!==renderVersion)return;
  const active=document.activeElement;
  const focusID=active?.id,focusNode=active?.dataset.node,focusButton=active?.dataset.nodeSelect;
  const focused = [...$('#view').querySelectorAll('summary')].indexOf(document.activeElement);
  for (const [index, detail] of [...$('#view').querySelectorAll('details')].entries()) {const next=fragment.querySelectorAll('details')[index];if(next)next.open=detail.open;}
  $('#view').replaceChildren(fragment);
  if (focused >= 0) $('#view').querySelectorAll('summary')[focused]?.focus({preventScroll:true});
  if(focusID)document.getElementById(focusID)?.focus({preventScroll:true});
  else if(focusNode)document.querySelector(`#circuit [data-node="${focusNode}"]`)?.focus({preventScroll:true});
  else if(focusButton)document.querySelector(`[data-node-select="${focusButton}"]`)?.focus({preventScroll:true});
}
function accepts(next) {
  if (next.run_id !== selectedRun || next.selected_worker!==selectedWorker) return false;
  if (runID && next.run_id !== runID) return false;
  if (next.status==='no_data') return !current || current.status==='no_data';
  if (next.sample && (next.sample.run_id!==next.run_id || next.sample.session_id!==next.session_id || next.sample.stream!=='training' || next.sample.worker!==selectedWorker)) return false;
  if (!current || current.status==='no_data') return true;
  if (next.generation !== current.generation) return next.generation > current.generation;
  if (next.session_id !== current.session_id || next.sequence < current.sequence || next.updated_unix < current.updated_unix) return false;
  return !next.sample || !current.sample || (next.sample.step>=current.sample.step && next.sample.policy_version>=current.sample.policy_version);
}
function workerChoices() {
  const select=$('#worker-choice');select.replaceChildren();
  for(const worker of catalog.runs.find(run=>run.run_id===selectedRun).workers){const option=document.createElement('option');option.value=worker;option.textContent=String(worker);select.append(option);}
  selectedWorker=Number(select.value);
}
function switchStream() {
  selectionVersion++;renderVersion++;requestController?.abort();current=null;runID=selectedRun;lastReceived=0;connected=false;inspector.reset();
  $('#view').replaceChildren(Object.assign(document.createElement('p'),{className:'empty',textContent:'선택 스트림의 관측을 기다립니다.'}));
  $('#run').textContent=selectedRun.slice(0,8);$('#session').textContent='—';$('#update').textContent='—';$('#stage').textContent='—';
  $('#model-kind').textContent='모델 확인 중';
  $('#run-kind').textContent='—';$('#run-identity').textContent='';$('#lineage').textContent='';$('#budget-info').textContent='';$('#budget-events').textContent='';
  $('#operation').textContent='선택 Run 상태 확인 중';$('#recovery').textContent='';$('#rollback').textContent='';$('#recovery-details').hidden=true;$('#reset-notice').textContent='';
  clearTimeout(timer);poll();
}
$('#run-choice').addEventListener('change',event=>{selectedRun=event.target.value;workerChoices();switchStream();});
$('#worker-choice').addEventListener('change',event=>{selectedWorker=Number(event.target.value);switchStream();});
async function poll() {
  if (busy) return;
  busy=true;$('#reconnect').disabled=true;
  const selection=selectionVersion;
  try {
    if(!catalog){
      const response=await fetch('/api/catalog',{cache:'no-store',signal:AbortSignal.timeout(2000)});if(!response.ok)throw new Error('Catalog unavailable');catalog=await response.json();
      const requested=new URLSearchParams(location.search).get('run');
      selectedRun=catalog.runs.some(run=>run.run_id===requested)?requested:catalog.default_run;
      for(const run of catalog.runs){const option=document.createElement('option');option.value=run.run_id;option.textContent=`${run.label} · ${run.run_id.slice(0,8)}`;$('#run-choice').append(option);}
      $('#run-choice').value=selectedRun;workerChoices();
    }
    requestController=new AbortController();
    const response = await fetch(`/api/latest?run=${encodeURIComponent(selectedRun)}&worker=${selectedWorker}`,{cache:'no-store',signal:AbortSignal.any([requestController.signal,AbortSignal.timeout(2000)])});
    if(!response.ok)throw new Error('Unavailable');
    const next=await response.json();
    if (selection!==selectionVersion)return;
    if (accepts(next)) {
      if (!current || next.generation!==current.generation || next.sequence!==current.sequence || next.status!==current.status) await render(next);
      if(selection!==selectionVersion)return;
      $('#compare-link').href='/compare?'+new URLSearchParams({run:selectedRun});
      $('#evaluation-link').href='/replay?'+new URLSearchParams({run:selectedRun});
      $('#model-kind').textContent=next.model_kind==='cnn'?'모델: CNN PPO':'모델: 커넥톰 PPO';
      operation(next.control);
      storageInformation(next.storage);
      runInformation(next.run_info);
      current=next;runID=next.run_id;lastReceived=Date.now()/1000;
      $('#run').textContent=next.run_id.slice(0,8);$('#run').title=next.run_id;
      $('#session').textContent=next.session_id?.slice(0,8)||'—';$('#session').title=next.session_id||'';
      $('#stage').textContent=next.sample?.stage||'—';$('#update').textContent=number(next.final_metrics?.updates ?? next.sample?.policy_version);
      $('#frequency').textContent=next.observer ? `최대 ${next.observer.requested_hz} Hz · 실측 ${number(next.observer.actual_hz,1)} Hz` : '표본 빈도 —';
    }
    connected=true;
  } catch {if(selection===selectionVersion)connected=false;}
  finally {busy=false;$('#reconnect').disabled=false;status();clearTimeout(timer);timer=setTimeout(poll,200);}
}
$('#reconnect').addEventListener('click',()=>{clearTimeout(timer);poll();});
setInterval(status,1000);
poll();
