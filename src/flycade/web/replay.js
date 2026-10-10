'use strict';
const $ = selector => document.querySelector(selector);
const inspector = new NeuronInspector();
let template, selectedRun, selectedEvaluation, record, version=0, renderVersion=0, timer;
const video = $('#replay-video');
const mode = () => $('#mode-choice').value;
function clearPanels(message) {
  ++renderVersion;
  $('#replay-panels').replaceChildren(Object.assign(document.createElement('p'), {className:'missing-observation',textContent:message}));
  $('#sample-identity').textContent='';$('#sample-time').textContent='';
  $('#replay-status').textContent=message;
  $('#live-image').replaceChildren();
}
async function draw() {
  const currentVersion=++renderVersion;
  if (record?.report.replay?.status === 'failed') { clearPanels('기록 실패 · '+record.report.replay.error+' · 디스크 공간·권한을 확인하고 다시 평가하세요.'); return; }
  if (!record?.document || !template) { clearPanels('회로 관측 데이터 없음 · 이 평가에는 재생 가능한 관찰 이력이 없습니다.'); return; }
  const documentData=record.document, live=mode()==='live';
  if (!live && !record.report.videos.length) { clearPanels('저장된 영상 없음 · 동기 재생을 사용할 수 없습니다.'); return; }
  if (!live && video.seeking) { clearPanels('탐색 중 · 해당 영상 시점의 관측을 기다립니다.'); return; }
  const time=video.currentTime;
  const sample=live?documentData.sample:documentData.samples.findLast(row=>row.video_time_seconds<=time);
  if (!sample || (!live && time-sample.video_time_seconds>=documentData.interval_emulator_frames/60)) {
    clearPanels('해당 시점의 관측 데이터 없음 · 이전 표본을 이어 붙이지 않습니다.');return;
  }
  const fragment=document.createDocumentFragment();
  fragment.append(template.querySelector('.network').cloneNode(true),template.querySelector('.detail').cloneNode(true));
  const find=selector=>fragment.querySelector(selector);
  for (const [index, uri] of sample.pixels.entries()) {
    const image=new Image();image.src=uri;image.alt=`기록된 정책 입력 프레임 ${index+1}`;find('#inputs').append(image);
    find('#frame-choice').add(new Option('프레임 '+(index+1),String(index)));
  }
  find('#input-contract').textContent=`실제 기록된 ${sample.pixel_shape[2]}×${sample.pixel_shape[1]} · ${sample.pixel_shape[0]}프레임 · 오래된 순 · 정책 내부 /255`;
  find('#argmax').textContent=sample.argmax;
  find('.action-note').textContent=(sample.selection==='deterministic'?'최대 확률 행동 선택':'범주 분포에서 샘플링')+' · 실제 선택 '+sample.action;
  for (const [index,p] of sample.probabilities.entries()) {
    const row=document.createElement('div');row.className='action-row'+(index===sample.action?' selected':'');
    const label=document.createElement('span');label.className='action-label';label.textContent=index+' · '+(sample.actions[index].join(' + ')||'NOOP')+(index===sample.action?' · 선택':'');
    const value=document.createElement('span');value.className='action-value';value.textContent=(p*100).toFixed(1)+'%';
    const meter=document.createElement('meter');meter.min=0;meter.max=1;meter.value=p;meter.setAttribute('aria-label',label.textContent);
    row.append(label,value,meter);find('#actions').append(row);
  }
  if (!live) inspector.history=[];
  for (const item of live?[sample]:documentData.samples.filter(row=>row.video_time_seconds<=sample.video_time_seconds && sample.video_time_seconds-row.video_time_seconds<=30)) {
    inspector.accept({graph:documentData.graph,generation:1,sequence:item.sequence,observer:{requested_hz:3},
      sample:{...item,observed_unix:live?item.observed_unix:item.video_time_seconds}});
  }
  inspector.render(fragment,documentData.graph,sample,draw);
  find('#history-note').textContent=live?'실제 최신 평가 표본 · 누락 구간은 끊음':'기록된 게임 시간 기준 최근 30초 · 최대 90개 · 누락은 끊음 · 영상 탐색 시 이력 재구성';
  const gameImage=new Image();gameImage.src=sample.raw;gameImage.alt='고정 정책 평가의 실제 행동 선택 전 게임 화면';
  try { await Promise.all([...fragment.querySelectorAll('img[src]'),...(live?[gameImage]:[])].map(image=>image.decode())); }
  catch { if(currentVersion===renderVersion)clearPanels('기록된 입력 이미지를 읽지 못했습니다. 평가본을 다시 선택하세요.');return; }
  if(currentVersion!==renderVersion)return;
  const focus=document.activeElement,focusID=focus?.id,focusNode=focus?.dataset.node;
  for(const [index, detail] of [...$('#replay-panels').querySelectorAll('details')].entries()) {const next=fragment.querySelectorAll('details')[index];if(next)next.open=detail.open;}
  $('#replay-panels').replaceChildren(fragment);
  if(live)$('#live-image').replaceChildren(gameImage);
  $('#sample-identity').textContent=sample.sample_id+' · snapshot '+sample.snapshot_id;
  $('#sample-time').textContent=`영상 ${sample.video_time_seconds.toFixed(3)}초 · 에피소드 ${sample.episode} / seed ${sample.seed} · emulator frame ${sample.emulator_frames} · step ${sample.step} · 표본 시각 ${new Date(sample.observed_unix*1000).toLocaleString('ko-KR')}`;
  $('#replay-status').textContent=live?`고정 평가 관측 · ${documentData.status==='running'?'실행 중':'종료 · 마지막 표본'} · 현재 학습과 별도 스트림`:'평가 기록 영상 · 해당 시점 직전의 실제 관측 표본';
  if(record.report.replay?.error)$('#replay-status').textContent+=' · 기록 오류: '+record.report.replay.error;
  if(focusID)document.getElementById(focusID)?.focus({preventScroll:true});
  else if(focusNode)document.querySelector(`#circuit [data-node="${focusNode}"]`)?.focus({preventScroll:true});
}
async function fetchSelection(selectionVersion) {
  const live=mode()==='live';
  try {
    const response=await fetch((live?'/api/evaluation-live?':'/api/replay?')+new URLSearchParams({run:selectedRun,evaluation:selectedEvaluation}),{signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error('Replay unavailable');
    const result=await response.json();
    if(selectionVersion!==version)return;
    record=result;
    $('#replay-identity').textContent='Run '+selectedRun+' · snapshot '+result.report.snapshot_id+' · protocol '+result.report.protocol_id+' · 평가 '+selectedEvaluation;
    $('#replay-contract').textContent=result.document?`첫 에피소드 · 최대 ${result.document.maximum_samples}개 · ${result.document.graph.nodes.length}개 실제 노드 · ${result.document.graph.edges.length}개 연결. 마지막 tanh 상태 벡터의 산술평균 · 생물학적 전압이 아닙니다.`:'이 평가에는 회로 관측 이력이 없습니다.';
    if(result.document?.graph.applicable===false)$('#replay-contract').textContent='CNN PPO · 회로 해당 없음 · 첫 에피소드의 실제 정책 입력과 행동만 기록합니다. 최대 '+result.document.maximum_samples+'개 표본.';
    if(!live && result.report.videos[0]) {
      video.src='/api/video?'+new URLSearchParams({run:selectedRun,file:'evaluations/'+selectedEvaluation+'/'+result.report.videos[0].file});
    }
    await draw();
  } catch {if(selectionVersion===version)clearPanels('관찰 기록을 읽지 못했습니다. 파일 누락·손상 여부를 확인하고 목록을 새로고침하세요.');}
  if(live && selectionVersion===version)timer=setTimeout(()=>fetchSelection(selectionVersion),400);
}
function selectEvaluation() {
  ++version;++renderVersion;clearTimeout(timer);record=null;inspector.reset();
  selectedEvaluation=$('#evaluation-choice').value;
  video.pause();video.removeAttribute('src');video.load();
  video.hidden=mode()==='live';$('#live-image').hidden=mode()!=='live';
  $('#playback-mode').textContent=mode()==='live'?'고정 평가의 최신 관측':'평가 기록 영상';
  clearPanels('선택한 평가의 관찰 기록을 불러오는 중입니다.');
  if(selectedEvaluation)fetchSelection(version);
  else clearPanels('평가 기록 없음 · 평가 실행 후 목록을 새로고침하세요.');
}
async function loadEvaluations() {
  const current=++version;++renderVersion;clearTimeout(timer);video.pause();video.removeAttribute('src');video.load();record=null;
  clearPanels('평가 목록을 불러오는 중입니다.');
  selectedRun=$('#run-choice').value;
  $('#compare-link').href='/compare?'+new URLSearchParams({run:selectedRun});
  try {
    const response=await fetch('/api/evaluation-streams?'+new URLSearchParams({run:selectedRun}),{signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error('List unavailable');
    const result=await response.json();if(current!==version)return;
    const select=$('#evaluation-choice');select.replaceChildren();
    for(const row of result.evaluations)select.add(new Option((row.snapshot_id==='initial'?'초기':row.snapshot_updates+' 갱신')+' · '+new Date(row.created_unix*1000).toLocaleString('ko-KR')+' · '+row.status,row.evaluation_id));
    const requested=new URLSearchParams(location.search).get('evaluation');
    if(result.evaluations.some(row=>row.evaluation_id===requested))select.value=requested;
    else if(requested) {
      select.selectedIndex=-1;selectedEvaluation='';
      clearPanels('평가 기록 없음 · 정리되었거나 누락된 평가입니다. 남아 있는 평가를 목록에서 선택하세요.');
      return;
    }
    selectEvaluation();
  } catch {if(current===version)clearPanels('평가 목록을 읽지 못했습니다. 로컬 서버 연결을 확인하고 새로고침하세요.');}
}
for(const event of ['timeupdate','seeked','loadeddata','pause'])video.addEventListener(event,()=>{if(mode()==='recorded')draw();});
video.addEventListener('seeking',()=>clearPanels('탐색 중 · 해당 영상 시점의 관측을 기다립니다.'));
video.addEventListener('error',()=>clearPanels('영상 파일을 읽지 못했습니다. 파일을 확인하고 평가본을 다시 선택하세요.'));
$('#mode-choice').addEventListener('change',selectEvaluation);
$('#evaluation-choice').addEventListener('change',selectEvaluation);
$('#run-choice').addEventListener('change',loadEvaluations);
$('#reload').addEventListener('click',loadEvaluations);
(async()=>{
  try {
    const [catalogResponse, templateResponse]=await Promise.all([fetch('/api/catalog'),fetch('/')]);
    if(!catalogResponse.ok||!templateResponse.ok)throw new Error('Load failed');
    const catalog=await catalogResponse.json();
    template=new DOMParser().parseFromString(await templateResponse.text(),'text/html').querySelector('#sample-template').content;
    for(const run of catalog.runs)$('#run-choice').add(new Option(run.label+' · '+run.run_id,run.run_id));
    const requested=new URLSearchParams(location.search).get('run');
    $('#run-choice').value=catalog.runs.some(run=>run.run_id===requested)?requested:catalog.default_run;
    await loadEvaluations();
  } catch {clearPanels('화면을 준비하지 못했습니다. 로컬 서버 연결을 확인하고 페이지를 새로고침하세요.');}
})();
