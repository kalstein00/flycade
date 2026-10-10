'use strict';
const $ = (selector) => document.querySelector(selector);
let catalog, group, requestVersion = 0;
const reasons = {completion:'완료', death:'사망', no_progress:'진행 없음', time_limit:'시간 제한', max_frames:'프레임 제한', external_limit:'프레임 제한', game_timeout:'게임 시간 초과'};
const number = (value) => Number(value).toLocaleString('ko-KR', {maximumFractionDigits:2});
for (const [side, title] of [['left','기준 평가'], ['right','비교 평가']]) {
  const panel = $('#comparison-panel').content.cloneNode(true);
  panel.querySelector('h2').textContent = title;
  for (const name of ['choice','identity','episodes','completion','video']) panel.querySelector('.'+name+', '+name).id = side+'-'+name;
  panel.querySelector('.circuit-note').id = side+'-circuit';
  panel.querySelector('section').id = side+'-panel';
  panel.querySelector('video').setAttribute('aria-label', title+' 플레이 영상');
  panel.querySelector('.choice').setAttribute('aria-label', title+' 평가본');
  $('#comparison').append(panel);
  $('#'+side+'-choice').addEventListener('change', () => render(side));
  $('#'+side+'-video').addEventListener('error', () => {
    $('#'+side+'-panel .video-status').textContent = '영상 파일을 읽지 못했습니다. 내역을 새로고침하거나 Run의 녹화 오류를 확인하세요.';
  });
}
function render(side) {
  const selected = $('#'+side+'-choice').value;
  const row = group.results.find(row => row.evaluation_id === (group[selected] || selected));
  const panel = $('#'+side+'-panel');
  const video = $('#'+side+'-video');
  video.pause(); video.removeAttribute('src'); video.load();
  if (!row) return;
  panel.querySelector('.mode').textContent = row.mode === 'deterministic_spectating' ? '기록 영상 · 결정론적 고정 정책 관전' : '평가 기록 영상 · 확률적 고정 정책';
  $('#'+side+'-identity').textContent = 'Run '+catalog.run_id+'\n모델 '+catalog.model_kind+'\nsnapshot '+row.snapshot_id+'\nprotocol '+row.protocol_id+'\n평가 '+row.evaluation_id+'\n'+new Date(row.created_unix*1000).toLocaleString('ko-KR')+' · '+row.snapshot_updates+' 갱신';
  $('#'+side+'-completion').textContent = row.completion_count+' / '+row.evaluation_count;
  panel.querySelector('.mean').textContent = number(row.mean_distance)+' px';
  panel.querySelector('.distribution').textContent = number(row.median_distance)+' · '+number(row.min_distance)+'~'+number(row.max_distance)+' px';
  const tbody = panel.querySelector('tbody'); tbody.replaceChildren();
  for (const episode of row.episodes) {
    const tr = document.createElement('tr');
    for (const value of [episode.seed, number(episode.distance), number(episode.reward), reasons[episode.reason] || episode.reason]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    tbody.append(tr);
  }
  const initial=group.results.find(result=>result.evaluation_id===group.initial);
  panel.querySelector('.outcome-change').textContent=initial?'초기 대비 평균 거리 '+number(row.mean_distance-initial.mean_distance)+' px · 완료율 '+number((row.completion_rate-initial.completion_rate)*100)+'%p · 사망 '+number((row.termination_counts.death||0)-(initial.termination_counts.death||0))+'회 ('+row.evaluation_count+'개 표본)':'같은 조건의 초기 평가 없음';
  const labels = [];
  if (row.snapshot_id === 'initial') labels.push('초기 정책');
  if (row.snapshot_id === group.best_snapshot) labels.push('최고 평가본');
  if (row.snapshot_id === group.latest_snapshot) labels.push('최근 평가본');
  if ((catalog.references?.pins || []).includes(row.snapshot_id)) labels.push('사용자 보존본');
  panel.querySelector('.protection').textContent = labels.length ? '보호 참조: '+labels.join(' · ') : '일반 평가 이력';
  const circuit = $('#'+side+'-circuit');
  circuit.replaceChildren();
  if (row.replay?.status === 'complete') {
    const link = document.createElement('a');
    link.textContent = catalog.model_kind==='cnn'?'입력·행동 이력과 함께 재생':'회로 이력과 함께 재생';
    link.href = '/replay?'+new URLSearchParams({run:catalog.run_id,evaluation:row.evaluation_id});
    circuit.append(link);
  } else circuit.textContent = row.replay?.error ? '회로 기록 실패 · '+row.replay.error : '회로 관측 데이터 없음';
  const clip = row.videos[0];
  panel.querySelector('.video-status').textContent = clip ? '첫 평가 에피소드의 짧은 영상 · 전체 평가 결과는 아래 표에 표시됩니다.' : '저장된 영상 없음 · 평가 결과는 아래 표에서 확인하세요.';
  if (clip) video.src = '/api/video?'+new URLSearchParams({run:catalog.run_id,file:'evaluations/'+row.evaluation_id+'/'+clip.file});
}
function selectProtocol() {
  group = catalog.protocols.find(g => g.protocol_id === $('#protocol-choice').value);
  if (!group) return;
  for (const side of ['left','right']) {
    const select = $('#'+side+'-choice'); select.replaceChildren();
    for (const [role, label] of [['initial','초기'],['latest','최근'],['best','최고']]) {
      if (group[role]) select.add(new Option(label,role));
    }
    for (const row of group.results) select.add(new Option(row.snapshot_updates+' 갱신 · '+new Date(row.created_unix*1000).toLocaleString('ko-KR')+((catalog.references?.pins || []).includes(row.snapshot_id) ? ' · 사용자 보존' : ''),row.evaluation_id));
    select.value = side === 'left' && group.initial ? 'initial' : 'best';
    render(side);
  }
}
async function load() {
  const version = ++requestVersion;
  for (const video of document.querySelectorAll('video')) { video.pause(); video.removeAttribute('src'); video.load(); }
  $('#comparison').hidden = true;
  $('#status').textContent = '평가 내역을 불러오는 중입니다.';
  try {
    const id = $('#run-choice').value;
    const response = await fetch('/api/evaluations?'+new URLSearchParams({run:id}), {signal:AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error('HTTP '+response.status);
    const result = await response.json();
    if (version !== requestVersion) return;
    catalog = result;
    $('#comparison-model').textContent=(catalog.model_kind==='cnn'?'CNN PPO · 회로 해당 없음':'커넥톰 PPO')+' · 총 예산 '+number(catalog.training_budget.total_updates)+' updates / '+number(catalog.training_budget.total_updates*catalog.training_config.rollout_steps)+'전이 · 파라미터 '+(catalog.parameter_count==null?'미기록':number(catalog.parameter_count)+'개');
    $('#live-link').href = '/?'+new URLSearchParams({run:id});
    const select = $('#protocol-choice'); select.replaceChildren();
    catalog.protocols.forEach((g,i) => select.add(new Option('조건 '+(i+1)+' · '+g.results.length+'회 평가 · '+g.protocol_id.slice(0,12),g.protocol_id)));
    $('#status').textContent = catalog.protocols.length ? '평가 기록 영상 모드 · 실시간 학습 화면과 분리되어 있습니다.' : '완료된 평가가 없습니다. 평가가 끝난 뒤 내역을 새로고침하세요.';
    $('#comparison').hidden = !catalog.protocols.length;
    selectProtocol();
  } catch (error) {
    if (version === requestVersion) $('#status').textContent = '평가 내역을 읽지 못했습니다. 로컬 서버 연결을 확인하고 내역을 새로고침하세요.';
  }
}
$('#protocol-choice').addEventListener('change', selectProtocol);
$('#run-choice').addEventListener('change', load);
$('#reload').addEventListener('click', load);
(async () => {
  try {
    const response = await fetch('/api/catalog', {signal:AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error('catalog');
    const result = await response.json();
    for (const run of result.runs) $('#run-choice').add(new Option(run.label+' · '+run.run_id,run.run_id));
    const requested = new URLSearchParams(location.search).get('run');
    $('#run-choice').value = result.runs.some(run => run.run_id === requested) ? requested : result.default_run;
    await load();
  } catch (error) { $('#status').textContent = 'Run 목록을 읽지 못했습니다. 로컬 서버 연결을 확인하고 페이지를 새로고침하세요.'; }
})();
