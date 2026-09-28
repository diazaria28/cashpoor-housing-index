const tasks = [
  {key:'luas', title:'Luas bangunan', hint:'Foto tampak rumah secara utuh', icon:'⌂'},
  {key:'atap', title:'Jenis atap', hint:'Pastikan material atap terlihat jelas', icon:'⌃'},
  {key:'dinding', title:'Dinding', hint:'Foto bidang dinding utama', icon:'▥'},
  {key:'lantai', title:'Lantai', hint:'Arahkan kamera ke permukaan lantai', icon:'▦'},
  {key:'air', title:'Sumber air', hint:'Meter PDAM, pompa, sumur, atau kemasan', icon:'◉'},
  {key:'wc', title:'WC', hint:'Pastikan bentuk WC terlihat penuh', icon:'◫'},
];

const state = {config:null, files:{}, results:{}, choices:{}, scores:{}};
const cards = document.querySelector('#cards');
const $ = s => document.querySelector(s);

function toast(message){const el=$('#toast');el.textContent=message;el.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>el.classList.remove('show'),3000)}
function optionMarkup(task){return (state.config?.kategori?.[task]||[]).map(x=>`<option value="${x.value}" data-score="${x.skor}">${x.nama} — skor ${x.skor}</option>`).join('')}

function renderCards(){
  cards.innerHTML=tasks.map((t,i)=>`<article class="card" data-task="${t.key}">
    <div class="card-head"><div><h3>${t.title}</h3><span class="card-sub">${t.hint}</span></div><span class="step">${i+1}</span></div>
    <label class="photo-zone" for="file-${t.key}"><input id="file-${t.key}" type="file" accept="image/*" capture="environment" hidden><img class="preview hidden" alt="Pratinjau ${t.title}"><span class="placeholder"><span style="font-size:2rem">${t.icon}</span><b>Ambil atau unggah foto</b><small>JPG, PNG, WEBP</small></span></label>
    <div class="card-body"><div class="actions"><button class="primary analyze" disabled>Analisis foto</button><button class="secondary clear" title="Hapus foto">Hapus</button></div>
    <div class="result empty">Belum dianalisis.</div>
    ${t.key==='luas'?'<div class="correction"><label>Perkiraan luas AI (m²)<input class="area-output" type="number" min="0" step="0.1" readonly></label></div>':`<div class="correction"><label>Hasil / koreksi petugas<select class="choice"><option value="">Belum dipilih</option>${optionMarkup(t.key)}</select></label></div>`}
    </div></article>`).join('');
  bindCards();
}

function bindCards(){
  document.querySelectorAll('.card').forEach(card=>{
    const task=card.dataset.task,file=card.querySelector('input[type=file]'),preview=card.querySelector('.preview'),placeholder=card.querySelector('.placeholder'),button=card.querySelector('.analyze');
    file.addEventListener('change',()=>{const selected=file.files[0];if(!selected)return;state.files[task]=selected;delete state.results[task];delete state.scores[task];delete state.choices[task];preview.src=URL.createObjectURL(selected);preview.classList.remove('hidden');placeholder.classList.add('hidden');button.disabled=false;card.querySelector('.result').className='result empty';card.querySelector('.result').textContent='Foto siap dianalisis.';const choice=card.querySelector('.choice');if(choice)choice.value='';updateSummary()});
    card.querySelector('.clear').addEventListener('click',()=>{file.value='';delete state.files[task];delete state.results[task];delete state.scores[task];delete state.choices[task];preview.classList.add('hidden');placeholder.classList.remove('hidden');button.disabled=true;card.querySelector('.result').className='result empty';card.querySelector('.result').textContent='Belum dianalisis.';const choice=card.querySelector('.choice');if(choice)choice.value='';const area=card.querySelector('.area-output');if(area)area.value='';updateSummary()});
    button.addEventListener('click',()=>analyze(task,card,button));
    const choice=card.querySelector('.choice');if(choice)choice.addEventListener('change',()=>{const opt=choice.selectedOptions[0];state.choices[task]=choice.value||null;state.scores[task]=choice.value?Number(opt.dataset.score):null;updateSummary()});
  });
}

async function analyze(task,card,button){
  const file=state.files[task];if(!file)return;
  if(task==='luas' && (!Number($('#members').value)||Number($('#members').value)<1)){toast('Isi jumlah anggota keluarga terlebih dahulu.');return}
  const original=button.textContent;button.disabled=true;button.innerHTML='<span class="spinner"></span>Menganalisis';
  const form=new FormData();form.append('image',file);if(task==='luas')form.append('members',$('#members').value);
  try{
    const res=await fetch(task==='luas'?'/api/analyze/luas':`/api/analyze/component/${task}`,{method:'POST',body:form});const data=await res.json();if(!res.ok)throw new Error(data.detail||'Analisis gagal');state.results[task]=data;
    if(task==='luas'){const p=data.prediksi||{};card.querySelector('.area-output').value=p.perkiraan_luas_m2??'';state.scores.luas=data.skor;state.choices.luas=p.perkiraan_luas_m2;setResult(card,'warn',`Perkiraan <b>${Number(p.perkiraan_luas_m2).toFixed(1)} m²</b> · ${Number(p.luas_per_orang_m2).toFixed(1)} m²/orang · skor <b>${data.skor}</b><br><small>Estimasi visual, wajib diverifikasi petugas.</small>`)}
    else if(data.status==='foto_tidak_sesuai'){state.scores[task]=null;state.choices[task]=null;card.querySelector('.choice').value='';setResult(card,'bad',`<b>Foto ditolak.</b> Lebih menyerupai ${data.validasi_foto?.komponen_terdeteksi||'objek lain'}. Ambil foto ulang.`)}
    else{const choice=card.querySelector('.choice');choice.value=data.hasil;state.choices[task]=data.hasil;state.scores[task]=data.skor;const warning=data.status==='perlu_verifikasi';const hybridInfo=task==='dinding'?`<br><small>Grounding DINO: ${data.grounding_dino_dipanggil?'dipakai untuk memeriksa campuran':'tidak diperlukan'}${data.alasan_fusi?` · ${data.alasan_fusi}`:''}</small>`:'';setResult(card,warning?'warn':'good',`${warning?'Perlu verifikasi':'Foto diterima'} · <b>${data.nama}</b> · skor <b>${data.skor}</b><br><small>Keyakinan ${Math.round((data.keyakinan_relatif||0)*100)}% · ${Number(data.durasi_total_detik).toFixed(3)} detik</small>${hybridInfo}`)}
  }catch(e){setResult(card,'bad',`<b>Gagal:</b> ${e.message}`);toast(e.message)}finally{button.disabled=false;button.textContent=original;updateSummary()}
}

function setResult(card,type,html){const el=card.querySelector('.result');el.className=`result ${type}`;el.innerHTML=html}
function updateSummary() {
  const values = tasks.map((task) => state.scores[task.key]);

  const completed = values
    .filter(Number.isFinite)
    .length;

  const total = values
    .filter(Number.isFinite)
    .reduce((current, score) => current + score, 0);

  const percentage = Math.max(
    0,
    Math.min(100, (total / 18) * 100)
  );

  $("#totalScore").textContent = total;

  const donut = $("#scoreDonut");

  if (donut) {
    donut.style.setProperty(
      "--score-progress",
      `${percentage}%`
    );

    donut.setAttribute(
      "aria-label",
      `Skor sementara ${total} dari 18`
    );
  }

  $("#completedCount").textContent = completed;

  $("#progressBar").style.width =
    `${(completed / 6) * 100}%`;

  if (completed === 6) {
    $("#summaryTitle").textContent =
      `Total skor ${total} dari 18`;

    $("#summaryText").textContent =
      "Semua komponen telah dinilai. Periksa kembali sebelum menyimpan.";
  } else {
    $("#summaryTitle").textContent =
      `${6 - completed} komponen belum selesai`;

    $("#summaryText").textContent =
      "Skor akan diperbarui setiap kali hasil AI diterima atau dikoreksi.";
  }

  $("#saveButton").disabled =
    completed !== 6 ||
    !$("#houseId").value.trim() ||
    !Number($("#members").value);
}
async function save(){const payload={id_rumah:$('#houseId').value.trim(),jumlah_anggota:Number($('#members').value),perkiraan_luas_m2:state.choices.luas,pilihan_petugas:Object.fromEntries(tasks.filter(t=>t.key!=='luas').map(t=>[t.key,state.choices[t.key]])),skor:{...state.scores},total:Object.values(state.scores).reduce((a,b)=>a+b,0),maksimum:18,hasil_ai:state.results};try{const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await res.json();if(!res.ok)throw new Error(data.detail||'Gagal menyimpan');const link=$('#downloadLink');link.href=`/api/download/${encodeURIComponent(data.filename)}`;link.classList.remove('hidden');toast('Hasil berhasil disimpan.')}catch(e){toast(e.message)}}

async function init(){try{state.config=await fetch('/api/config').then(r=>r.json());renderCards()}catch(e){toast('API tidak dapat dihubungi.')}['houseId','members'].forEach(id=>$(`#${id}`).addEventListener('input',updateSummary));$('#saveButton').addEventListener('click',save);$('#themeButton').addEventListener('click',()=>{document.documentElement.classList.toggle('dark');localStorage.theme=document.documentElement.classList.contains('dark')?'dark':'light'});if(localStorage.theme==='dark'||(!localStorage.theme&&matchMedia('(prefers-color-scheme: dark)').matches))document.documentElement.classList.add('dark')}
init();
