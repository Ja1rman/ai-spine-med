"use strict";

(() => {
  const root = new URL("./", window.location.href);
  const $ = id => document.getElementById(id);
  const copy = value => JSON.parse(JSON.stringify(value));
  const state = {catalog:null,episodes:[],uploads:[],record:null,draft:null,dirty:0,saved:0,saving:null,timer:null,storageKey:null,recovery:null,view:"front",tab:"intake",request:0,conflict:false,localEmergency:false,localUrgent:false,pendingSafetyEvents:[],signalKeys:new Set(),acknowledgements:{}};
  const sideNames = {left:"Слева",right:"Справа",midline:"По центру",bilateral:"С обеих сторон",unknown:"Сторона не уточнена"};
  const symptomNames = {pain:"Боль",numbness:"Онемение",weakness:"Субъективная слабость"};
  const statusNames = {draft:"Черновик",submitted:"Анкета зафиксирована",reviewed:"Проверка зафиксирована",partial:"Частично",completed:"Завершено",needs_review:"Нужна проверка",emergency_alert:"Срочный сигнал",emergency_stop:"Плановый осмотр остановлен"};
  const observationStatuses = [{value:"",label:"Не заполнено"},{value:"observed",label:"Выполнено"},{value:"not_tested",label:"Не выполнено"},{value:"refused",label:"Отказ"},{value:"unsafe_to_perform",label:"Небезопасно выполнять"},{value:"invalid",label:"Недостоверный результат"},{value:"unknown",label:"Неизвестно"}];

  function element(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function feedback(message, error=false) {
    $("feedback").textContent = message;
    $("feedback").className = "feedback" + (error ? " error" : "");
    $("feedback").hidden = !message;
  }
  async function api(route, payload) {
    const address = new URL(route, root);
    if (address.origin !== root.origin || !address.pathname.startsWith(root.pathname)) throw new Error("Недопустимый адрес.");
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(),30000);
    try {
      const options = {cache:"no-store",credentials:"same-origin",referrerPolicy:"no-referrer",signal:controller.signal};
      if (payload !== undefined) Object.assign(options,{method:"POST",headers:{"Content-Type":"application/json","X-PSP-Intent":"clinical-record"},body:JSON.stringify(payload)});
      const response = await fetch(address,options);
      let data;
      try { data = await response.json(); } catch { throw new Error("Сервер не вернул данные. Попробуйте ещё раз."); }
      if (!response.ok) { const error = new Error(data.message || "Не удалось сохранить запись."); error.status = response.status;error.code=data.error || data.code; throw error; }
      $("connection-status").textContent = "Хранилище доступно";
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("Сервер не ответил вовремя. Черновик остаётся в этой вкладке.");
      throw error;
    } finally { clearTimeout(timer); }
  }
  function date(value) {
    if (!value || Number.isNaN(new Date(value).getTime())) return "—";
    return new Date(value).toLocaleString("ru-RU",{dateStyle:"short",timeStyle:"short"});
  }
  function isDirty() { return state.draft && state.dirty !== state.saved; }
  function setSaveState() {
    const label = $("save-state");
    label.textContent = state.saving ? "Сохраняю…" : state.conflict ? "Конфликт версий" : state.pendingSafetyEvents.length ? "Тревожный ответ ожидает отправки" : isDirty() ? "Есть несохранённые изменения" : state.record ? "Сохранено · " + date(state.record.updated_at) : "";
    label.className = "save-state" + (state.conflict ? " error" : isDirty() ? " dirty" : "");
    $("save-record").disabled = Boolean(state.saving) || !state.draft;
  }
  function draftKey(id=state.record?.episode_id) { return state.storageKey + ":" + id; }
  function stash() {
    if (!state.storageKey || !state.draft) return;
    try {
      const unconfirmed = (state.localEmergency && state.record.safety?.level !== "E") || (state.localUrgent && !["E","U"].includes(state.record.safety?.level));
      const recovery=state.recovery && !isDirty() ? state.recovery : null;
      if (isDirty() || state.pendingSafetyEvents.length || unconfirmed || recovery) sessionStorage.setItem(draftKey(),JSON.stringify({episode_id:state.record.episode_id,base_revision:recovery?.base_revision ?? state.record.revision,draft:recovery?.draft || state.draft,localEmergency:state.localEmergency,localUrgent:state.localUrgent,pendingSafetyEvents:state.pendingSafetyEvents,signalKeys:[...state.signalKeys],saved_at:recovery?.saved_at || new Date().toISOString()}));
      else sessionStorage.removeItem(draftKey());
    } catch { feedback("Браузер не сохранил локальную копию. Для сохранения на сервере нажмите «Сохранить».",true); }
  }
  function changed(immediate=false) {
    state.dirty += 1;
    captureSignals();
    stash();
    setSaveState();
    renderSafety();
    clearTimeout(state.timer);
    state.timer = setTimeout(() => save().catch(() => {}),immediate ? 0 : 650);
  }
  function editable() {
    const result = {};
    for (const key of ["identity_confirmed","adult_confirmed","consent","upload_id","answers","pain_map","exam"]) if (state.draft[key] !== undefined) result[key] = copy(state.draft[key]);
    return result;
  }
  async function save(action="save", extras={}) {
    clearTimeout(state.timer);
    if (!state.record) return null;
    if (state.saving) {
      await state.saving;
      return save(action,extras);
    }
    if (state.conflict) throw new Error("Откройте эпизод заново и сверьте локальный черновик с серверной записью.");
    if(state.pendingSafetyEvents.length) await flushSafety();
    if (!isDirty() && action === "save") return state.record;
    const invalid = Array.from(document.querySelectorAll("#episode-panel input,#episode-panel textarea,#episode-panel select")).find(control => !control.checkValidity());
    if (invalid) { invalid.reportValidity(); throw new Error("Проверьте диапазоны и формат введённых значений."); }
    for(const [key,observation] of Object.entries(state.draft.exam.observations || {})) {
      if(observation.status !== "observed")continue;
      const test=state.catalog.tests.find(item => key.startsWith(item.id + "_"));
      if(test && !test.fields.some(field => ["mrc","force_n","value","angle","typical_symptoms"].includes(field.key) && observation[field.key] !== undefined)) {
        feedback("Укажите результат выполненной пробы «" + test.title + "» или выберите причину её невыполнения.",true);
        throw new Error("Для выполненной пробы требуется результат.");
      }
    }
    const version = state.dirty, id = state.record.episode_id;
    const payload = {...editable(),...extras,revision:state.record.revision,idempotency_key:crypto.randomUUID(),action};
    const operation = (async () => {
      try {
        const record = await api("api/psp/episodes/" + id,payload);
        if (state.record?.episode_id !== id) return record;
        state.record = record;
        if (version === state.dirty) { state.draft = copy(record); state.saved = version; }
        else state.draft.revision = record.revision;
        $("episode-upload").disabled=Boolean(record.upload_id);
        stash();
        renderSafety(); renderSummary(); renderStatus();
        feedback("");
        state.episodes = state.episodes.map(item => item.episode_id === id ? record : item);
        renderList();
        if (action !== "save") feedback(action === "submit_intake" ? "Анкета зафиксирована. Изменения создадут новую версию." : action === "review_exam" ? "Проверка осмотра сохранена вместе с автором и версией." : "Подтверждение сохранено. Исходный сигнал остаётся в истории.");
        return record;
      } catch (error) {
        if (error.status === 409 && error.code !== "identity_conflict") state.conflict = true;
        if(error.code === "identity_conflict" && state.record.upload_id) {state.draft.upload_id=state.record.upload_id;$("episode-upload").value=state.record.upload_id;$("episode-upload").disabled=true;}
        $("connection-status").textContent = error.status === 409 ? "Нужна сверка версий" : "Запись не подтверждена сервером";
        feedback(error.message + (error.status === 409 ? " Ваш черновик сохранён в этой вкладке. Обновите эпизод и восстановите нужные изменения." : " Черновик остаётся в этой вкладке."),true);
        stash(); renderSafety(); throw error;
      } finally {
        state.saving = null;
        setSaveState();
        renderSafety();
        if (isDirty() && !state.conflict && state.record?.episode_id === id && state.dirty !== version) state.timer = setTimeout(() => save().catch(() => {}),650);
      }
    })();
    state.saving = operation;
    setSaveState();
    return operation;
  }
  async function leaveCurrent() {
    if (state.saving) try { await state.saving; } catch {}
    if(state.pendingSafetyEvents.length)try{await flushSafety();}catch{return window.confirm("Тревожный ответ ещё не записан на сервере. Локальная копия сохранится в этой вкладке. Перейти к другому эпизоду?");}
    if (!isDirty()) return true;
    try { await save(); return !isDirty() || window.confirm("Остались несохранённые поля. Перейти к другому эпизоду?"); }
    catch { return window.confirm("Изменения не сохранены на сервере. Локальная копия останется в этой вкладке. Перейти к другому эпизоду?"); }
  }
  function renderList() {
    const list = $("episode-list"), search = $("episode-search").value.trim().toLocaleLowerCase("ru");
    list.replaceChildren();
    const items = state.episodes.filter(item => String(item.patient_code).toLocaleLowerCase("ru").includes(search));
    $("episode-count").textContent = items.length + " / " + state.episodes.length;
    for (const item of items) {
      const button = element("button","episode-option" + (item.episode_id === state.record?.episode_id ? " active" : ""));
      button.type = "button";
      button.append(element("strong","",item.patient_code),element("small","",date(item.updated_at)),element("small","",(statusNames[item.intake_status] || "Анкета: черновик") + " · " + (statusNames[item.exam_status] || "Осмотр: черновик")));
      button.addEventListener("click",() => openEpisode(item.episode_id).catch(error => feedback(error.message,true)));
      list.append(button);
    }
    if (!items.length) list.append(element("p","empty",search ? "Нет эпизодов с таким кодом." : "Создайте первый эпизод."));
  }
  async function refreshList() {
    const data = await api("api/psp/episodes");
    state.episodes = Array.isArray(data) ? data : data.episodes || [];
    renderList();
  }
  function fillUploads(select,selected="") {
    select.replaceChildren(new Option("Без МРТ", ""));
    const sorted = [...state.uploads].sort((a,b) => String(a.created_at_utc).localeCompare(String(b.created_at_utc)));
    sorted.forEach((upload,index) => select.append(new Option("Загрузка " + String(index+1).padStart(3,"0") + " · " + date(upload.created_at_utc) + " · " + (upload.n_slices || "—") + " срезов",upload.upload_id)));
    if (selected && !sorted.some(item => item.upload_id === selected)) select.append(new Option("Исследование недоступно",selected));
    select.value = selected || "";
  }
  async function openEpisode(id,force=false) {
    if (!force && !(await leaveCurrent())) return;
    const request = ++state.request;
    const record = await api("api/psp/episodes/" + id);
    if (request !== state.request) return;
    state.record = record; state.draft = copy(record); state.dirty = 0; state.saved = 0; state.conflict = false; state.localEmergency = false;state.localUrgent=false;state.pendingSafetyEvents=[];state.signalKeys=new Set();
    state.draft.answers ||= {}; state.draft.pain_map ||= []; state.draft.exam ||= {};
    state.draft.exam.observations ||= {}; state.draft.exam.notes ||= {}; state.draft.exam.safety_answers ||= {};
    $("create-panel").hidden = true; $("episode-panel").hidden = false;
    $("episode-title").textContent = record.patient_code;
    $("episode-meta").textContent = "Создан " + date(record.created_at) + " · версия " + record.revision;
    $("intake-confirmation").checked = record.answers?.Q29?.value === "yes";
    $("exam-confirmation").checked = false;
    renderIntake(); renderExam(); renderMap(); renderSafety(); renderSummary(); renderStatus(); renderList(); setSaveState();
    fillUploads($("episode-upload"),record.upload_id);
    $("episode-upload").disabled=Boolean(record.upload_id);
    $("recovery-panel").hidden = true; state.recovery = null;
    try {
      const stored = JSON.parse(sessionStorage.getItem(draftKey(id)) || "null");
      if (stored?.episode_id === id && stored.draft) {
        state.recovery = stored;
        state.localEmergency ||= Boolean(stored.localEmergency) || detectEmergency(stored.draft);
        state.localUrgent ||= Boolean(stored.localUrgent);
        state.pendingSafetyEvents=Array.isArray(stored.pendingSafetyEvents) ? stored.pendingSafetyEvents : [];
        state.signalKeys=new Set(stored.signalKeys || []);
        renderSafety();
        $("recovery-panel").hidden = false;
        $("recovery-description").textContent = "В этой вкладке есть несохранённая копия от " + date(stored.saved_at) + (stored.base_revision !== record.revision ? ". Серверная версия изменилась; проверьте данные перед сохранением." : ". Можно восстановить её и проверить перед сохранением.");
        if(state.pendingSafetyEvents.length)flushSafety().catch(error => feedback(error.message,true));
      }
    } catch {}
    const url = new URL(location.href); url.search = new URLSearchParams({episode:id}).toString(); history.replaceState(null,"",url);
  }
  function renderStatus() {
    if (!state.record) return;
    $("intake-status").textContent = statusNames[state.record.intake_status] || state.record.intake_status || "Черновик";
    $("exam-status").textContent = statusNames[state.record.exam_status] || state.record.exam_status || "Черновик";
    $("episode-meta").textContent = "Создан " + date(state.record.created_at) + " · версия " + state.record.revision;
  }
  function control(field,value,onChange,prefix) {
    const label = element("div","field" + (field.type === "textarea" || field.type === "multi" ? " wide" : ""));
    const id = prefix + "-" + field.key;
    const caption = element("label","",field.label); caption.htmlFor = id; label.append(caption);
    if (field.type === "multi") {
      const choices = element("div","choice-list"); choices.setAttribute("role","group"); choices.setAttribute("aria-label",field.label);
      let chosen = Array.isArray(value) ? [...value] : [];
      const inputs=[];
      for (const option of field.options || []) {
        const row = element("label","choice-option"), input = element("input"); input.type="checkbox"; input.value=option.value; input.checked=chosen.includes(option.value); inputs.push(input);
        input.addEventListener("change",() => {
          const exclusive = ["none","unknown","refused"];
          if (input.checked && exclusive.includes(input.value)) chosen = [input.value];
          else { chosen=chosen.filter(item => item !== input.value && !exclusive.includes(item)); if(input.checked) chosen.push(input.value); }
          inputs.forEach(item => { item.checked=chosen.includes(item.value); });
          onChange([...chosen]);
        });
        row.append(input,element("span","",option.label)); choices.append(row);
      }
      label.append(choices); return label;
    }
    let input;
    if (field.type === "choice") {
      input=element("select"); input.append(new Option("Не выбрано", ""));
      for(const option of field.options || []) input.append(new Option(option.label,option.value));
      input.value=typeof value === "boolean" ? value ? "yes" : "no" : value ?? "";
    } else {
      input=element(field.type === "textarea" ? "textarea" : "input");
      if(field.type !== "textarea") input.type=field.type === "number" ? "number" : "text";
      if(field.type === "number") { input.min=field.min ?? 0; input.max=field.max ?? 100; input.step=["force_n","angle","distance_m"].includes(field.key) ? "any" : "1"; }
      else input.maxLength=field.type === "textarea" ? 4000 : 500;
      input.value=value ?? "";
    }
    input.id=id;
    input.autocomplete="off";
    input.addEventListener(field.type === "choice" ? "change" : "input",() => {
      if (!input.checkValidity()) { input.reportValidity(); return; }
      onChange(input.value === "" ? undefined : field.type === "number" ? Number(input.value) : input.value);
    });
    label.append(input); return label;
  }
  function setAnswer(id,key,value,exam=false) {
    const target=exam ? state.draft.exam.safety_answers : state.draft.answers;
    target[id] ||= {};
    if (value === undefined || (Array.isArray(value) && !value.length)) delete target[id][key]; else target[id][key]=value;
    if(!exam && id !== "Q29")invalidateIntakeReview();
    if(id === "Q12") updateBranches();
    changed(Boolean(state.catalog.questions.find(question => question.id === id)?.safety || id === "Q13" || id === "Q12"));
  }
  function invalidateIntakeReview() {
    if(state.draft.answers.Q29?.value === "yes")state.draft.answers.Q29={value:"no"};
    $("intake-confirmation").checked=false;
  }
  function updateBranches() {
    for (const question of state.catalog.questions) {
      const box=document.getElementById("question-" + question.id), condition=question.show_if;
      if (!box || !condition) continue;
      const value=state.draft.answers[condition.question]?.[condition.field];
      box.hidden=!(condition.any || []).some(item => Array.isArray(value) ? value.includes(item) : value === item) && !(question.id === "Q13" && (Object.keys(state.draft.answers.Q13 || {}).length || state.draft.pain_map.some(mark => mark.type === "weakness")));
    }
  }
  function questionCard(question,exam=false) {
    const card=element("section","question-card"); card.id=(exam ? "exam-question-" : "question-") + question.id;
    const heading=element("h3","question-title"); heading.append(element("span","question-code",question.id),element("span","",question.title)); card.append(heading);
    if(question.type === "map") {
      const button=element("button","button quiet","Открыть карту симптомов →");button.type="button";button.addEventListener("click",() => { if(question.id === "Q11") $("map-type").value="numbness"; showTab("map"); }); card.append(button);return card;
    }
    const fields=element("div","question-fields"), source=exam ? state.draft.exam.safety_answers : state.draft.answers;
    for(const field of question.fields) fields.append(control(field,source[question.id]?.[field.key],value => setAnswer(question.id,field.key,value,exam),(exam ? "exam-" : "") + question.id));
    card.append(fields);return card;
  }
  function renderIntake() {
    const list=$("question-list");list.replaceChildren();
    const ordered=["Важные симптомы","Жалобы","Слабость и ходьба","Карта симптомов","Анамнез","Функция","Проверка"];
    for(const section of ordered) {
      const questions=state.catalog.questions.filter(question => question.section === section && question.id !== "Q29");if(!questions.length)continue;
      const group=element("div","question-section");group.append(element("h3","",section));
      questions.forEach(question => group.append(questionCard(question)));list.append(group);
    }
    updateBranches();
  }
  function renderExam() {
    $("exam-clinician").value=state.draft.exam.clinician || "";
    const notes=$("exam-notes"), noteGrid=element("div","exam-note-grid");notes.replaceChildren();
    for(const step of state.catalog.exam_steps || []) {
      const label=element("label","field"),caption=element("span","",step.id + " · " + step.title), input=element("textarea");input.maxLength=4000;input.value=state.draft.exam.notes[step.id] || "";input.setAttribute("aria-label",step.title);input.placeholder=["N00","N01"].includes(step.id) ? "Обязательно для фиксации проверки" : "Наблюдения, ограничения или причины пропуска";
      input.addEventListener("input",() => {state.draft.exam.notes[step.id]=input.value;changed();});label.append(caption,input);noteGrid.append(label);
    }
    notes.append(noteGrid);
    const safety=$("exam-safety-questions");safety.replaceChildren();
    state.catalog.questions.filter(q => state.catalog.safety_question_ids.includes(q.id)).forEach(question => safety.append(questionCard(question,true)));
    const tests=$("exam-test-list");tests.replaceChildren();let category=null,group=null;
    for(const test of state.catalog.tests || []) {
      if(test.category !== category) { category=test.category;group=element("section","test-category");group.append(element("h3","",category));tests.append(group); }
      const card=element("article","test-card"),heading=element("h4","test-title",test.id + " · " + test.title),sides=element("div","test-sides");card.append(heading,sides);
      for(const side of test.sides) {
        const key=test.id + "_" + side, obs=state.draft.exam.observations[key] || {};
        const box=element("div","test-side");box.append(element("h4","",sideNames[side] + " / пациента"));
        box.append(control({key:"status",label:"Выполнение",type:"choice",options:observationStatuses.filter(option => option.value)},obs.status,value => {
          if(!value) {delete state.draft.exam.observations[key];changed();renderExam();return;}
          const current=state.draft.exam.observations[key] ||= {};
          current.status=value;current.method="manual";
          if(value !== "observed") for(const field of test.fields) delete current[field.key];
          if(value === "invalid")current.quality="invalid";
          else if(current.quality === "invalid")current.quality=value === "observed" ? "valid" : "limited";
          changed();renderExam();
        },key));
        for(const field of test.fields) box.append(control(field,obs[field.key],value => {
          const current=state.draft.exam.observations[key] ||= {status:"observed",method:"manual",quality:"valid"};
          if(value === undefined)delete current[field.key];else current[field.key]=field.key === "pain_limited" ? value === "yes" : value;
          const hasValue=test.fields.some(f => ["mrc","force_n","value","angle","typical_symptoms"].includes(f.key) && current[f.key] !== undefined);
          current.status=hasValue ? "observed" : "not_tested";
          if(hasValue && current.quality === "invalid") current.quality="limited";
          changed();
          const select=document.getElementById(key + "-status");if(select)select.value=current.status;
        },key));
        box.append(control({key:"quality",label:"Качество результата",type:"choice",options:[{value:"valid",label:"Достаточное"},{value:"limited",label:"Есть ограничения"},{value:"invalid",label:"Недостоверно"}]},obs.quality,value => {
          const current=state.draft.exam.observations[key] ||= {status:"not_tested",method:"manual"};if(value)current.quality=value;else delete current.quality;
          if(value === "invalid") {current.status="invalid";test.fields.forEach(field => delete current[field.key]);changed();renderExam();}else changed();
        },key));
        for(const field of [{key:"reason",label:"Причина пропуска / ограничения",type:"text"},{key:"note",label:"Комментарий",type:"text"}]) box.append(control(field,obs[field.key],value => {const current=state.draft.exam.observations[key] ||= {status:"not_tested",method:"manual"};current[field.key]=value || "";changed();},key));
        sides.append(box);
      }
      group.append(card);
    }
    const bulk=element("div","summary-card"),reason=element("input"),button=element("button","button quiet","Указать причину для незаполненных проб");
    reason.id="bulk-missing-reason";reason.placeholder="Почему остальные пробы не выполнены";reason.maxLength=500;reason.setAttribute("aria-label","Общая причина невыполненных проб");button.type="button";
    button.addEventListener("click",() => {
      if(!reason.value.trim()) {reason.focus();feedback("Укажите причину невыполненных проб.",true);return;}
      for(const test of state.catalog.tests)for(const side of test.sides) {const key=test.id+"_"+side,obs=state.draft.exam.observations[key];if(!obs || (obs.status !== "observed" && !obs.reason)) state.draft.exam.observations[key]={...(obs || {}),status:obs?.status || "not_tested",reason:reason.value.trim(),method:"manual",quality:obs?.quality || "limited"};}
      changed();renderExam();feedback("Причина добавлена только к невыполненным пробам без объяснения. Выполненные результаты сохранены.");
    });
    bulk.append(element("h3","","Невыполненные пробы"),reason,button);tests.append(bulk);
  }
  function showTab(name) {
    state.tab=name;
    for(const button of $("module-tabs").querySelectorAll("button")) {const active=button.dataset.panel === name;button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active));$("panel-" + button.dataset.panel).hidden=!active;}
    if(name === "summary")renderSummary();
  }
  function detectEmergency(draft) {
    const sources=[draft?.answers || {},draft?.exam?.safety_answers || {}];
    return sources.some(answers => (state.catalog.safety_question_ids || []).some(id => answers[id]?.value === "yes") || (answers.Q13?.side === "bilateral" && ["new","progressive"].includes(answers.Q13?.onset) && (answers.Q13?.severity === "severe" || answers.Q13?.onset === "progressive")));
  }
  function localEmergency() {
    const triggered=detectEmergency(state.draft);
    state.localEmergency ||= triggered;
    return state.localEmergency;
  }
  function triggerKeys(draft) {
    const keys=[];
    for(const [source,answers] of [["intake",draft.answers || {}],["exam",draft.exam?.safety_answers || {}]]) {
      for(const id of state.catalog.safety_question_ids)if(answers[id]?.value === "yes")keys.push(source+":"+id+":E");
      if(answers.Q13?.side === "bilateral" && ["new","progressive"].includes(answers.Q13?.onset) && (answers.Q13?.severity === "severe" || answers.Q13?.onset === "progressive"))keys.push(source+":Q13:E");
      for(const id of ["Q12","Q20","Q21","Q22","Q23"])if(Array.isArray(answers[id]?.value) && answers[id].value.some(v => !["none","unknown","refused"].includes(v)))keys.push(source+":"+id+":U");
      if(answers.Q03?.value === "yes")keys.push(source+":Q03:U");
      if(answers.Q08?.course === "worsening")keys.push(source+":Q08:U");
    }
    if(draft.pain_map?.some(mark => mark.type === "weakness"))keys.push("map:weakness:U");
    return keys;
  }
  function captureSignals() {
    const keys=triggerKeys(state.draft),fresh=keys.filter(key => !state.signalKeys.has(key));
    state.localEmergency ||= keys.some(key => key.endsWith(":E"));state.localUrgent ||= keys.some(key => key.endsWith(":U"));
    if(!fresh.length)return;
    fresh.forEach(key => state.signalKeys.add(key));
    function riskAnswers(answers) {
      const result={};
      for(const key of ["Q03","Q08","Q12","Q13","Q15","Q16","Q17","Q18","Q19","Q20","Q21","Q22","Q23","Q24"]) {
        const value=answers[key];if(!value)continue;
        const fields=key === "Q08" ? ["course"] : key === "Q13" ? ["onset","side","severity"] : ["value"];
        const evidence={};fields.forEach(field => {if(value[field] !== undefined)evidence[field]=copy(value[field]);});
        if(Object.keys(evidence).length)result[key]=evidence;
      }
      return result;
    }
    state.pendingSafetyEvents.push({id:crypto.randomUUID(),occurred_at:new Date().toISOString(),answers:riskAnswers(state.draft.answers),exam_answers:riskAnswers(state.draft.exam.safety_answers),pain_map:copy(state.draft.pain_map.filter(mark => mark.type === "weakness"))});
  }
  async function flushSafety() {
    if(state.saving) {try{await state.saving;}catch{}return flushSafety();}
    if(!state.pendingSafetyEvents.length || !state.record)return;
    const events=copy(state.pendingSafetyEvents.slice(0,100)),ids=new Set(events.map(event => event.id)),id=state.record.episode_id;
    const operation=(async () => {
      try {
        const record=await api("api/psp/episodes/"+id,{revision:state.record.revision,idempotency_key:crypto.randomUUID(),action:"record_safety",safety_events:events});
        if(state.record?.episode_id !== id)return;
        state.record=record;state.draft.revision=record.revision;
        state.pendingSafetyEvents=state.pendingSafetyEvents.filter(event => !ids.has(event.id));
        stash();renderSafety();renderStatus();
      } catch(error) {
        if(error.status === 409)state.conflict=true;
        $("connection-status").textContent="Тревожный ответ не подтверждён сервером";
        feedback("Тревожный ответ сохранён в этой вкладке, но сервер не подтвердил запись. " + error.message,true);
        stash();renderSafety();throw error;
      } finally {state.saving=null;setSaveState();renderSafety();}
    })();
    state.saving=operation;setSaveState();return operation;
  }
  function localLevel() {
    if(localEmergency())return "E";
    const sources=[state.draft.answers,state.draft.exam.safety_answers];
    const positive=value => Array.isArray(value) && value.some(item => !["none","unknown","refused"].includes(item));
    if(state.localUrgent || state.draft.pain_map.some(mark => mark.type === "weakness") || sources.some(answers => ["Q20","Q21","Q22","Q23","Q12"].some(id => positive(answers[id]?.value)) || answers.Q03?.value === "yes" || answers.Q08?.course === "worsening"))return "U";
    const screen=["Q12","Q20","Q21","Q22","Q23",...state.catalog.safety_question_ids];
    if(screen.some(id => {const value=state.draft.answers[id]?.value;return value === undefined || ["unknown","refused"].includes(value) || (Array.isArray(value) && (!value.length || value.some(item => ["unknown","refused"].includes(item))));}))return "X";
    return "R";
  }
  function renderSafety() {
    if(!state.draft)return;
    const safety=state.record.safety || {},local=localEmergency(),draftLevel=localLevel();
    const level=(safety.alerts || []).some(alert => alert.level === "E") || local ? "E" : (safety.alerts || []).some(alert => alert.level === "U") || draftLevel === "U" ? "U" : draftLevel;
    $("safety-panel").className="safety-panel level-" + level;
    $("safety-title").textContent={E:"Нужна срочная очная медицинская оценка",U:"Нужна ускоренная оценка специалиста",X:"Важные сведения ещё не уточнены",R:"Ответы скрининга записаны"}[level] || "Проверьте важные ответы";
    const localMessages={U:"В ответах есть сведения, требующие оценки специалистом. Срок и маршрут определяет врач.",X:"Ответьте на вопросы о новых симптомах. Пропуски не считаются отрицательными ответами.",R:"Введённые ответы скрининга заполнены. Это не исключает заболевание."};
    $("safety-message").textContent=local ? "Зафиксирован тревожный ответ. Не ждите завершения анкеты или анализа МРТ, чтобы обратиться за срочной медицинской помощью." : level === safety.level && !isDirty() ? safety.message : localMessages[level] || safety.message;
    $("safety-delivery").textContent=isDirty() || state.saving ? "Последние изменения ещё не подтверждены сервером. Уведомление врачу автоматически не отправляется." : "Ответы сохранены. Автоматическая отправка уведомления врачу не настроена.";
    const list=$("safety-alerts");list.replaceChildren();
    for(const alert of safety.alerts || []) {
      const row=element("li","",alert.message + (alert.acknowledged ? " · получение подтверждено" : ""));
      if(!alert.acknowledged) {
        const detail=element("label","field"),note=element("textarea");note.maxLength=2000;note.value=state.acknowledgements[alert.id] || "";note.placeholder="Кто получил сигнал и какие действия предприняты";note.setAttribute("aria-label","Действия по предупреждению");note.addEventListener("input",() => {state.acknowledgements[alert.id]=note.value;});detail.append(note);row.append(detail);
        const button=element("button","text-button","Подтвердить получение");button.type="button";
        button.addEventListener("click",async () => {
          const clinician=state.draft.exam.clinician?.trim();
          if(!clinician) {feedback("Укажите специалиста на вкладке «Осмотр», затем подтвердите получение сигнала.",true);return;}
          const acknowledgement=state.acknowledgements[alert.id]?.trim();
          if(!acknowledgement) {feedback("Укажите предпринятые действия по предупреждению.",true);note.focus();return;}
          try {await save("acknowledge_alert",{alert_id:alert.id,acknowledged_by:clinician,acknowledgement});}catch(error){feedback(error.message,true);}
        });row.append(document.createTextNode(" "),button);
      }
      list.append(row);
    }
    $("submit-intake").disabled=level === "E";
    $("review-exam").disabled=level === "E";
  }
  function addMark(x,y,side,region="") {
    if(state.draft.pain_map.length >= 150) {feedback("На карте уже 150 отметок. Уберите лишние перед добавлением.",true);return;}
    const value=$("map-intensity");if(!value.checkValidity() || value.value === "") {value.reportValidity();value.focus();return;}
    state.draft.pain_map.push({id:crypto.randomUUID(),view:state.view,x,y,type:$("map-type").value,side,intensity:Number(value.value),region,body_version:state.catalog.body_version || "psp-body-2d-v1",recorded_at:new Date().toISOString()});
    invalidateIntakeReview();
    changed();renderMap();
  }
  function renderMap() {
    const marks=$("map-marks"),list=$("map-mark-list");marks.replaceChildren();list.replaceChildren();
    $("map-count").textContent=state.draft.pain_map.length + " отметок";
    $("map-front").classList.toggle("active",state.view === "front");$("map-back").classList.toggle("active",state.view === "back");
    $("map-front").setAttribute("aria-pressed",String(state.view === "front"));$("map-back").setAttribute("aria-pressed",String(state.view === "back"));
    $("map-screen-left").textContent=state.view === "front" ? "Правая сторона" : "Левая сторона";$("map-screen-right").textContent=state.view === "front" ? "Левая сторона" : "Правая сторона";
    for(const [index,mark] of state.draft.pain_map.entries()) {
      if(mark.view === state.view) {
        const circle=document.createElementNS("http://www.w3.org/2000/svg","circle");circle.setAttribute("cx",mark.x*300);circle.setAttribute("cy",mark.y*600);circle.setAttribute("r",11);circle.setAttribute("class","map-mark " + mark.type);const title=document.createElementNS("http://www.w3.org/2000/svg","title");title.textContent=symptomNames[mark.type] + " · " + sideNames[mark.side] + " · " + mark.intensity + "/10";circle.append(title);marks.append(circle);
      }
      const row=element("div","mark-row"),text=element("span","",(index+1) + ". " + symptomNames[mark.type] + " · " + sideNames[mark.side] + " · " + (mark.view === "front" ? "спереди" : "сзади") + " · " + mark.intensity + "/10"),remove=element("button","text-button","Убрать");remove.type="button";remove.setAttribute("aria-label","Убрать отметку " + (index+1));remove.addEventListener("click",() => {state.draft.pain_map=state.draft.pain_map.filter(item => item.id !== mark.id);invalidateIntakeReview();changed();renderMap();});row.append(text,remove);list.append(row);
    }
    $("undo-map").disabled=!state.draft.pain_map.length;$("clear-map").disabled=!state.draft.pain_map.length;
  }
  function listText(id,items,empty) {
    const list=$(id);list.replaceChildren();for(const item of items || [])list.append(element("li","",typeof item === "string" ? item : JSON.stringify(item)));if(!list.children.length && empty)list.append(element("li","",empty));
  }
  function renderSummary() {
    if(!state.record)return;
    const summary=state.record.summary || {};
    const missing=(summary.missing || []).map(value => {const question=state.catalog.questions.find(item => item.id === value);return question ? question.id + " · " + question.title : value;});
    listText("summary-findings",summary.findings,"В сводке пока нет сохранённых наблюдений.");listText("summary-missing",missing,"Список незаполненных полей пуст.");listText("summary-unperformed",summary.unperformed,"Невыполненные пробы не записаны.");listText("summary-limitations",summary.limitations,"");
    $("missing-caption").textContent="Незаполненные вопросы · " + missing.length;
    $("unperformed-caption").textContent="Невыполненные пробы · " + (summary.unperformed || []).length;
    if(isDirty())$("summary-limitations").prepend(element("li","","Сводка относится к последней серверной версии. Сохраните изменения для обновления."));
    const mri=$("mri-summary");mri.replaceChildren();
    const data=state.record.mri;
    if(!state.draft.upload_id) mri.append(element("p","meta","МРТ не связано. Анкета и осмотр доступны без него."));
    else if(data && data.available && data.upload_id === state.draft.upload_id) {
      const box=element("div","mri-findings");box.append(element("p","","Источник: исследовательский анализ МРТ"));
      const findings=data.patient_summary?.findings || [];const list=element("ul");findings.forEach(text => list.append(element("li","",text)));box.append(list);
      const link=element("a","","Открыть снимки и результаты МРТ ↗");
      const url=new URL(data.viewer_url || "./?view=uploads&upload=" + encodeURIComponent(data.upload_id),root);
      if(url.origin === root.origin && url.pathname.startsWith(root.pathname)) link.href=url.href;
      box.append(link);mri.append(box);
    } else mri.append(element("p","meta","Связанное МРТ недоступно или ещё не загружено. Клинические данные сохранены отдельно."));
    const historyList=$("episode-history");historyList.replaceChildren();
    for(const item of state.record.history || []) {
      const row=element("div","history-row","Версия " + (item.revision ?? "—") + " · " + (item.action || "изменение"));row.append(element("p","meta",date(item.recorded_at || item.occurred_at || item.created_at || item.timestamp)));historyList.append(row);
    }
    if(!historyList.children.length)historyList.append(element("p","meta","Полная история версий доступна в JSON-экспорте."));
  }
  async function initialize() {
    try {
      const bytes=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(root.pathname));
      state.storageKey="psp-draft-" + Array.from(new Uint8Array(bytes)).map(b => b.toString(16).padStart(2,"0")).join("").slice(0,24);
      const [catalog,uploads]=await Promise.all([api("api/psp/catalog"),api("api/uploads").catch(() => ({uploads:[]}))]);
      state.catalog=catalog;state.uploads=uploads.uploads || [];
      fillUploads($("create-upload"),new URLSearchParams(location.search).get("upload"));
      await refreshList();
      const id=new URLSearchParams(location.search).get("episode");
      if(id)await openEpisode(id);
      else if(state.episodes.length && !new URLSearchParams(location.search).has("upload"))await openEpisode(state.episodes[0].episode_id);
      else $("create-panel").hidden=false;
    } catch(error) {feedback(error.message,true);$("connection-status").textContent="Хранилище недоступно";$("new-episode").disabled=true;}
  }
  $("new-episode").addEventListener("click",async () => {if(!(await leaveCurrent()))return;state.request++;state.record=null;state.draft=null;$("episode-panel").hidden=true;$("create-panel").hidden=false;$("patient-code").focus();renderList();});
  $("episode-search").addEventListener("input",renderList);
  $("refresh-list").addEventListener("click",() => refreshList().catch(error => feedback(error.message,true)));
  $("create-form").addEventListener("submit",async event => {
    event.preventDefault();$("create-submit").disabled=true;
    try {
      const record=await api("api/psp/episodes",{patient_code:$("patient-code").value.trim(),identity_confirmed:$("create-identity").checked,adult_confirmed:$("create-adult").checked,consent:$("create-consent").checked,upload_id:$("create-upload").value || null,idempotency_key:crypto.randomUUID()});
      await refreshList();await openEpisode(record.episode_id,true);feedback("Эпизод создан. Ответы будут сохраняться по мере заполнения.");
    } catch(error) {feedback(error.message,true);}finally{$("create-submit").disabled=false;}
  });
  $("save-record").addEventListener("click",() => save().catch(error => feedback(error.message,true)));
  $("module-tabs").addEventListener("click",event => {const button=event.target.closest("button[data-panel]");if(button)showTab(button.dataset.panel);});
  $("intake-confirmation").addEventListener("change",() => {state.draft.answers.Q29={value:$("intake-confirmation").checked ? "yes" : "no"};changed();});
  $("submit-intake").addEventListener("click",async () => {if(!$("intake-confirmation").checked){feedback("Сначала просмотрите ответы и отметьте подтверждение.",true);return;}try{await save("submit_intake");}catch(error){feedback(error.message,true);}});
  $("exam-clinician").addEventListener("input",() => {state.draft.exam.clinician=$("exam-clinician").value;changed();});
  $("review-exam").addEventListener("click",async () => {if(!$("exam-confirmation").checked){feedback("Проверьте выполненные и пропущенные пробы и отметьте подтверждение.",true);return;}try{await save("review_exam");}catch(error){feedback(error.message,true);}});
  $("map-front").addEventListener("click",() => {state.view="front";renderMap();});$("map-back").addEventListener("click",() => {state.view="back";renderMap();});
  $("body-map").addEventListener("click",event => {if(!state.draft)return;const svg=$("body-map"),point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;const transform=svg.getScreenCTM();if(!transform)return;const result=point.matrixTransform(transform.inverse()),x=Math.min(1,Math.max(0,result.x/300)),y=Math.min(1,Math.max(0,result.y/600));const side=Math.abs(x-.5)<.055 ? "midline" : (state.view === "front" ? x<.5 ? "right" : "left" : x<.5 ? "left" : "right");addMark(x,y,side);});
  $("add-region").addEventListener("click",() => {const side=$("map-side").value,region=$("map-region").value,y={lumbar:.4,buttock:.49,thigh:.6,shin:.76,foot:.93}[region];if(side === "bilateral"){addMark(state.view === "front" ? .58 : .42,y,"left",region);addMark(state.view === "front" ? .42 : .58,y,"right",region);}else{const leftOnScreen=state.view === "front" ? side === "right" : side === "left";addMark(side === "midline" ? .5 : leftOnScreen ? .42 : .58,y,side,region);}});
  $("undo-map").addEventListener("click",() => {state.draft.pain_map.pop();invalidateIntakeReview();changed();renderMap();});$("clear-map").addEventListener("click",() => {if(window.confirm("Убрать все отметки карты в этом эпизоде?")){state.draft.pain_map=[];invalidateIntakeReview();changed();renderMap();}});
  $("episode-upload").addEventListener("change",() => {state.draft.upload_id=$("episode-upload").value || null;changed();});
  $("restore-draft").addEventListener("click",() => {
    if(!state.recovery)return;
    const recovery=state.recovery;state.recovery=null;
    state.draft=copy(recovery.draft);state.draft.revision=state.record.revision;state.dirty++;state.conflict=false;state.localEmergency ||= Boolean(recovery.localEmergency);state.localUrgent ||= Boolean(recovery.localUrgent);captureSignals();
    $("recovery-panel").hidden=true;renderIntake();renderExam();renderMap();renderSafety();renderSummary();fillUploads($("episode-upload"),state.draft.upload_id);$("intake-confirmation").checked=state.draft.answers?.Q29?.value === "yes";stash();setSaveState();feedback("Черновик восстановлен. Проверьте его, затем нажмите «Сохранить».");
  });
  $("discard-draft").addEventListener("click",async () => {
    try {
      if(state.pendingSafetyEvents.length)await flushSafety();
      if((state.localEmergency && state.record.safety?.level !== "E") || (state.localUrgent && !["E","U"].includes(state.record.safety?.level)))throw new Error("Сначала сохраните тревожные сведения. Их локальная копия остаётся доступна.");
      sessionStorage.removeItem(draftKey());state.recovery=null;$("recovery-panel").hidden=true;
    }catch(error){feedback(error.message,true);}
  });
  $("export-record").addEventListener("click",async () => {
    try {await save();const data=await api("api/psp/episodes/"+state.record.episode_id+"/export"),blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),link=element("a");link.href=url;link.download="psp-"+state.record.episode_id+".json";document.body.append(link);link.click();link.remove();setTimeout(() => URL.revokeObjectURL(url),1000);}catch(error){feedback(error.message,true);}
  });
  window.addEventListener("beforeunload",event => {if(isDirty() || state.pendingSafetyEvents.length){stash();event.preventDefault();event.returnValue="";}});
  initialize();
})();
