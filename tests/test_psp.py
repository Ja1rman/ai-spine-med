import hashlib
import json
import uuid

import pytest

from ai_spine_med.psp import BODY_VERSION, PSPError, PSPStore


def key():
    return uuid.uuid4().hex


@pytest.fixture
def store(tmp_path):
    return PSPStore(tmp_path / 'private-psp')


def create(store, **fields):
    return store.create({'idempotency_key': key(), 'patient_code': 'Пациент А',
                         'identity_confirmed': True, 'adult_confirmed': True, 'consent': True,
                         **fields})


def save(store, record, **fields):
    return store.save(record['episode_id'], {'revision': record['revision'],
                      'idempotency_key': key(), **fields})


def normal_answers():
    result = {q: {'value': 'no'} for q in ['Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q24']}
    result.update({q: {'value': ['none']} for q in ['Q12', 'Q20', 'Q21', 'Q22', 'Q23']})
    result['Q29'] = {'value': 'yes'}
    return result


def test_catalog_has_all_questions_and_bilateral_manual_tests(store):
    catalog = store.catalog()
    assert [q['id'] for q in catalog['questions']] == [f'Q{i:02}' for i in range(1, 30)]
    assert len(catalog['tests']) == 16
    assert all(t['sides'] == ['left', 'right'] for t in catalog['tests'])
    catalog['questions'].clear()
    assert len(store.catalog()['questions']) == 29


def test_durable_record_private_permissions_and_zero_not_missing(store):
    record = create(store)
    assert record['safety']['level'] == 'X'
    record = save(store, record, answers={'Q06': {'current': 0, 'worst': None}})
    reopened = PSPStore(store.root).get(record['episode_id'])
    assert reopened == record
    assert reopened['answers']['Q06'] == {'current': 0}
    assert (store.database.stat().st_mode & 0o777) == 0o600
    assert (store.root.stat().st_mode & 0o777) == 0o700
    assert 'Сейчас: 0' in ' '.join(record['summary']['findings'])


def test_revision_and_idempotency_prevent_overwrite(store):
    record = create(store)
    payload = {'revision': record['revision'], 'idempotency_key': key(),
               'answers': {'Q06': {'current': 7}}}
    saved = store.save(record['episode_id'], payload)
    assert store.save(record['episode_id'], payload) == saved
    with pytest.raises(PSPError, match='Ключ запроса'):
        store.save(record['episode_id'], {**payload, 'answers': {'Q06': {'current': 8}}})
    with pytest.raises(PSPError) as conflict:
        save(store, record, answers={'Q06': {'current': 1}})
    assert conflict.value.status == 409
    assert store.get(record['episode_id']) == saved


def test_emergency_immediate_and_latched_after_no_and_acknowledgement(store):
    record = create(store)
    record = save(store, record, answers={'Q17': {'value': 'yes'}})
    assert record['safety']['level'] == 'E'
    assert record['exam_status'] == 'emergency_stop'
    alert = record['safety']['alerts'][0]
    record = save(store, record, answers=normal_answers())
    assert record['safety']['level'] == 'E'
    record = save(store, record, action='acknowledge_alert', alert_id=alert['id'],
                  acknowledged_by='Врач А', acknowledgement='Получено, организована очная оценка')
    assert record['safety']['level'] == 'E'
    assert record['safety']['alerts'][0]['acknowledged'] is True
    assert len(record['safety']['alerts']) == 1
    assert PSPStore(store.root).get(record['episode_id'])['safety']['level'] == 'E'


def test_severe_bilateral_weakness_trumps_unknown_fields(store):
    record = create(store)
    record = save(store, record, answers={'Q13': {'onset': 'new', 'side': 'bilateral',
                                                'severity': 'severe'}})
    assert record['safety']['level'] == 'E'
    assert record['safety']['missing_questions']


def test_urgent_then_unknown_then_complete_are_distinct(store):
    record = create(store)
    answers = normal_answers()
    answers['Q15'] = {'value': 'unknown'}
    record = save(store, record, answers=answers)
    assert record['safety']['level'] == 'X'
    record = save(store, record, answers=normal_answers(), action='submit_intake')
    assert record['safety']['level'] == 'R'
    assert record['intake_status'] == 'submitted'
    answers['Q20'] = {'value': ['fever']}
    record = save(store, record, answers=answers)
    assert record['safety']['level'] == 'U'
    assert record['intake_status'] == 'draft'


def test_neuro_emergency_does_not_get_overwritten_by_normal_intake(store):
    record = create(store)
    record = save(store, record, answers=normal_answers(),
                  exam={'safety_answers': {'Q16': {'value': 'yes'}}})
    assert record['safety']['level'] == 'E'
    record = save(store, record, exam={'safety_answers': {'Q16': {'value': 'no'}}})
    assert record['safety']['level'] == 'E'


def test_episode_isolation_and_identity_binding(store):
    first = create(store)
    second = create(store, patient_code='Пациент Б')
    first = save(store, first, answers={'Q15': {'value': 'yes'}})
    assert store.get(second['episode_id'])['safety']['level'] == 'X'
    assert first['patient_id'] != second['patient_id']
    with pytest.raises(PSPError):
        save(store, first, patient_code='Пациент Б')
    first = save(store, first, upload_id='upload-' + 'a' * 32)
    with pytest.raises(PSPError):
        save(store, first, upload_id='upload-' + 'b' * 32)
    assert len(store.list_episodes()['episodes']) == 2


@pytest.mark.parametrize('answers', [
    {'Q06': {'current': -1}}, {'Q06': {'current': 11}}, {'Q06': {'current': True}},
    {'Q06': {'current': float('nan')}}, {'Q06': {'current': 1.5}},
    {'Q12': {'value': ['none', 'weakness']}}, {'Q15': {'value': 'normal'}},
    {'Q999': {'value': 'yes'}}, {'Q15': {'typo': 'yes'}},
])
def test_invalid_data_cannot_mutate_record(store, answers):
    record = create(store)
    with pytest.raises(PSPError):
        save(store, record, answers=answers)
    assert store.get(record['episode_id']) == record


def test_consent_required_to_collect_and_no_mri_needed(store):
    record = create(store, consent=False)
    with pytest.raises(PSPError):
        save(store, record, answers={'Q06': {'current': 7}})
    record = save(store, record, consent=True, answers=normal_answers(), action='submit_intake')
    assert record['intake_status'] == 'submitted'
    assert record['upload_id'] is None


def test_map_layers_stay_subjective_and_side_explicit(store):
    record = create(store)
    common = {'view': 'back', 'x': 0.4, 'y': 0.5, 'intensity': 7,
              'body_version': BODY_VERSION}
    record = save(store, record, pain_map=[
        {**common, 'id': 'point-a', 'side': 'left', 'type': 'pain'},
        {**common, 'id': 'point-b', 'side': 'right', 'type': 'numbness'},
    ])
    facts = store.export(record['episode_id'])['feature_records']
    assert [f['source'] for f in facts] == ['patient_self_report'] * 2
    assert [f['anatomy']['side'] for f in facts] == ['left', 'right']
    with pytest.raises(PSPError):
        save(store, record, pain_map=[{**common, 'id': 'bad', 'side': 'left',
                                    'type': 'pain', 'x': 2}])


def reviewed_exam(store):
    observations = {}
    for test in store.catalog()['tests']:
        for side in test['sides']:
            observations[test['id'] + '_' + side] = {'status': 'not_tested',
                                                   'reason': 'Не выполнено в этом осмотре'}
    observations['M05_right'] = {'status': 'observed', 'mrc': 4, 'force_n': 92,
                                 'pain_limited': True, 'quality': 'limited'}
    observations['M05_left'] = {'status': 'observed', 'mrc': 5}
    return {'clinician': 'Невролог А', 'notes': {'N00': 'Пациент и согласие проверены',
            'N01': 'Ограничения движений уточнены'}, 'observations': observations}


def test_manual_review_records_missing_reasons_mrc_force_and_amendments(store):
    record = create(store)
    with pytest.raises(PSPError) as error:
        save(store, record, exam={'clinician': 'Врач'}, action='review_exam')
    assert error.value.code == 'review_incomplete'
    record = save(store, record, exam=reviewed_exam(store), action='review_exam')
    assert record['exam_status'] == 'reviewed'
    obs = record['exam']['observations']['M05_right']
    assert obs['mrc'] == 4 and obs['force_n'] == 92 and obs['pain_limited'] is True
    record = save(store, record, answers={'Q06': {'current': 8}})
    assert record['exam_status'] == 'draft'
    history = store.export(record['episode_id'])['history']
    assert history[-2]['snapshot']['exam_status'] == 'reviewed'
    assert 'signature' not in history[-2]['snapshot']['exam']


def test_audit_chain_and_source_links_verify(store):
    record = create(store)
    record = save(store, record, answers={'Q06': {'current': 5}})
    exported = store.export(record['episode_id'])
    previous = ''
    for event in exported['history']:
        assert event['previous_hash'] == previous
        encoded = json.dumps(event['snapshot'], ensure_ascii=False, allow_nan=False,
                             sort_keys=True, separators=(',', ':'))
        assert hashlib.sha256((previous + event['action'] + encoded).encode()).hexdigest() == event['hash']
        previous = event['hash']
    fact = exported['feature_records'][0]
    assert fact['episode_id'] == record['episode_id']
    assert fact['source_record_ids'] == [fact['record_id']]
    assert fact['source'] == 'patient_self_report'


def test_connections_close_and_creation_is_idempotent(store):
    payload = {'idempotency_key': key(), 'patient_code': 'Пациент А'}
    record = store.create(payload)
    assert store.create(payload) == record
    assert len(store.list_episodes()['episodes']) == 1
    with store._connect() as connection:
        assert connection.execute('SELECT count(*) FROM episodes').fetchone()[0] == 1
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute('SELECT 1')


def test_observation_provenance_stable_until_that_observation_changes(store):
    first = create(store)
    first = save(store, first, exam=reviewed_exam(store), action='review_exam')
    obs = first['exam']['observations']['M05_right']
    second = create(store, patient_code='Пациент Б')
    second = save(store, second, exam=reviewed_exam(store))
    assert second['exam']['observations']['M05_right']['record_id'] != obs['record_id']
    first = save(store, first, exam=first['exam'])
    assert first['exam_status'] == 'reviewed'
    assert first['exam']['observations']['M05_right']['record_id'] == obs['record_id']
    assert first['exam']['observations']['M05_right']['recorded_at'] == obs['recorded_at']
    changed = first['exam']
    changed['observations']['M05_right']['mrc'] = 3
    changed['observations']['M05_right']['source'] = 'device_measurement'
    first = save(store, first, exam=changed)
    updated = first['exam']['observations']['M05_right']
    assert updated['record_id'] != obs['record_id']
    assert updated['source'] == 'clinician_exam'
    assert first['exam_status'] == 'draft'
    assert 'reviewed_at' not in first['exam']
    exported = store.export(first['episode_id'])
    fact = next(f for f in exported['feature_records'] if f['code'] == 'M05' and f['anatomy']['side'] == 'right')
    assert fact['record_id'] == updated['record_id']
    assert fact['recorded_at'] == updated['recorded_at']


def test_emergency_evidence_retains_original_answer(store):
    record = create(store)
    record = save(store, record, answers={'Q15': {'value': 'yes', 'onset_details': 'Сегодня'}})
    record = save(store, record, answers={'Q15': {'value': 'no'}})
    assert record['safety']['alerts'][0]['evidence']['value'] == 'yes'
    assert record['safety']['alerts'][0]['evidence']['onset_details'] == 'Сегодня'


def test_weakness_map_conflicting_with_none_is_not_routine(store):
    record = create(store)
    record = save(store, record, answers=normal_answers(), pain_map=[
        {'id': 'weak', 'view': 'back', 'type': 'weakness', 'side': 'left',
         'x': 0.5, 'y': 0.5, 'intensity': 2, 'body_version': BODY_VERSION}])
    assert record['safety']['level'] == 'U'
    assert 'Q13' in record['safety']['missing_questions']


def safety_event(**values):
    return {'id': key(), 'occurred_at': '2026-11-26T10:00:00+00:00', **values}


def test_offline_positive_then_final_negative_remains_emergency(store):
    record = create(store)
    event = safety_event(answers={'Q15': {'value': 'yes'}})
    record = save(store, record, answers=normal_answers(), safety_events=[event])
    assert record['answers']['Q15']['value'] == 'no'
    assert record['safety']['level'] == 'E'
    alert = record['safety']['alerts'][0]
    assert alert['evidence']['value'] == 'yes'
    assert alert['client_occurred_at'] == event['occurred_at']
    assert alert['source_event_id'] == event['id']
    assert record['safety_events'][0]['received_at']
    assert store.export(record['episode_id'])['safety_events'][0]['answers']['Q15']['value'] == 'yes'


def test_record_safety_bypasses_unrelated_in_progress_exam_and_is_deduplicated(store):
    record = create(store)
    record = save(store, record, exam=reviewed_exam(store), action='review_exam')
    old_exam = record['exam']
    old_answers = record['answers']
    event = safety_event(exam_answers={'Q17': {'value': 'yes'}})
    record = save(store, record, action='record_safety', safety_events=[event])
    assert record['exam'] == old_exam
    assert record['answers'] == old_answers
    assert record['safety']['level'] == 'E'
    assert record['exam_status'] == 'emergency_stop'
    record = save(store, record, action='record_safety', safety_events=[event])
    assert len(record['safety_events']) == 1
    assert len(record['safety']['alerts']) == 1


@pytest.mark.parametrize('change', [
    {'level': 'E'}, {'id': 'bad'}, {'occurred_at': '2026-11-26'},
    {'answers': {'Q15': {'value': 'normal'}}},
    {'exam_answers': {'Q28': {'value': 'hello'}}},
])
def test_safety_events_validate_raw_input_and_reject_fake_severity(store, change):
    record = create(store)
    event = {**safety_event(answers={'Q15': {'value': 'yes'}}), **change}
    with pytest.raises(PSPError):
        save(store, record, action='record_safety', safety_events=[event])
    assert store.get(record['episode_id']) == record


def test_record_safety_does_not_accept_whole_form_and_conflicting_events(store):
    record = create(store)
    event = safety_event(answers={'Q20': {'value': ['fever']}})
    with pytest.raises(PSPError):
        save(store, record, action='record_safety', safety_events=[event],
             exam={'observations': {'M01_left': {'status': 'observed'}}})
    record = save(store, record, action='record_safety', safety_events=[event])
    assert record['safety']['level'] == 'U'
    with pytest.raises(PSPError) as conflict:
        save(store, record, action='record_safety', safety_events=[
            {**event, 'answers': {'Q20': {'value': ['none']}}}])
    assert conflict.value.code == 'event_conflict'


def test_summary_separates_nonperformed_probes_and_uses_russian_labels(store):
    record = create(store)
    record = save(store, record, exam=reviewed_exam(store))
    summary = record['summary']
    assert any('Разгибание большого пальца (справа)' in item for item in summary['findings'])
    assert any('Ограничено болью?: Да' in item for item in summary['findings'])
    assert len(summary['unperformed']) == 30
    assert all('не выполнено' in item for item in summary['unperformed'])
    assert all('Не выполнено в этом осмотре' not in item for item in summary['findings'])
