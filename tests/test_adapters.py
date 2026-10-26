from phase1.adapters import normalize_tool_message


def test_exact_valid_tool_json_is_converted_without_changing_arguments():
    raw = {'role': 'assistant', 'content': '{"name":"get_order","arguments":{"order_id":"x"}}'}
    result, converted = normalize_tool_message(raw, {'get_order'})
    assert converted
    assert result['tool_calls'][0]['function']['arguments'] == '{"order_id": "x"}'
    assert raw['content'] is not None


def test_ordinary_text_unknown_tools_and_bad_json_are_not_repaired():
    for content in ['Hello', '{"name":"invented","arguments":{}}',
                    '{"name":"get_order","arguments":[]}', '```json\n{}\n```']:
        raw = {'role': 'assistant', 'content': content}
        result, converted = normalize_tool_message(raw, {'get_order'})
        assert not converted and result == raw


def test_existing_structured_call_is_unchanged():
    raw = {'tool_calls': [{'id': 'existing'}], 'content': None}
    assert normalize_tool_message(raw, {'x'}) == (raw, False)
