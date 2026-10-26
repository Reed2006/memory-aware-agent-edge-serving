"""Strict compatibility normalization; never invent a tool or an argument."""

import copy
import json
import uuid


def normalize_tool_message(message: dict, allowed: set[str]) -> tuple[dict, bool]:
    if message.get('tool_calls') or not message.get('content'):
        return message, False
    try:
        value = json.loads(message['content'])
    except (ValueError, TypeError):
        return message, False
    if (not isinstance(value, dict) or set(value) != {'name', 'arguments'}
            or value.get('name') not in allowed or not isinstance(value.get('arguments'), dict)):
        return message, False
    result = copy.deepcopy(message)
    result['content'] = None
    result['tool_calls'] = [{
        'id': 'normalized_' + uuid.uuid4().hex,
        'type': 'function', 'function': {
            'name': value['name'], 'arguments': json.dumps(value['arguments']),
        },
    }]
    return result, True
