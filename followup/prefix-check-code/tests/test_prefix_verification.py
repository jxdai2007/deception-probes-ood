import pytest

from p04.extract_activations import response_token_spans
from p04.prefix_verification import (
    PrefixContractError, compare_token_contracts, validate_evaluation_reuse,
)


class Tokenizer:
    is_fast = True

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize,
                            continue_final_message=False):
        assert not (add_generation_prompt and continue_final_message)
        text = '^' + ''.join(m['role'][0].upper() + m['content'] + '|' for m in messages)
        if continue_final_message:
            text = text[:-1]
        if add_generation_prompt:
            text += 'G'
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert return_offsets_mapping
        return {'input_ids': self.encode(text, add_special_tokens=add_special_tokens),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}


@pytest.mark.parametrize('messages', [
    [{'role': 'user', 'content': 'Q'}],
    [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'Q'}],
    [{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': ''}],
])
def test_user_final_inputs_match_actual_frozen_extractor(messages):
    row = {'input_messages': messages, 'output_str': 'A'}
    actual = response_token_spans(Tokenizer(), [row], 32)[0]
    result = compare_token_contracts(Tokenizer(), row, max_tokens=32)
    assert result['frozen']['input_ids'] == actual['ids']
    assert result['frozen']['scored_span'] == [actual['prompt_len'], len(actual['ids'])]
    assert result['candidate'] == result['frozen']


def test_nonempty_assistant_prefix_is_continued_not_reopened():
    row = {'input_messages': [{'role': 'user', 'content': 'Q'},
                             {'role': 'assistant', 'content': 'P'},
                             {'role': 'assistant', 'content': ''}], 'output_str': 'A'}
    result = compare_token_contracts(Tokenizer(), row, max_tokens=32)
    actual = response_token_spans(Tokenizer(), [row], 32)[0]
    assert result['frozen']['input_ids'] == actual['ids']
    assert result['candidate']['input_ids'] == list(map(ord, '^UQ|APA'))
    start, end = result['candidate']['scored_span']
    assert result['candidate']['input_ids'][start:end] == [ord('A')]
    assert result['candidate']['input_ids'] != actual['ids']


def test_full_string_boundary_token_is_scored_and_disclosed():
    class MergingTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            i = text.index('PA')
            encoded['input_ids'][i:i + 2] = [999]
            encoded['offset_mapping'][i:i + 2] = [(i, i + 2)]
            return encoded

    row = {'input_messages': [{'role': 'user', 'content': 'Q'},
                             {'role': 'assistant', 'content': 'P'}], 'output_str': 'A'}
    result = compare_token_contracts(MergingTokenizer(), row, max_tokens=32)
    assert result['candidate']['input_ids'][-1] == 999
    assert result['candidate']['scored_span'] == [5, 6]
    assert result['candidate']['boundary_spanning_token'] is True


def test_token_contract_rejects_empty_or_truncated_scored_response():
    with pytest.raises(PrefixContractError, match='empty'):
        compare_token_contracts(Tokenizer(), {'input_messages': [], 'output_str': ''}, max_tokens=8)
    with pytest.raises(PrefixContractError, match='truncat'):
        compare_token_contracts(Tokenizer(), {'input_messages': [], 'output_str': 'abc'}, max_tokens=1)


def test_evaluation_reuse_requires_all_identity_fields():
    original = {'input_ids_sha256': 'a' * 64, 'scoring_mask_sha256': 'b' * 64,
                'layer': 7, 'runtime': 'x', 'row_mapping_sha256': 'c' * 64}
    assert validate_evaluation_reuse(original, dict(original))['eligible'] is True
    with pytest.raises(PrefixContractError, match='layer'):
        validate_evaluation_reuse(original, dict(original, layer=8))


@pytest.mark.parametrize('prefix', ['', 'P'])
def test_vendor_detect_metadata_matches_frozen_role_content_projection(prefix):
    row = {'input_messages': [
        {'role': 'user', 'content': 'Q', 'detect': False},
        {'role': 'assistant', 'content': prefix, 'detect': True},
    ], 'output_str': 'A'}
    result = compare_token_contracts(Tokenizer(), row, max_tokens=32)
    actual = response_token_spans(Tokenizer(), [row], 32)[0]
    projected = dict(row, input_messages=[
        {'role': item['role'], 'content': item['content']}
        for item in row['input_messages']
    ])
    assert result['frozen']['input_ids'] == actual['ids']
    assert result == compare_token_contracts(Tokenizer(), projected, max_tokens=32)


# Exact template from unsloth/Llama-3.2-3B-Instruct revision
# 006f5dcd1393c3add266de40994ba96225e9689d; inline for offline portability.
PINNED_CHAT_TEMPLATE = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'


@pytest.fixture
def pinned_template_tokenizer():
    from tokenizers import Tokenizer as Backend, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    # A complete byte vocabulary preserves differing header characters without
    # downloading the model vocabulary; the actual Jinja template is unchanged.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Backend(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend,
        bos_token='<|begin_of_text|>', eos_token='<|eot_id|>', pad_token='<|eot_id|>',
        chat_template=PINNED_CHAT_TEMPLATE)


def set_template_clock(monkeypatch, day):
    from datetime import datetime
    from transformers.utils import chat_template_utils
    class Clock:
        @classmethod
        def now(cls):
            return datetime.fromisoformat(day + 'T12:00:00')
    monkeypatch.setattr(chat_template_utils, 'datetime', Clock)


def date_test_rows():
    return [{'input_messages': [{'role': 'user', 'content': 'Question ' + 'q' * n},
                                {'role': 'assistant', 'content': 'Answer: '}],
             'output_str': 'yes indeed'} for n in (1, 5)]


def test_pinned_template_unbound_clock_changes_tokens_but_bound_contracts_are_stable(
        pinned_template_tokenizer, monkeypatch):
    from p04 import prefix_verification
    tokenizer = pinned_template_tokenizer
    row = date_test_rows()[0]
    # Exercise the old unbound behavior before requiring the production adapter.
    outputs = []
    for day in ('2026-08-24', '2026-09-04'):
        set_template_clock(monkeypatch, day)
        outputs.append(compare_token_contracts(tokenizer, row, max_tokens=2048))
    for branch in ('frozen', 'candidate'):
        assert outputs[0][branch]['input_ids'] != outputs[1][branch]['input_ids']
        assert outputs[0][branch]['scored_span'] == outputs[1][branch]['scored_span']
    adapter = getattr(prefix_verification, 'DateBoundTokenizer', None)
    bound = adapter(tokenizer, '2026-08-24') if adapter else tokenizer
    stable = []
    for day in ('2026-08-24', '2026-09-04'):
        set_template_clock(monkeypatch, day)
        stable.append(compare_token_contracts(bound, row, max_tokens=2048))
    assert stable[0] == stable[1] == outputs[0], 'protocol date must bind original AND candidate inputs'


@pytest.mark.parametrize('day', [None, '', '2026-8-24', '20260824', '2026-02-30', 20260824])
def test_date_adapter_rejects_missing_noncanonical_or_invalid_date(day):
    from p04.prefix_verification import DateBoundTokenizer
    with pytest.raises(ValueError, match='ISO date'):
        DateBoundTokenizer(Tokenizer(), day)


def test_date_adapter_delegates_calls_and_rejects_conflicting_template_override(
        pinned_template_tokenizer):
    from p04.prefix_verification import DateBoundTokenizer
    raw = pinned_template_tokenizer
    bound = DateBoundTokenizer(raw, '2026-08-24')
    assert bound.is_fast == raw.is_fast
    assert bound.encode('abc') == raw.encode('abc')
    assert bound('abc', return_offsets_mapping=True) == raw('abc', return_offsets_mapping=True)
    messages = date_test_rows()[0]['input_messages']
    expected = raw.apply_chat_template(messages, tokenize=False, date_string='24 Aug 2026')
    assert bound.apply_chat_template(messages, tokenize=False) == expected
    assert bound.apply_chat_template(messages, tokenize=False, date_string='24 Aug 2026') == expected
    for override in ('04 Sep 2026', None, '2026-08-24'):
        with pytest.raises(ValueError, match='date_string'):
            bound.apply_chat_template(messages, tokenize=False, date_string=override)
