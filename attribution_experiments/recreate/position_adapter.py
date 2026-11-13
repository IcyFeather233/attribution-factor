"""Process-local position evidence handoff for the new ReCreate protocol only."""
import json

from attribution_experiments.common.position_evidence import INSTRUCTIONS, PROTOCOL


def instance_template(text):
    condition = "{% if s.get('provided_failure_analysis') and s.get('provided_failure_analysis').startswith('## Historical control/data-flow evidence') %}"
    if condition not in text:
        raise ValueError('ReCreate success-evidence template boundary changed')
    return INSTRUCTIONS + '\n' + text.replace(condition, "{% if s.get('provided_failure_analysis') %}").replace(
        '**Provided Failure Analysis (Trusted Evidence)**:', '**Historical indexed step evidence:**')


def synthesis_template(text):
    start = text.index("{% if mod.get('historical_flow_evidence') %}")
    end = text.index('{% endif %}', start) + len('{% endif %}')
    block = '''{% if mod.get('historical_flow_evidence') %}
**Historical position evidence accompanying this local proposal:**
{{ mod.historical_flow_evidence.text }}
{{ mod.historical_flow_evidence.handoff | tojson }}
Compare this ORIGINAL failed step with the local proposal and current rollout.
Preserve accepted/rejected task/message-index references in synthesis_summary.md,
not in the final task-agent scaffold. Additional flow graphs, when present, are
an explicit separate information condition; do not invent graph citations.
{% endif %}'''
    return INSTRUCTIONS + '\n' + text[:start] + block + text[end:]


def install(legacy, payloads):
    from recreate_agent.recreate_agent import ReCreateAgent
    from evolve_utils import evolution
    init = ReCreateAgent.__init__

    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.config.instance_template = instance_template(self.config.instance_template)

    ReCreateAgent.__init__ = initialize
    template = evolution.Template

    def render(source, *args, **kwargs):
        if '# Batch {{ batch_idx }} Synthesis' in source:
            source = synthesis_template(source)
        return template(source, *args, **kwargs)

    evolution.Template = render
    batch = legacy.run_batch_synthesis

    def synthesize(**kwargs):
        evidence = dict(kwargs.get('provided_flow_evidence') or {})
        for iid in kwargs['batch_results']:
            info = payloads[iid]['provided_information']
            if not info:
                continue
            evidence[iid] = {'text': json.dumps(info, ensure_ascii=False),
                             'handoff': {'protocol': PROTOCOL, 'task_id': iid,
                                         'stage': 'resolved original step with local proposal',
                                         'claim': 'delivery, not proven causal use'}}
        return batch(**{**kwargs, 'provided_flow_evidence': evidence})

    legacy.run_batch_synthesis = synthesize
