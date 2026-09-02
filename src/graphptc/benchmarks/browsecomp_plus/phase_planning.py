"""BrowseComp-Plus phase-planning prompt suffix."""

PHASE_PLANNING_SUFFIX = """

Before a programmatic_tool_call, write one compact phase plan in the same assistant response:
<phase_plan>
stage_goal: the evidence goal for this research phase
parallel_subgoals: independent retrieval or verification subgoals that can be executed before seeing their results
return_condition: the semantic uncertainty or sufficient evidence that should return control to the model
</phase_plan>
This is not a separate action or turn. Immediately compile the phase plan into one Python research
program. Include all listed subgoals and their mechanical downstream retrieval, filtering, comparison,
or aggregation in that program when they do not require a new semantic judgment. Keep the plan short;
do not narrate chain-of-thought, step-by-step reasoning, or one action at a time. If research is not
needed, answer directly without a phase plan.
"""
