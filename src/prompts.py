"""Prompt templates used by the KERNEL inference pipeline.
"""

from __future__ import annotations

from string import Template


SYSTEM_PROMPT = (
    "You are a careful emergency medicine assistant. "
    "You must follow the allowed test_tag list exactly and return JSON only."
)


KERNEL_FIRST_PASS_PROMPT = Template(
    """You are assisting emergency department laboratory test recommendation.
Return JSON only. Every recommended test_tag must exactly match one of the allowed test_tag categories.

Clinical reasoning frame:
A clinician first reads the patient's presentation and background, forms likely working diagnoses, considers disease mechanisms and related clinical concepts, then orders laboratory tests to evaluate or rule in/out those concerns.

Patient context:
${patient_context}

Working diagnostic categories suggested by the ICD diagnosis ranker:
${working_diagnoses}

${kg_section}
Clinician-curated ICD-to-test_tag knowledge:
${candidate_knowledge}
${disease_prior_section}
${context_rag_section}
${kg_rag_section}
How to use the evidence:
${kg_instruction}
- The clinician-curated links describe test_tag categories clinically relevant for predicted ICD diagnoses.
- Local training-set ordering patterns and retrieved similar cases come only from the training split. They are non-binding evidence, not labels for this patient.
- Choose tags supported by the patient context, working diagnoses, ${kg_concepts_phrase}clinician-curated links, and test-ordering evidence. Do not force a fixed number of recommendations.
- If laboratory testing does not appear clinically necessary for this patient, return an empty recommended_tags list and explain briefly in final_explanation.
${recall_relaxed_instruction}
Allowed test_tag categories:
${allowed_tags}

Output JSON schema:
{
  "recommended_tags": [
    {
      "test_tag": "one exact string from allowed test_tag categories",
      "rationale": "brief clinical reason using patient context, predicted diagnoses, ${rationale_evidence_phrase}and retrieval evidence",
      "linked_icd_codes": ["ICD codes supporting this tag"]
    }
  ],
  "final_explanation": "short paragraph explaining the overall recommendation"
}
"""
)


REASSESSMENT_PROMPT = Template(
    """You are performing a second-pass evidence reassessment for emergency department lab test_tag recommendations.
Return JSON only. Use only exact strings from the allowed test_tag categories.

Task:
- Review only the screened candidates listed below.
- Add an omitted candidate only when the structured evidence appears clinically applicable to this specific patient.
- Do not add a tag just because it is common in the local training data; connect it to the patient context or ${applicability_anchor}.
- For each candidate, separately judge evidence_strength and patient_applicability as strong, moderate, or weak.
- Add a candidate only when evidence_strength is at least ${min_evidence_strength} and patient_applicability is at least ${min_patient_applicability}.
${remove_instruction}
Patient context:
${patient_context}

Working diagnostic categories:
${working_diagnoses}

First-pass recommended test_tags:
${current_recommended_tags}

${kg_section}

Clinician-curated ICD-to-test_tag knowledge:
${candidate_knowledge}

Local ordering evidence:
${prior_evidence}

${context_rag_evidence}

${kg_rag_evidence}

Screened candidates for reassessment:
${screened_candidates}

Allowed test_tag categories:
${allowed_tags}

Output JSON schema:
{
  "tag_decisions": [
    {
      "test_tag": "one screened candidate tag",
      "decision": "${decision_options}",
      "evidence_strength": "strong, moderate, or weak",
      "patient_applicability": "strong, moderate, or weak",
      "rationale": "brief patient-specific reason"
    }
  ],
  "add_tags": ["exact allowed test_tag strings to add"],
  "remove_tags": ["exact allowed test_tag strings to remove"],
  "final_explanation": "brief explanation of the reassessment"
}
"""
)


FINAL_RATIONALE_PROMPT = Template(
    """You are writing a final clinician-facing rationale for emergency department laboratory test-category recommendations.
Return JSON only.

Write one concise but clinically informative paragraph in English, no more than 120 words.
Use the available word limit to provide sufficient patient-specific clinical reasoning rather than generic summary statements.

Do not use internal experiment or system terms such as "ICD", "ranker", "RAG", "KG", "local training data", "MedlinePlus", "prompt", or "model".
Do not mention dataset or institution names.
If historical evidence is relevant, use natural clinical wording such as "similar prior emergency visits".
Do not use method-style phrases such as "diagnosis-level evidence", "diagnosis-aware evidence", or "external lab-test knowledge".
Do not include ICD codes.

Focus primarily on the patient's current emergency-department presentation.
Mention only current findings that materially contribute to the diagnostic considerations or laboratory recommendations.
Do not repeatedly restate the full patient history.

Clearly distinguish the current presentation from chronic medical history, family history, and long-term risk factors.
Do not present chronic conditions, family history, or general health risks as current diagnoses.
Do not list past medical history separately merely because it is available.
When a chronic condition is clinically relevant, integrate it directly into the reasoning by explaining how it modifies the current diagnostic concern or supports a recommended test category.
Otherwise, omit it.

Use only clinically meaningful working diagnostic categories that help explain the current presentation.
Exclude administrative, encounter-related, screening-related, status-related, or other non-diagnostic categories.
Exclude overly broad categories when a more clinically meaningful category is available.
Do not mention categories such as "persons encountering health services in other circumstances" or similar encounter-based labels.
Do not add diagnostic categories merely to reach a target number.
Usually mention only 1-3 of the most clinically relevant diagnostic considerations.

Write the rationale as patient-specific clinical reasoning:
current findings -> relevant diagnostic considerations -> why the recommended test categories are appropriate for this patient.

Do not merely list diagnostic categories and test categories.
Explain the clinically meaningful relationship between the patient's findings, diagnostic considerations, and recommended test categories.

Do not justify a test category merely by restating its name or routine function, and do not enumerate the routine purpose of each recommended category.
For example, do not explain "Renal Function" only as "to evaluate renal/kidney function", "Liver Function" only as "to evaluate liver function", or "Electrolytes" only as "to evaluate electrolyte balance".
Instead, explain why the category is relevant to this patient's presentation, diagnostic considerations, or clinically relevant comorbidities.

When several test categories address the same clinical concern, group them into one concise rationale rather than explaining each category separately.
Do not provide a one-by-one purpose or definition for every recommended test category.
Prioritize the most clinically important reasons and explain test categories collectively when possible.
Only give an individual rationale for a test category when its relevance would otherwise be unclear from the clinical context.

Mention each final recommended test_tag no more than once.
Do not restate the same test category using synonymous wording.
Avoid repetitive explanations of the same clinical concern.

Do not mention specific individual laboratory test names from external references unless directly necessary.
Explain recommendations at the provided test_tag category level.

Similar prior emergency visits and external laboratory reference information are supporting evidence only.
Mention them only when they provide a clinically meaningful reason that cannot be adequately explained from the patient's current presentation and diagnostic context.
Do not mention evidence sources merely to demonstrate that they were used.

If a test_tag was added after patient-specific reassessment or evidence completion, briefly explain its inclusion when clinically meaningful.
If direct patient-specific support is limited, state that it is supported by related diagnostic context or similar prior emergency visits without overstating certainty.

Do not end with generic statements about management, clinical utility, or the overall necessity of testing.
Every sentence must add patient-specific diagnostic or test-recommendation reasoning.
Do not use generic concluding phrases such as:
"These tests are necessary to assess the patient's current presentation."
"These tests are necessary to guide further management."
"These tests are necessary to inform treatment and management."
"These tests will help guide clinical management."

Patient context:
${patient_context}

Working diagnostic categories:
${working_diagnoses}

First-pass selected test_tags:
${first_pass_tags}

Final recommended test_tags:
${final_tags}

Added after patient-specific reassessment:
${reassessment_added_tags}

Removed after patient-specific reassessment:
${reassessment_removed_tags}

Added by high-confidence evidence completion:
${evidence_completion_added_tags}

Evidence summary:
${evidence_summary}

Output JSON schema:
{
  "final_explanation": "one concise, patient-specific clinician-facing rationale of no more than 120 words"
}
"""
)


JSON_REPAIR_PROMPT = Template(
    """The previous response was invalid JSON or did not follow the required JSON schema. Return corrected JSON only, with no markdown and no extra text.
Use this exact schema:
{
  "recommended_tags": [
    {
      "test_tag": "one exact string from allowed_test_tags",
      "rationale": "brief clinical reason",
      "linked_icd_codes": ["ICD code strings"]
    }
  ],
  "final_explanation": "short paragraph"
}

allowed_test_tags:
${allowed_tags}

Original task:
${original_prompt}

Invalid previous response:
${invalid_response}"""
)


INVALID_TAG_REPAIR_PROMPT = Template(
    """${original_prompt}

Previous response used invalid tags: ${invalid_tags}. Return corrected JSON using only allowed_test_tags."""
)


def render_prompt(template: Template, /, **values: object) -> str:
    """Render a prompt template and fail if a required value is missing."""

    return template.substitute({key: str(value) for key, value in values.items()})


__all__ = [
    "FINAL_RATIONALE_PROMPT",
    "INVALID_TAG_REPAIR_PROMPT",
    "JSON_REPAIR_PROMPT",
    "KERNEL_FIRST_PASS_PROMPT",
    "REASSESSMENT_PROMPT",
    "SYSTEM_PROMPT",
    "render_prompt",
]
