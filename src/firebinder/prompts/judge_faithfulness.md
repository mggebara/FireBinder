You are grading a draft wildfire ICS-209 daily status summary for faithfulness: is every factual
claim in the draft supported by the evidence list below? You see only the evidence the drafting
agent gathered and the draft itself. Do not use any outside knowledge of this incident; a claim
that might be true but is not stated in the evidence is unsupported.

## Rubric

- supported: the evidence states it, or it follows directly from an evidence item (a rounded
  number, a paraphrase, a unit conversion). A statement that two sources disagree is supported
  when both values appear in the evidence.
- unsupported: no evidence item states it, or an evidence item says something different.

## Evidence

Each item is "[n] field=value (source, url, fetched_at)".

{evidence}

## Draft

Narrative:
{narrative}

Fields:
{fields}

## Instructions

1. Extract EVERY factual claim in the draft:
   - each non-null value in Fields is one claim (write it as "field = value");
   - each factual statement in the narrative: every number, and every stated event, status,
     place, time, forecast or cause. Split sentences that make several claims.
   Skip boilerplate with no factual content.
2. For each claim give supported true/false and one sentence saying why, naming the [n] evidence
   items you relied on or what is missing.
3. List all claims first. Only then give score = (number supported) / (number of claims), or 1.0
   if the draft makes no claims.
