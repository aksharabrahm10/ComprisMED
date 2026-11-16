"""Rebuild the authored synthetic benchmark from its scenario definitions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUT = Path(__file__).resolve().with_name("benchmark.jsonl")

def complexity_level(source: str) -> str:
    word_count = len(re.findall(r"\b[\w'-]+\b", source))
    conditional = re.search(
        r"\b(?:if|unless|then|while|after|before)\b", source, flags=re.IGNORECASE
    )
    if word_count >= 38 or (conditional and word_count >= 28):
        return "complex"
    if word_count >= 22 or conditional:
        return "moderate"
    return "simple"


# Each family stays wholly in one split so the same scenario template never crosses
# train, validation, and test. Texts and fictional medicine labels are authored here.
FAMILIES: dict[str, dict[str, object]] = {
    "scheduled_oral": {
        "split": "train",
        "instruction_type": "scheduled oral medicine",
        "cases": [
            ("Take Sampletab A 5 mg by mouth once daily for 7 days. Do not take with Sample Juice A.", "negation", "Do not", "Please", ["restriction", "negation"]),
            ("Take Sampletab B 10 mg by mouth every morning for 14 days. Avoid driving if you feel dizzy.", "number_or_dose", "10 mg", "15 mg", ["warning", "restriction"]),
            ("Take 1 Sampletab C 25 mg capsule twice daily for 5 days. Do not take an extra capsule.", "number_or_dose", "25 mg", "30 mg", ["negation"]),
            ("Take Sampletab D 2 tablets at bedtime for 3 weeks. Call the clinic if the rash spreads.", "number_or_dose", "2 tablets", "3 tablets", ["warning", "follow_up"]),
            ("Take Sampletab E 7.5 mg by mouth every 12 hours for 4 days. Do not take it with Sample Food A.", "frequency", "every 12 hours", "every 8 hours", ["restriction", "negation"]),
        ],
    },
    "respiratory_device": {
        "split": "train",
        "instruction_type": "inhaler and respiratory device",
        "cases": [
            ("Use Sample Inhaler A for 2 puffs every 6 hours as needed for fictional wheezing. Do not exceed 8 puffs per day.", "number_or_dose", "2 puffs", "3 puffs", ["warning", "negation"]),
            ("Use Sample Inhaler B for 1 puff every 4 hours. Rinse your mouth after each use; call the clinic on 06/18/2026 if breathing worsens.", "date", "06/18/2026", "06/19/2026", ["follow_up", "warning"]),
            ("Use Sample Nebulizer A for 3 treatments daily for 5 days. Do not share the device.", "negation", "Do not", "Please", ["negation"]),
            ("Use Sample Inhaler C for 2 puffs in the morning and 2 puffs at bedtime. Avoid using it near Sample Chemical A.", "frequency", "at bedtime", "every morning", ["restriction"]),
            ("Use Sample Inhaler D for 1 puff every 8 hours as needed. If symptoms continue after 2 days, call the clinic.", "frequency", "every 8 hours", "every 6 hours", ["follow_up"]),
        ],
    },
    "topical_wound_care": {
        "split": "train",
        "instruction_type": "topical medicine and wound care",
        "cases": [
            ("Apply Sample Cream A 1 time daily to the marked area for 6 days. Do not apply near the eyes.", "negation", "Do not", "Please", ["warning", "negation"]),
            ("Put Sample Ointment B on the dressing every 24 hours for 3 days. Call the clinic on July 2, 2026 if redness spreads.", "date", "July 2, 2026", "July 3, 2026", ["follow_up"]),
            ("Clean the fictional wound with Sample Wash A twice daily, then cover it. Avoid soaking the area for 10 days.", "duration", "10 days", "12 days", ["restriction"]),
            ("Apply 2 drops of Sample Solution A to the marked skin each night for 4 weeks. Do not swallow it.", "number_or_dose", "2 drops", "3 drops", ["negation", "warning"]),
            ("Change the sample dressing every 12 hours for 2 days. If bleeding does not stop, call the clinic immediately.", "frequency", "every 12 hours", "every 8 hours", ["follow_up", "warning"]),
        ],
    },
    "eye_ear_care": {
        "split": "train",
        "instruction_type": "eye or ear drops",
        "cases": [
            ("Place 2 drops of Sample Eye Solution A in the left eye twice daily for 7 days. Do not put drops in the right eye.", "negation", "Do not", "Please", ["restriction", "negation"]),
            ("Place 1 drop of Sample Eye Solution B in each eye every 8 hours for 5 days. Keep the bottle tip away from your eye.", "duration", "5 days", "6 days", ["warning"]),
            ("Use 3 drops of Sample Ear Solution A in the left ear at bedtime for 4 days. Do not use it in either eye.", "frequency", "at bedtime", "every morning", ["restriction", "negation"]),
            ("Use Sample Ear Solution B, 4 drops in the right ear twice a day until August 9, 2026. Call if pain gets worse.", "date", "August 9, 2026", "August 10, 2026", ["follow_up"]),
            ("After putting in 2 drops of Sample Eye Solution C, close the eye gently for 1 minute. Repeat every 6 hours for 3 days.", "duration", "3 days", "5 days", ["follow_up"]),
        ],
    },
    "as_needed_symptoms": {
        "split": "validation",
        "instruction_type": "as-needed symptom instruction",
        "cases": [
            ("Take 1 Sampletab F 200 mg every 8 hours as needed for fictional pain. Do not take more than 3 tablets in 24 hours.", "number_or_dose", "200 mg", "300 mg", ["warning", "negation"]),
            ("Use 1 puff of Sample Inhaler E every 4 hours as needed for fictional chest tightness. If it does not help after 2 uses, call the clinic.", "frequency", "every 4 hours", "every 6 hours", ["follow_up"]),
            ("Take Sampletab G 5 mg only when needed for fictional nausea, no more than twice daily. Avoid driving if sleepy.", "number_or_dose", "5 mg", "7 mg", ["restriction", "warning"]),
            ("Use 5 mL of Sample Liquid A every 6 hours as needed for fictional cough. Do not use it for more than 3 days.", "duration", "3 days", "5 days", ["negation"]),
            ("Apply Sample Gel A 1 time daily as needed for fictional itching. Stop and call the clinic if swelling appears.", "number_or_dose", "1", "2", ["follow_up", "warning"]),
        ],
    },
    "multi_day_course": {
        "split": "validation",
        "instruction_type": "multi-day course and completion",
        "cases": [
            ("Take Sampletab H 250 mg twice daily for 8 days. Finish the full course unless your clinician tells you otherwise.", "duration", "8 days", "10 days", ["warning"]),
            ("Take 1 Samplecap A 125 mg every 12 hours for 6 days. Do not double a dose you miss; call the clinic on 09/14/2026 if vomiting continues.", "negation", "Do not", "Please", ["negation", "follow_up"]),
            ("Use Sample Rinse A 10 mL three times daily for 4 days. Spit it out; do not swallow.", "number_or_dose", "10 mL", "15 mL", ["warning", "negation"]),
            ("Take Sampletab I 1 tablet every 6 hours for 5 days with food. Call the clinic if a new rash appears.", "frequency", "every 6 hours", "every 8 hours", ["warning", "follow_up"]),
            ("Take Sampletab J 50 mg once daily for 9 days. Avoid Sample Food B during the course.", "duration", "9 days", "11 days", ["restriction"]),
        ],
    },
    "follow_up_monitoring": {
        "split": "test",
        "instruction_type": "monitoring and follow-up",
        "cases": [
            ("Record the fictional measurement each morning for 6 days. Call the clinic on October 5, 2026 if any reading is outside the written range.", "date", "October 5, 2026", "October 6, 2026", ["follow_up", "warning"]),
            ("Check the sample device reading twice daily for 2 weeks and bring the log to your visit on 11/12/2026.", "duration", "2 weeks", "3 weeks", ["follow_up"]),
            ("Weigh the sample object once daily for 5 days. Do not change any medicine based on this log; call the clinic if the number changes suddenly.", "frequency", "once daily", "twice daily", ["restriction", "negation", "follow_up"]),
            ("Use the sample monitor at bedtime every day until the follow-up on December 4, 2026. Contact the clinic if it shows an error.", "frequency", "at bedtime", "each morning", ["follow_up"]),
            ("Write down the fictional symptom score every 4 hours while awake for 3 days. If it reaches 8, call the clinic; do not wait for the next visit.", "number_or_dose", "8", "9", ["follow_up", "negation"]),
        ],
    },
    "restrictions_safety": {
        "split": "test",
        "instruction_type": "restriction or safety warning",
        "cases": [
            ("Do not drive for 12 hours after using Sample Medicine K 1 tablet. Ask another adult to help with travel.", "negation", "Do not", "Please", ["restriction", "negation"]),
            ("Take Sampletab L 10 mg at bedtime for 7 days. Avoid alcohol and call the clinic on 01/08/2027 if you feel unusually sleepy.", "date", "01/08/2027", "01/09/2027", ["restriction", "follow_up"]),
            ("Keep Sample Medicine M 5 mL in its original container at room temperature for 30 days. Do not freeze it.", "number_or_dose", "5 mL", "10 mL", ["restriction", "negation"]),
            ("Take 1 Sampletab N 20 mg each morning for 10 days. Do not take it with Sample Supplement A; ask the clinic before restarting.", "duration", "10 days", "14 days", ["restriction", "follow_up", "negation"]),
            ("Store Sample Drops D upright and use 2 drops twice daily. Do not use after March 3, 2027.", "date", "March 3, 2027", "March 4, 2027", ["restriction", "negation"]),
        ],
    },
    "taper_schedule": {
        "split": "test",
        "instruction_type": "multi-step taper schedule",
        "cases": [
            ("For the fictional taper, take Sampletab O 20 mg each morning for 3 days, then 10 mg each morning for 3 days. Do not skip a step.", "number_or_dose", "20 mg", "25 mg", ["negation"]),
            ("Days 1 to 4: take Sampletab P 15 mg daily. Days 5 to 8: take 10 mg daily. Call the clinic on April 6, 2027 with questions.", "date", "April 6, 2027", "April 7, 2027", ["follow_up"]),
            ("Take Sampletab Q 2 tablets daily for 2 days, then 1 tablet daily for 5 days, then stop. Do not restart without calling the clinic.", "number_or_dose", "2 tablets", "3 tablets", ["negation", "follow_up"]),
            ("Use Sample Patch A for 7 days, then remove it for 2 days before starting the next patch. Do not cut the patch.", "duration", "7 days", "8 days", ["restriction", "negation"]),
            ("Take Sampletab R 8 mg every 12 hours for 4 days, then once daily for 4 days. Contact the clinic if you miss both doses in one day.", "frequency", "every 12 hours", "every 8 hours", ["follow_up"]),
        ],
    },
    "injection_device": {
        "split": "test",
        "instruction_type": "fictional injection or device use",
        "cases": [
            ("Use 1 marked unit of Sample Device A once daily for 5 days. Do not share the device or needle.", "negation", "Do not", "Please", ["restriction", "negation"]),
            ("Inject the fictional Sample Solution B 0.5 mL every 24 hours for 4 days, using a new sample needle each time.", "duration", "4 days", "6 days", ["warning"]),
            ("Use Sample Device C at 8:00 AM on May 8, 2027 and May 15, 2027. Call the clinic if the indicator turns red.", "date", "May 8, 2027", "May 9, 2027", ["follow_up"]),
            ("Take 2 readings with Sample Device D every morning for 7 days. Do not adjust its settings; bring it to the visit.", "number_or_dose", "2 readings", "3 readings", ["restriction", "follow_up"]),
            ("Apply Sample Device E for 15 minutes twice daily for 3 days. Stop using it if the skin becomes irritated.", "duration", "3 days", "5 days", ["warning", "follow_up"]),
        ],
    },
    "conditional_multistep": {
        "split": "test",
        "instruction_type": "conditional and multi-step instructions",
        "cases": [
            ("If the fictional reading is above 180, wait 15 minutes and check again. If it remains high twice, call the clinic; do not take extra Sampletab S.", "negation", "do not", "please", ["conditional", "follow_up", "negation"]),
            ("Take Sampletab T 5 mg with breakfast for 6 days. If you miss breakfast, take it with lunch; do not take two doses together.", "duration", "6 days", "8 days", ["conditional", "negation"]),
            ("Use 1 drop of Sample Eye Solution D in the right eye at bedtime. If pain develops, stop and call the clinic within 24 hours.", "number_or_dose", "1 drop", "2 drops", ["conditional", "follow_up"]),
            ("For 10 days, rinse the sample device after each use and let it air dry. Do not put it in Sample Cleaner A.", "duration", "10 days", "14 days", ["restriction", "negation"]),
            ("Take Sampletab U 12 mg once daily for 5 days. If you develop a rash, do not take the next dose and call the clinic on June 2, 2027.", "date", "June 2, 2027", "June 3, 2027", ["conditional", "warning", "follow_up", "negation"]),
        ],
    },
    "date_sensitive_care": {
        "split": "test",
        "instruction_type": "date-sensitive care plan",
        "cases": [
            ("Use Sample Cream B twice daily through July 7, 2027, then stop. Do not continue past that date unless the clinic confirms.", "negation", "Do not", "Please", ["negation", "follow_up"]),
            ("Take Sampletab V 4 mg once daily for 8 days, starting June 10, 2027. Your check-in is June 19, 2027.", "date", "June 19, 2027", "June 20, 2027", ["follow_up"]),
            ("Use 2 sprays of Sample Nasal Solution A every morning until the appointment on 08/21/2027. Do not use it after the appointment.", "date", "08/21/2027", "08/22/2027", ["negation", "follow_up"]),
            ("Take Sampletab W 1 tablet every 12 hours for 3 days. Repeat the fictional test on September 6, 2027 and call with the result.", "date", "September 6, 2027", "September 7, 2027", ["follow_up"]),
            ("Apply Sample Patch B at 9:00 AM on October 1, 2027 and remove it after 24 hours. Do not apply a second patch before October 2.", "date", "October 2", "October 3", ["restriction", "negation"]),
        ],
    },
}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, definition in FAMILIES.items():
        split = str(definition["split"])
        instruction_type = str(definition["instruction_type"])
        for index, (source, mutation_category, target, replacement, context_tags) in enumerate(
            definition["cases"], start=1
        ):
            # Each scenario explicitly names the single salient detail used for
            # its controlled mutation; labels never come from the runtime parser.
            expected_facts = {mutation_category: [target]}
            mutation_output = re.sub(re.escape(target), replacement, source, count=1)
            if mutation_output == source or target not in source:
                raise ValueError(f"{family}_{index:02d}: mutation did not change source")
            rows.append(
                {
                    "id": f"{family}_{index:02d}",
                    "split": split,
                    "scenario_family": family,
                    "instruction_type": instruction_type,
                    "complexity": complexity_level(source),
                    "synthetic": True,
                    "source": source,
                    "required_facts": expected_facts,
                    "expected_extraction": expected_facts,
                    "annotation_method": (
                        "scenario-author-selected required detail, explicitly paired "
                        "with the controlled mutation; not generated by the extractor"
                    ),
                    "context_annotations": context_tags,
                    "context_verification_scope": (
                        "not_checked_by_current_deterministic_verifier"
                    ),
                    "expected_clean_outcome": "preserved",
                    "expected_clean_checks": [
                        {
                            "category": category,
                            "fact": fact,
                            "status": "preserved",
                        }
                        for category, facts in expected_facts.items()
                        for fact in facts
                    ],
                    "mutations": [
                        {
                            "label": "deliberately_altered",
                            "category": mutation_category,
                            "required_fact": target,
                            "replacement": replacement,
                            "output": mutation_output,
                            "expected_status": "flag",
                            "expected_verifier_outcome": "flag",
                        }
                    ],
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"])
        counts[split] = counts.get(split, 0) + 1
    print(f"Wrote {len(rows)} synthetic scenarios to {OUTPUT}")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
