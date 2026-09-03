"""
Phase 2 starter: download & format RSVQA-LR for fine-tuning.

Not run yet in the prototype phase — this is here so Phase 2 has a clear
starting point. RSVQA-LR is small (~772 images) and a good first fine-tuning
target on free-tier Colab.

Dataset info: https://rsvqa.sylvainlobry.com/
(Download requires manually grabbing the LR split's images.zip and the
question/answer JSON files from the site above — there's no clean pip/CLI
installer for it.)

Usage (once you have the raw files):
    python data/prepare_rsvqa.py --raw_dir ./raw/rsvqa_lr --out_dir ./data/rsvqa_lr_processed
"""

import argparse
import json
import os


def build_qa_pairs(questions_json_path: str, answers_json_path: str) -> list[dict]:
    with open(questions_json_path) as f:
        questions = json.load(f)["questions"]
    with open(answers_json_path) as f:
        answers = json.load(f)["answers"]

    answers_by_qid = {a["question_id"]: a["answer"] for a in answers}

    pairs = []
    for q in questions:
        qid = q["id"]
        if qid not in answers_by_qid:
            continue
        pairs.append({
            "image_id": q["img_id"],
            "question": q["question"],
            "answer": answers_by_qid[qid],
            "type": q.get("type", "unknown"),
        })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Directory with RSVQA-LR raw JSON/images")
    parser.add_argument("--out_dir", required=True, help="Where to write the processed jsonl")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        q_path = os.path.join(args.raw_dir, f"LR_split_{split}_questions.json")
        a_path = os.path.join(args.raw_dir, f"LR_split_{split}_answers.json")
        if not (os.path.exists(q_path) and os.path.exists(a_path)):
            print(f"Skipping {split}: files not found at {q_path} / {a_path}")
            continue

        pairs = build_qa_pairs(q_path, a_path)
        out_path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(out_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Wrote {len(pairs)} examples to {out_path}")


if __name__ == "__main__":
    main()
