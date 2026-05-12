# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Extended to extract entity metadata for latent trace analysis.

import json
import re

PROPER_NOUNS = {"Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"}


def extract_concept_from_conclusion(sentence):
    """Extract concept name from a conclusion sentence like 'Fae is a zumpus.'
    or 'Fae is an impus.' Returns None for property sentences like 'Fae is opaque.'"""
    m = re.match(r".+ is (?:a |an )(\w+)\.", sentence)
    if m:
        return m.group(1)
    return None


def extract_entities_and_graph(question, chain_of_thought, query):
    """Extract entity list, graph edges, root, target, and neg_target from a
    ProntoQA example.

    The chain_of_thought alternates:
      [0] Subject is a concept0.           (starting fact)
      [1] Concept0s are concept1s.          (rule)
      [2] Subject is a concept1.            (conclusion)
      ...
      [9] ConceptNs are [not] property.     (property rule)
      [10] Subject is [not] property.       (final conclusion)

    We extract:
      - All concept names appearing in the ontology
      - The proper noun subject (root)
      - The chain of concepts traversed in reasoning
      - A graph of "is-a" edges from the question sentences
    """
    # --- Extract the subject (root) from the query ---
    # Query format: "True or false: <Subject> is [not] <property>."
    query_match = re.match(r"True or false: (\w+) is", query)
    root_name = query_match.group(1) if query_match else None

    # --- Extract chain concepts from even-indexed CoT entries ---
    # These are the conclusion sentences: "Subject is a <concept>."
    chain_concepts = []
    for i in range(0, len(chain_of_thought), 2):
        concept = extract_concept_from_conclusion(chain_of_thought[i])
        if concept:
            chain_concepts.append(concept)

    # --- Extract all concept names from the question text ---
    # Parse "is a/an <concept>" and "are <plural>" patterns.
    # All concept names are lowercased to avoid sentence-initial caps duplication.
    concepts = set()

    def singularize(plural):
        """Convert ProntoQA plural to singular, lowercased."""
        p = plural.lower()
        if p.endswith("uses"):
            return p[:-2]       # "vumpuses" -> "vumpus"
        elif p.endswith("les"):
            return p[:-1]       # "timples" -> "timple"
        elif p.endswith("ies"):
            return p[:-3] + "y" # "dragonflies" -> "dragonfly"
        elif p.endswith("es"):
            return p[:-2]       # edge case
        elif p.endswith("s"):
            return p[:-1]
        return p

    # "is a/an <word>" patterns
    for m in re.finditer(r"is (?:a |an )(\w+)", question):
        word = m.group(1).lower()
        if word.capitalize() not in PROPER_NOUNS:
            concepts.add(word)

    # "Every/Each <word> is" patterns (singular concept names)
    for m in re.finditer(r"(?:Every|Each) (\w+) is", question):
        concepts.add(m.group(1).lower())

    # "<Plural> are" patterns — extract the root form
    for m in re.finditer(r"(\w+(?:es|s)) are ", question):
        singular = singularize(m.group(1))
        if singular.capitalize() not in PROPER_NOUNS:
            concepts.add(singular)

    # Also add chain concepts (they're definitely valid)
    concepts.update(chain_concepts)

    # --- Build idx_to_symbol ---
    # Put root first, then chain concepts in order, then remaining concepts
    seen = set()
    idx_to_symbol = []

    # Root proper noun first
    if root_name:
        idx_to_symbol.append(root_name)
        seen.add(root_name)

    # Chain concepts in reasoning order
    for c in chain_concepts:
        if c not in seen:
            idx_to_symbol.append(c)
            seen.add(c)

    # Remaining concepts (sorted for determinism)
    for c in sorted(concepts - seen):
        idx_to_symbol.append(c)
        seen.add(c)

    symbol_to_idx = {s: i for i, s in enumerate(idx_to_symbol)}

    # --- Build edges from question ---
    # "Every/Each X is a Y" -> edge from X to Y
    edges = []
    for m in re.finditer(r"(?:Every|Each) (\w+) is (?:a |an )(\w+)", question):
        src, dst = m.group(1).lower(), m.group(2).lower()
        if src in symbol_to_idx and dst in symbol_to_idx:
            edges.append([symbol_to_idx[src], symbol_to_idx[dst]])

    # "Xs are Ys" -> edge from X to Y (need to singularize both)
    for m in re.finditer(r"(\w+(?:es|s)) are (\w+(?:es|s))", question):
        src = singularize(m.group(1))
        dst = singularize(m.group(2))
        if src in symbol_to_idx and dst in symbol_to_idx:
            edges.append([symbol_to_idx[src], symbol_to_idx[dst]])

    # --- Identify root and target indices ---
    root_idx = symbol_to_idx.get(root_name, 0)

    # Target: last concept in the chain (the one just before the property step)
    target_idx = symbol_to_idx.get(chain_concepts[-1], -1) if chain_concepts else -1

    # Neg target: pick the first concept NOT on the chain (a distractor)
    chain_set = set(chain_concepts)
    neg_target_idx = -1
    for c in idx_to_symbol:
        if c not in chain_set and c not in PROPER_NOUNS:
            neg_target_idx = symbol_to_idx[c]
            break

    return {
        "idx_to_symbol": idx_to_symbol,
        "edges": edges,
        "root": root_idx,
        "target": target_idx,
        "neg_target": neg_target_idx,
        "chain_concepts": chain_concepts,
    }


def process_file(input_path, train_path, valid_path, test_path):
    file = json.load(open(input_path))
    data = []

    for k, v in file.items():
        example = v["test_example"]

        # Original fields (compatible with dataset.py)
        steps = [
            " ".join(example["chain_of_thought"][i : i + 2])
            for i in range(0, len(example["chain_of_thought"]), 2)
        ]

        # Entity metadata for analysis
        meta = extract_entities_and_graph(
            example["question"],
            example["chain_of_thought"],
            example["query"],
        )

        data.append(
            {
                "question": example["question"] + " " + example["query"],
                "steps": steps,
                "answer": example["answer"],
                "idx_to_symbol": meta["idx_to_symbol"],
                "edges": meta["edges"],
                "root": meta["root"],
                "target": meta["target"],
                "neg_target": meta["neg_target"],
                "chain_concepts": meta["chain_concepts"],
            }
        )

    json.dump(data[:9000], open(train_path, "w"))
    json.dump(data[9000:9200], open(valid_path, "w"))
    json.dump(data[9200:], open(test_path, "w"))
    print(f"Processed {len(data)} examples: {len(data[:9000])} train, "
          f"{len(data[9000:9200])} valid, {len(data[9200:])} test")


if __name__ == "__main__":
    process_file(
        "data/5hop_0shot_random.json",
        "data/prontoqa_train.json",
        "data/prontoqa_valid.json",
        "data/prontoqa_test.json",
    )
