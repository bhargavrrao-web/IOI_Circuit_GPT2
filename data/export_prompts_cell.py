# ======================================================================
# COLAB CELL: export the EXACT prompts used in the experiments to CSV.
# Run it AFTER the notebook's data cell (04_data) and model cell have run.
# It repeats the notebook's sampling step by step (same seed, same word filter),
# so the prompts are identical to the 100-per-task sets used for all results.
# ======================================================================
import random, os
import pandas as pd

rows, used_words = [], []
for name in TASKS:                                   # IOI (people) + the 7 new families
    spec = TASKS[name]
    rng = random.Random(SEED)                        # same seed as build_task
    pool = [w for w in spec["pool"] if is_single_token(" " + w)]      # words that are ONE GPT-2 token
    templates = spec["templates"] + [swap_roles(t) for t in spec["templates"]]
    used_words += [dict(task=name, word=w) for w in pool]
    print(f"{name:14s} words kept as single tokens: {len(pool):2d} of {len(spec['pool'])}")
    for i in range(N_PROMPTS):
        t = rng.choice(templates)
        io, s, a, b, c = rng.sample(pool, 5)         # io = correct, s = repeated, a/b/c = corrupted fillers
        used = {io, s, a, b, c}
        kw = dict(PLACE=rng.choice([p for p in PLACES if p not in used]),
                  OBJECT=rng.choice([o for o in OBJECTS if o not in used]))
        rows.append(dict(task=name, id=i + 1, template=t,
                         clean_prompt=t.format(IO=io, S1=s, S2=s, **kw),        # repeated entity appears twice
                         corrupted_prompt=t.format(IO=a, S1=b, S2=c, **kw),     # three different entities
                         correct_answer=" " + io, repeated_entity=" " + s))

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/prompts_all_tasks.csv", index=False)
pd.DataFrame(used_words).to_csv(f"{OUT_DIR}/word_lists_used.csv", index=False)
print(f"saved {len(rows)} prompts and the word lists to {OUT_DIR}/")
try:                                                  # on Colab: also download both files
    from google.colab import files
    files.download(f"{OUT_DIR}/prompts_all_tasks.csv"); files.download(f"{OUT_DIR}/word_lists_used.csv")
except ImportError:
    pass
