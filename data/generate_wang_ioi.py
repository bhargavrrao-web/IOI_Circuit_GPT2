# ======================================================================
# Generate the ORIGINAL IOI dataset of Wang et al. (ICLR 2023) with THEIR OWN code.
# Source: https://github.com/redwoodresearch/Easy-Transformer  (file easy_transformer/ioi_dataset.py, MIT licence)
#
# In Colab, first run:   !git clone https://github.com/redwoodresearch/Easy-Transformer
# then run this file / cell. Nothing from the original code is modified; we only call its functions.
# ======================================================================
import importlib.util, os, random
import numpy as np, pandas as pd

REPO = os.environ.get("WANG_REPO", "Easy-Transformer")          # folder created by git clone
N = int(os.environ.get("WANG_N", 1000)); SEED = 0
spec = importlib.util.spec_from_file_location("ioi_dataset", f"{REPO}/easy_transformer/ioi_dataset.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)     # their module (no tokenizer needed for these functions)

random.seed(SEED); np.random.seed(SEED)                                  # their code uses the global RNGs
templates = m.BABA_TEMPLATES + m.ABBA_TEMPLATES                          # 15 templates x both orders = 30, as in the paper
nouns = {"[PLACE]": m.PLACES, "[OBJECT]": m.OBJECTS}

# p_IOI: clean prompts (the repeated name S appears twice; the answer is IO)
p_ioi = m.gen_prompt_uniform(templates, m.NAMES, nouns, N=N, symmetric=False)

# p_ABC: exactly how their completeness/minimality scripts build it: replace IO, S and S1 by random names
p_abc = p_ioi
for flip in [("IO", "RAND"), ("S", "RAND"), ("S1", "RAND")]:
    p_abc = m.gen_flipped_prompts(p_abc, m.NAMES, flip)

def rows(prompts):
    out = []
    for i, p in enumerate(prompts):
        text = p["text"]
        out.append(dict(id=i + 1, template_idx=p["TEMPLATE_IDX"], order="BABA" if p["TEMPLATE_IDX"] < len(m.BABA_TEMPLATES) else "ABBA",
                        full_sentence=text, model_input=" ".join(text.split(" ")[:-1]),       # the model sees the sentence WITHOUT the last word
                        correct_answer=" " + p["IO"], repeated_name=" " + p["S"], place=p.get("[PLACE]"), object=p.get("[OBJECT]")))
    return pd.DataFrame(out)

os.makedirs("out", exist_ok=True)
rows(p_ioi).to_csv("out/wang_p_IOI.csv", index=False); rows(p_abc).to_csv("out/wang_p_ABC.csv", index=False)
pd.DataFrame(dict(type="name", word=m.NAMES)).to_csv("out/wang_names.csv", index=False)
print(f"saved {len(p_ioi)} p_IOI prompts and {len(p_abc)} p_ABC prompts to ./out  | names={len(m.NAMES)} places={len(m.PLACES)} objects={len(m.OBJECTS)} templates={len(templates)}")
