import json
import pandas as pd

BASE = "/root/autodl-tmp/gen-depth-contamination"

cases = [
    {
        'label': 'A',
        'title': 'C4 Web Content (Stylistic Convergence)',
        'doc_id': 2468,
        'base': f'{BASE}/data_exp016_qwen_base',
        'desc': 'A vacation rental listing (depth~0) with fragmented promotional phrases converges to generic real-estate prose by depth~1. Depths~2 and~3 produce near-identical property descriptions, illustrating feature saturation consistent with $K^*{=}2$.',
    },
    {
        'label': 'B',
        'title': 'C4 Encyclopedic (Factual Degradation)',
        'doc_id': 3000,
        'base': f'{BASE}/data_exp016_qwen_base',
        'desc': 'A factual description of a historical stone (depth~0) degrades into a hallucinated Glastonbury legend by depth~1. Subsequent depths elaborate on the fabricated narrative with decreasing lexical diversity, again saturating after the first generation step.',
    },
    {
        'label': 'C',
        'title': 'arXiv (Domain-Specific Collapse)',
        'doc_id': 1079,
        'base': f'{BASE}/data/exp_024_arxiv',
        'desc': 'An ML abstract on multiplex graphs (depth~0) retains topical coherence at depth~1 but collapses into hallucinated acknowledgement boilerplate by depth~2. Raw \\LaTeX{} commands (e.g., \\texttt{\\textbackslash cite}, \\texttt{\\textbackslash section*}) appear in the generated text, reflecting the model\'s tendency to reproduce structural templates rather than substantive content.',
    },
]

def escape_latex(s):
    """Escape for use inside \textit{} blocks."""
    replacements = [
        ('\\', 'BACKSLASH_PLACEHOLDER'),
        ('&', '\\&'),
        ('%', '\\%'),
        ('$', '\\$'),
        ('#', '\\#'),
        ('_', '\\_'),
        ('{', '\\{'),
        ('}', '\\}'),
        ('~', '\\textasciitilde{}'),
        ('^', '\\textasciicircum{}'),
        ('BACKSLASH_PLACEHOLDER', '\\textbackslash{}'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s

def get_text(path, doc_id, max_words=55):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d['doc_id'] == doc_id:
                words = d['text'].split()[:max_words]
                result = ' '.join(words)
                if len(words) < len(d['text'].split()):
                    result += ' [...]'
                return result
    return ''

def get_features(csv_path, doc_id):
    df = pd.read_csv(csv_path)
    doc = df[df['doc_id'] == doc_id].sort_values('depth')
    feats = {}
    for _, r in doc[doc['depth'] <= 3].iterrows():
        d = int(r['depth'])
        feats[d] = {
            'ttr': r['ttr'],
            'surp_mean': r['surp_mean'],
            'freq_entropy': r['freq_entropy'],
        }
    return feats

L = []  # lines

L.append(r'\subsection{Qualitative Case Study}')
L.append(r'\label{app:case-study}')
L.append('')
L.append(r'We present three representative document chains from our depth-generation pipeline, illustrating how text properties evolve across generation depths 0--3.')
L.append(r'Depth~0 is the original human-written text; each subsequent depth is produced by prompting the generator (Qwen-1.5B) with the text from the preceding depth.')
L.append(r'For each case, we report three key features from our 15-dimensional OBD feature set: type-token ratio (TTR), mean surprisal, and frequency entropy.')
L.append('')

for case in cases:
    doc_id = case['doc_id']
    feats = get_features(f"{case['base']}/features.csv", doc_id)
    
    L.append(r'\paragraph{Case ' + case['label'] + ': ' + case['title'] + '.}')
    L.append(case['desc'])
    L.append('')
    
    # Feature table
    L.append(r'\begin{center}')
    L.append(r'{\footnotesize')
    L.append(r'\begin{tabular}{lcccc}')
    L.append(r'\toprule')
    L.append(r'& \textbf{Depth 0} & \textbf{Depth 1} & \textbf{Depth 2} & \textbf{Depth 3} \\')
    L.append(r'\midrule')
    
    for fname, display in [('ttr', 'TTR'), ('surp_mean', 'Mean Surprisal'), ('freq_entropy', 'Freq. Entropy')]:
        vals = []
        raw_vals = [feats[d][fname] for d in range(4)]
        for i, v in enumerate(raw_vals):
            s = f'{v:.3f}'
            # Bold the depth-0 value if it's notably different from depth 2-3
            if i == 0 and abs(raw_vals[0] - raw_vals[2]) > 0.3:
                s = r'\textbf{' + s + '}'
            vals.append(s)
        L.append(f'{display} & ' + ' & '.join(vals) + r' \\')
    
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}}')
    L.append(r'\end{center}')
    L.append('')
    
    # Text excerpts in a compact format
    for depth in range(4):
        text = get_text(f"{case['base']}/depth_{depth}.jsonl", doc_id, max_words=50)
        escaped = escape_latex(text)
        
        dlabel = f'Depth~{depth}'
        if depth == 0:
            dlabel += ' (human)'
        
        L.append(r'\noindent\textsc{' + dlabel + r':} {\small\itshape ' + escaped + r'}')
        if depth < 3:
            L.append(r'\smallskip')
        L.append('')
    
    L.append(r'\medskip')
    L.append('')

latex = '\n'.join(L)
out_path = f"{BASE}/artifacts/case_study_qualitative.tex"
with open(out_path, 'w') as f:
    f.write(latex)

print(f"Written {len(L)} lines to {out_path}")
