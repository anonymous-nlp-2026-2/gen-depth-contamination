"""exp-027: Strip LaTeX markup from arXiv depth-chain data."""
import json
import re
import sys
from pathlib import Path

def strip_latex(text: str) -> str:
    """Remove LaTeX commands from text, keeping readable content."""
    s = text

    # Remove \begin{env}...\end{env} for math/figure environments (keep content for text envs)
    for env in ['equation', 'equation*', 'align', 'align*', 'eqnarray', 'eqnarray*',
                'displaymath', 'math', 'figure', 'figure*', 'table', 'table*',
                'tikzpicture', 'array', 'matrix', 'pmatrix', 'bmatrix']:
        s = re.sub(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}', ' ', s, flags=re.DOTALL)

    # Remove display math $$...$$ and \[...\]
    s = re.sub(r'\$\$.*?\$\$', ' ', s, flags=re.DOTALL)
    s = re.sub(r'\\\[.*?\\\]', ' ', s, flags=re.DOTALL)

    # Remove inline math $...$
    s = re.sub(r'\$[^$]+?\$', ' ', s)

    # Remove \cite{...}, \ref{...}, \label{...}, \eqref{...}, \cref{...}
    s = re.sub(r'\\(?:cite|citep|citet|ref|eqref|label|cref|Cref|pageref|autoref)\{[^}]*\}', '', s)

    # Remove \url{...} and \href{...}{text} -> keep text
    s = re.sub(r'\\url\{[^}]*\}', '', s)
    s = re.sub(r'\\href\{[^}]*\}\{([^}]*)\}', r'\1', s)

    # \textbf{x}, \textit{x}, \emph{x}, \underline{x}, etc -> keep x
    s = re.sub(r'\\(?:textbf|textit|texttt|textrm|textsf|textsc|emph|underline|textcolor\{[^}]*\})\{([^}]*)\}', r'\1', s)

    # \section{x}, \subsection{x}, etc -> keep x
    s = re.sub(r'\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]*)\}', r'\1', s)

    # \footnote{x} -> remove
    s = re.sub(r'\\footnote\{[^}]*\}', '', s)

    # \caption{x} -> keep x
    s = re.sub(r'\\caption\{([^}]*)\}', r'\1', s)

    # Generic \command{...} -> keep content inside braces
    s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)

    # Remove remaining \commands (no braces)
    s = re.sub(r'\\[a-zA-Z]+\*?', '', s)

    # Escape sequences: \\, \&, \%, \#, \$, \_, \{, \}
    s = s.replace('\\&', '&').replace('\\%', '%').replace('\\#', '#')
    s = s.replace('\\$', '$').replace('\\_', '_')
    s = s.replace('\\{', '{').replace('\\}', '}')
    s = s.replace('\\\\', ' ')

    # Remove empty braces
    s = re.sub(r'\{\s*\}', '', s)
    # Remove lone braces
    s = s.replace('{', '').replace('}', '')

    # Collapse whitespace
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = s.strip()

    return s


def main():
    src_dir = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_024_arxiv")
    dst_dir = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_027_latex_ablation")
    dst_dir.mkdir(parents=True, exist_ok=True)

    for d in range(6):
        src_path = src_dir / f"depth_{d}.jsonl"
        dst_path = dst_dir / f"depth_{d}.jsonl"

        records = []
        with open(src_path) as f:
            for line in f:
                rec = json.loads(line)
                rec["text"] = strip_latex(rec["text"])
                records.append(rec)

        with open(dst_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Stats
        orig_lens = []
        stripped_lens = []
        with open(src_path) as f:
            for line in f:
                orig_lens.append(len(json.loads(line)["text"]))
        for rec in records:
            stripped_lens.append(len(rec["text"]))

        import numpy as np
        orig_mean = np.mean(orig_lens)
        strip_mean = np.mean(stripped_lens)
        reduction = 1 - strip_mean / orig_mean
        print(f"depth_{d}: {len(records)} docs, avg len {orig_mean:.0f} -> {strip_mean:.0f} ({reduction:.1%} reduction)")

    print(f"\nStripped data saved to {dst_dir}")


if __name__ == "__main__":
    main()
