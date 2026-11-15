set -euo pipefail
cp -R "$1/paper-stub/." .
printf '%s\n' '' '@inproceedings{zheng2023mtbench,' '  title     = {Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},' '  author    = {Zheng, Lianmin and others},' '  booktitle = {NeurIPS},' '  year      = {2023}' '}' '' '@inproceedings{wang2023selfconsistency,' '  title     = {Self-Consistency Improves Chain of Thought Reasoning in Language Models},' '  author    = {Wang, Xuezhi and others},' '  booktitle = {ICLR}' '}' >> paper/refs.bib
